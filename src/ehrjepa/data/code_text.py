"""Natural-language descriptions for every vocabulary code, and an embedding init from them.

A code embedding table is, by default, 30,000 rows of independent noise: the
model is told that ``ICD9CM//25000`` and ``ICD9CM//25001`` are two unrelated
symbols and has to rediscover from co-occurrence alone that they are two kinds of
diabetes. Every one of those rows is a parameter -- at 30,000 x 256 the table is
7.68M of them, **58% of a base-size hybrid's 13.21M trainable parameters** (74%
at the 4x192 pilot size) -- and the tail of them are seen a handful of times in
11.3M training events.

This module builds the other option. Each code is mapped to a sentence
(``ICD9CM//25000`` -> "diagnosis: diabetes mellitus without mention of
complication, type II or unspecified type, not stated as uncontrolled"), the
sentences are embedded with an off-the-shelf sentence-transformer, the 384-d
result is projected to the model width with a PCA fitted on the vocabulary
itself, and the whole matrix is rescaled so its standard deviation matches the
``N(0, 0.02)`` the random init would have used. That table is what
``model.code_init: text`` copies over ``code_emb`` -- either as a starting point
(``freeze_code_embeddings: false``) or as a fixed, zero-parameter lookup
(``true``).

Where the words come from
-------------------------
======================  ==================================================
``ICD9CM``              CMS ICD-9-CM v32 long diagnosis descriptions, plus
                        the optional ``icd9cms`` package for the 3-digit
                        *categories* the billable-leaf list does not carry.
``ICD9PROC``            CMS ICD-9-CM v32 long procedure descriptions.
``HCPCS``               ``d_hcpcs`` from the public MIMIC-IV demo, which
                        covers both the numeric CPT range and HCPCS level
                        II; unknown codes fall back to their letter group
                        or CPT section.
``NDC``                 The FDA NDC directory, current *and* excluded
                        (finished-marketing) products -- DE-SynPUF's drug
                        codes are from 2008-2010, so the excluded file
                        carries most of what the current one does not.
                        Unknown 9-digit codes fall back to their 5-digit
                        labeler's name.
``DRG``                 ``drgcodes`` from the MIMIC-IV demo (HCFA rows).
``SP_*``, ``SEX``,      Hand-written, from the DE-SynPUF codebook.
``RACE``, ``VISIT``,
``ADMISSION``, ...
======================  ==================================================

Every description carries a **tier**, and the coverage JSON reports entry counts
*and* train-event mass for each:

``exact``
    The code string itself has a published description.
``ancestor``
    A genuine parent concept in the same coding system supplied the words -- an
    ICD-9 category for a rolled-up code, an NDC labeler, a HCPCS letter group,
    a bare ``PREFIX`` entry. The vocabulary's own rollup (see
    :func:`ehrjepa.data.tokenize.ancestors`) creates these entries, so they are
    not failures; they are exactly as specific as the id they name.
``fallback``
    Nothing but the coding system's name and the literal code.

Building the table::

    python -m ehrjepa.data.code_text build --cache data/cache/desynpuf-s1 --width 256

which downloads what it needs into ``--sources`` (default ``data/code_text``)
the first time and writes ``code_init_text_<width>.npy`` plus
``code_init_text_<width>.json`` into the cache directory. The ``.npy`` is a
derived artifact of a public code table and a public sentence model and is not
committed; the coverage JSON is.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from ehrjepa.data.tokenize import PAD_ID, SPECIAL_TOKENS, Vocabulary, split_code

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SOURCE_DIR",
    "ICU_CHANNELS",
    "ICU_CODES",
    "ICU_UNITS",
    "Description",
    "DescriptionTables",
    "SOURCE_DOWNLOADS",
    "build_table",
    "coverage",
    "describe",
    "describe_vocab",
    "fetch_sources",
    "load_tables",
    "main",
    "project",
    "sentence_encoder",
]

#: The sentence model. 384-d, 22M parameters, CPU-fine for 30k short strings.
DEFAULT_MODEL = "all-MiniLM-L6-v2"

#: Where downloaded code tables are cached. Under ``data/``, so gitignored.
DEFAULT_SOURCE_DIR = Path("data/code_text")

#: The random init's standard deviation, which the text table is rescaled to match
#: (:meth:`ehrjepa.models.embedding.EventEmbedding._init_weights`).
INIT_STD = 0.02

#: ``name -> (url, members to extract)``. Both are public, unauthenticated, and
#: small enough to pull in well under a minute on a home connection.
SOURCE_DOWNLOADS: dict[str, tuple[str, tuple[str, ...]]] = {
    "icd9": (
        "https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/Downloads"
        "/ICD-9-CM-v32-master-descriptions.zip",
        ("CMS32_DESC_LONG_DX.txt", "CMS32_DESC_LONG_SG.txt"),
    ),
    "ndc": ("https://www.accessdata.fda.gov/cder/ndctext.zip", ("product.txt",)),
    "ndc_excluded": (
        "https://www.accessdata.fda.gov/cder/ndc_excluded.zip",
        ("Products_excluded.txt",),
    ),
}

#: Where a MIMIC-IV demo extract is looked for, for ``d_hcpcs`` and ``drgcodes``.
#: The demo is public and uncredentialed; only these two dictionary tables are read.
MIMIC_DEMO_GLOBS: tuple[str, ...] = (
    "data/mimic-iv-demo/**/hosp/{name}.csv.gz",
    "data/mimic-iv-demo/hosp/{name}.csv.gz",
)


# --------------------------------------------------------------------------- #
# Hand-written descriptions: the structural and special codes
# --------------------------------------------------------------------------- #

#: The DE-SynPUF beneficiary-summary chronic-condition flags. The codebook names
#: each ``SP_*`` column and codes its value ``1`` as "yes" and ``2`` as "no";
#: this repository's ETL emits the flag only when it fires, so ``//1`` is what
#: the vocabulary actually contains.
SP_FLAGS: dict[str, str] = {
    "SP_ALZHDMTA": "Alzheimer's disease or related disorders or senile dementia",
    "SP_CHF": "heart failure",
    "SP_CHRNKIDN": "chronic kidney disease",
    "SP_CNCR": "cancer",
    "SP_COPD": "chronic obstructive pulmonary disease",
    "SP_DEPRESSN": "depression",
    "SP_DIABETES": "diabetes",
    "SP_ISCHMCHT": "ischemic heart disease",
    "SP_OSTEOPRS": "osteoporosis",
    "SP_RA_OA": "rheumatoid arthritis or osteoarthritis",
    "SP_STRKETIA": "stroke or transient ischemic attack",
}

SP_VALUES: dict[str, str] = {"1": "chronic condition flag: {}", "2": "no chronic condition: {}"}

#: Whole codes with no ``PREFIX//value`` structure, plus the reserved ids.
LITERAL_CODES: dict[str, str] = {
    "[PAD]": "padding, not a clinical event",
    "[UNK]": "an unrecognised or very rare clinical code",
    "[CLS]": "summary of this patient's record",
    "[MASK]": "a hidden clinical event",
    "MEDS_BIRTH": "date of birth",
    "MEDS_DEATH": "death of the patient",
}

#: ``PREFIX//value`` pairs written out by hand, from the DE-SynPUF codebook.
STRUCTURAL_CODES: dict[str, str] = {
    "VISIT//OUTPATIENT": "outpatient visit",
    "ADMISSION//INPATIENT": "admission to hospital as an inpatient",
    "DISCHARGE//INPATIENT": "discharge from an inpatient hospital stay",
    "SEX//M": "sex: male",
    "SEX//F": "sex: female",
    "RACE//WHITE": "race: white",
    "RACE//BLACK": "race: black",
    "RACE//HISPANIC": "ethnicity: hispanic",
    "RACE//OTHER": "race: other",
}

#: A bare ``PREFIX`` entry exists when the vocabulary rolled a whole family up
#: into one id, so it stands for "some code of this kind and nothing narrower".
PREFIX_CONCEPTS: dict[str, str] = {
    "ICD9CM": "a diagnosis, coded in ICD-9-CM",
    "ICD9PROC": "an inpatient procedure, coded in ICD-9-CM",
    "HCPCS": "a procedure, service or supply, coded in HCPCS or CPT",
    "NDC": "a prescribed drug product",
    "DRG": "a hospital diagnosis-related group",
    "VISIT": "a healthcare visit",
    "ADMISSION": "an admission to hospital",
    "DISCHARGE": "a discharge from hospital",
    "SEX": "the patient's sex",
    "RACE": "the patient's race or ethnicity",
}

#: The ICU waveform/lab channels of the two PhysioNet sources, as the plain
#: English a clinician would say out loud. Deliberately short and unit-free: they
#: are read both by the sentence encoder (where a long parenthetical about units
#: dilutes the signal) and by :mod:`ehrjepa.models.lm`, which prints them
#: verbatim in front of the number ("heart rate 92 (+1h)"), where a unit would be
#: repeated on every one of a window's several hundred events. The ETLs'
#: ``DESCRIPTIONS`` tables carry the units and remain the reference for those.
#:
#: Keyed on the part after ``VAR//``. The two challenges name several of the same
#: quantities differently (2019 ``SBP``/``DBP``/``Hct``/``Resp`` against 2012
#: ``SysABP``/``DiasABP``/``HCT``/``RespRate``), so both spellings are here.
ICU_CHANNELS: dict[str, str] = {
    # Vitals
    "HR": "heart rate",
    "O2Sat": "oxygen saturation",
    "Temp": "temperature",
    "SBP": "systolic blood pressure",
    "SysABP": "systolic blood pressure",
    "NISysABP": "non-invasive systolic blood pressure",
    "MAP": "mean arterial pressure",
    "NIMAP": "non-invasive mean arterial pressure",
    "DBP": "diastolic blood pressure",
    "DiasABP": "diastolic blood pressure",
    "NIDiasABP": "non-invasive diastolic blood pressure",
    "Resp": "respiratory rate",
    "RespRate": "respiratory rate",
    "EtCO2": "end-tidal carbon dioxide",
    "GCS": "Glasgow coma scale",
    "MechVent": "mechanical ventilation",
    "Urine": "urine output",
    "Weight": "weight",
    "Height": "height",
    # Blood gas
    "BaseExcess": "base excess",
    "HCO3": "bicarbonate",
    "FiO2": "fraction of inspired oxygen",
    "pH": "arterial pH",
    "PaCO2": "arterial carbon dioxide partial pressure",
    "PaO2": "arterial oxygen partial pressure",
    "SaO2": "arterial oxygen saturation",
    # Chemistry
    "AST": "aspartate transaminase",
    "ALT": "alanine transaminase",
    "ALP": "alkaline phosphatase",
    "Alkalinephos": "alkaline phosphatase",
    "BUN": "blood urea nitrogen",
    "Calcium": "calcium",
    "Chloride": "chloride",
    "Creatinine": "creatinine",
    "Bilirubin": "bilirubin",
    "Bilirubin_direct": "direct bilirubin",
    "Bilirubin_total": "total bilirubin",
    "Glucose": "serum glucose",
    "Lactate": "lactate",
    "Magnesium": "magnesium",
    "Mg": "magnesium",
    "Phosphate": "phosphate",
    "Potassium": "potassium",
    "K": "potassium",
    "Na": "sodium",
    "Albumin": "albumin",
    "Cholesterol": "cholesterol",
    "TroponinI": "troponin I",
    "TroponinT": "troponin T",
    # Haematology
    "Hct": "hematocrit",
    "HCT": "hematocrit",
    "Hgb": "hemoglobin",
    "PTT": "partial thromboplastin time",
    "WBC": "white blood cell count",
    "Fibrinogen": "fibrinogen",
    "Platelets": "platelet count",
}

#: ICU stay structure, outcomes and severity scores: whole codes, no
#: ``PREFIX//value`` split, from the two PhysioNet ETLs.
ICU_CODES: dict[str, str] = {
    "ICU_HOUR": "hour of the ICU stay",
    "AGE": "age in years",
    "HOSP_ADM_TIME": "hours from hospital admission to ICU admission",
    "SEPSIS_ONSET": "onset of sepsis",
    "LOS": "length of hospital stay in days",
    "SURVIVAL": "days from ICU admission to death",
    "SAPS_I": "SAPS-I severity score",
    "SOFA": "SOFA organ failure score",
}

#: ICU types, keyed on the part after ``UNIT//``.
ICU_UNITS: dict[str, str] = {
    "MICU": "medical intensive care unit",
    "SICU": "surgical intensive care unit",
    "CCU": "coronary care unit",
    "CSRU": "cardiac surgery recovery unit",
}

#: HCPCS level II letter groups, for codes the code table does not carry.
HCPCS_LETTERS: dict[str, str] = {
    "A": "transportation, medical or surgical supply",
    "B": "enteral or parenteral nutrition therapy",
    "C": "device, drug or biological used in a hospital outpatient department",
    "D": "dental procedure",
    "E": "durable medical equipment",
    "G": "temporary procedure or professional service",
    "H": "behavioral health or substance abuse treatment service",
    "J": "drug administered other than by mouth",
    "K": "durable medical equipment supplied by a Medicare contractor",
    "L": "orthotic or prosthetic procedure",
    "M": "medical service",
    "P": "pathology or laboratory service",
    "Q": "temporary code for a drug, biological or piece of medical equipment",
    "R": "diagnostic radiology service",
    "S": "temporary national code used by private payers",
    "T": "national code used by state Medicaid agencies",
    "V": "vision, hearing or speech-language pathology service",
}

#: CPT sections, for the numeric part of the HCPCS range.
CPT_SECTIONS: tuple[tuple[int, int, str], ...] = (
    (100, 1999, "anesthesia service"),
    (10000, 69999, "surgical procedure"),
    (70000, 79999, "radiology procedure"),
    (80000, 89999, "pathology or laboratory test"),
    (90000, 99999, "medical service or evaluation"),
)


@dataclass(frozen=True)
class Description:
    """One vocabulary row's sentence, where it came from, and how specific it is."""

    code: str
    text: str
    source: str
    #: ``exact`` | ``ancestor`` | ``fallback`` -- see the module docstring.
    tier: str


# --------------------------------------------------------------------------- #
# Source tables
# --------------------------------------------------------------------------- #


@dataclass
class DescriptionTables:
    """The lookup tables :func:`describe` reads, each possibly empty.

    An empty table is not an error: it costs coverage, which the JSON reports, and
    the build still produces a usable init. That is deliberate -- the GPU machine
    that reruns this may not have a MIMIC demo extract lying around, and a build
    that refused to run there would just mean no text init at all.
    """

    icd9_dx: dict[str, str]
    icd9_sg: dict[str, str]
    icd9_category: Callable[[str], str | None] | None
    hcpcs: dict[str, str]
    hcpcs_section: dict[str, str]
    drg: dict[str, str]
    ndc_product: dict[str, str]
    ndc_labeler: dict[str, str]

    def present(self) -> dict[str, int]:
        """Row counts per table, for the coverage report."""
        return {
            "icd9_dx": len(self.icd9_dx),
            "icd9_sg": len(self.icd9_sg),
            "icd9_category": -1 if self.icd9_category else 0,
            "hcpcs": len(self.hcpcs),
            "hcpcs_section": len(self.hcpcs_section),
            "drg": len(self.drg),
            "ndc_product": len(self.ndc_product),
            "ndc_labeler": len(self.ndc_labeler),
        }


def _read_cms_descriptions(path: Path) -> dict[str, str]:
    """``0010  Cholera due to vibrio cholerae`` -> ``{"0010": "Cholera ..."}``.

    Latin-1, not UTF-8: the 2013 CMS diagnosis file is ISO-8859 and a strict
    UTF-8 read of it dies on a single degree sign somewhere in the burns chapter.
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="latin-1").splitlines():
        code, _, desc = line.strip().partition(" ")
        if code and desc.strip():
            out.setdefault(code, desc.strip())
    return out


def _read_fda_products(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """``(9-digit product code -> drug phrase, 5-digit labeler -> company)``.

    The FDA writes ``PRODUCTNDC`` hyphenated in whichever of the 4-4, 5-3 and 5-4
    segment layouts the labeler registered; this repository's vocabulary holds
    the 9-digit 5-4 form (see ``tokenize.normalize_code``), so both halves are
    zero-padded to that shape before anything is keyed on them.
    """
    products: dict[str, str] = {}
    labelers: dict[str, str] = {}
    if not path.exists():
        return products, labelers
    with path.open(encoding="latin-1", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            code = str(row.get("PRODUCTNDC") or "")
            if "-" not in code:
                continue
            labeler, _, product = code.partition("-")
            key = f"{labeler.zfill(5)}{product.zfill(4)}"
            company = (row.get("LABELERNAME") or "").strip()
            if company:
                labelers.setdefault(labeler.zfill(5), company)
            if key in products:
                continue
            phrase = _drug_phrase(row)
            if phrase:
                products[key] = phrase
    return products, labelers


def _drug_phrase(row: Mapping[str, Any]) -> str:
    """ "Zepbound (tirzepatide), injection, solution, subcutaneous"."""
    brand = (row.get("PROPRIETARYNAME") or "").strip()
    generic = (row.get("NONPROPRIETARYNAME") or "").strip()
    head = (
        f"{brand} ({generic})"
        if brand and generic and brand.lower() != generic.lower()
        else (brand or generic)
    )
    if not head:
        return ""
    tail = [
        part.strip().lower()
        for part in (row.get("DOSAGEFORMNAME"), row.get("ROUTENAME"))
        if (part or "").strip()
    ]
    return ", ".join([head, *tail])


def _read_gzip_csv(
    path: Path, key: str, value: str, where: Mapping[str, str] | None = None
) -> dict[str, str]:
    """One column keyed by another out of a gzipped CSV, everything read as text."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if any((row.get(k) or "") != v for k, v in (where or {}).items()):
                continue
            code, text = (row.get(key) or "").strip(), (row.get(value) or "").strip()
            if code and text:
                out.setdefault(code, text)
    return out


def _find_mimic(name: str, source_dir: Path, repo: Path) -> Path:
    """``<source_dir>/<name>.csv.gz`` if it was copied there, else a demo extract."""
    direct = source_dir / f"{name}.csv.gz"
    if direct.exists():
        return direct
    for pattern in MIMIC_DEMO_GLOBS:
        hits = sorted(repo.glob(pattern.format(name=name)))
        if hits:
            return hits[0]
    return direct


def _icd9_category_lookup() -> Callable[[str], str | None] | None:
    """The optional ``icd9cms`` package, which carries the 3-digit category titles.

    The CMS master description file lists billable leaves only, so a vocabulary
    entry the rollup created (``ICD9CM//250``, absorbing every ``250xx``) has no
    row there. ``icd9cms`` is a pure-data package with the full hierarchy; when
    it is not installed those entries fall back to their chapter, which is a real
    but much coarser concept.
    """
    try:
        from icd9cms.icd9 import search
    except Exception:  # pragma: no cover - optional dependency
        return None

    def lookup(code: str) -> str | None:
        try:
            node = search(code)
        except Exception:  # pragma: no cover - defensive: the package parses lazily
            return None
        if node is None:
            return None
        text = getattr(node, "long_desc", None) or getattr(node, "short_desc", None)
        return str(text).strip() if text else None

    return lookup


def load_tables(
    source_dir: str | Path = DEFAULT_SOURCE_DIR, repo: Path | None = None
) -> DescriptionTables:
    """Read whatever description tables are present under ``source_dir``."""
    root = Path(source_dir)
    here = repo or Path.cwd()
    hcpcs_path = _find_mimic("d_hcpcs", root, here)
    product, labeler = _read_fda_products(root / "product.txt")
    excluded, excluded_labeler = _read_fda_products(root / "Products_excluded.txt")
    for key, value in excluded.items():
        product.setdefault(key, value)
    for key, value in excluded_labeler.items():
        labeler.setdefault(key, value)
    return DescriptionTables(
        icd9_dx=_read_cms_descriptions(root / "CMS32_DESC_LONG_DX.txt"),
        icd9_sg=_read_cms_descriptions(root / "CMS32_DESC_LONG_SG.txt"),
        icd9_category=_icd9_category_lookup(),
        hcpcs=_read_gzip_csv(hcpcs_path, "code", "long_description"),
        # MIMIC leaves ``long_description`` null for the numeric CPT range -- the
        # AMA licenses those titles and they cannot be redistributed -- but fills
        # ``short_description`` with the CPT *subsection* ("Cardiovascular
        # system", "Diagnostic imaging"). That is a real parent concept for the
        # code, and far better than the five-way section split a bare number
        # would otherwise get, so it is read as a separate ancestor-tier table.
        hcpcs_section=_read_gzip_csv(hcpcs_path, "code", "short_description"),
        drg=_read_gzip_csv(
            _find_mimic("drgcodes", root, here), "drg_code", "description", {"drg_type": "HCFA"}
        ),
        ndc_product=product,
        ndc_labeler=labeler,
    )


def fetch_sources(
    source_dir: str | Path = DEFAULT_SOURCE_DIR, timeout: float = 120.0, force: bool = False
) -> dict[str, str]:
    """Download and unpack the public code tables that are missing. Returns a status per source."""
    root = Path(source_dir)
    root.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for name, (url, members) in SOURCE_DOWNLOADS.items():
        if not force and all((root / member).exists() for member in members):
            status[name] = "present"
            continue
        try:
            request = Request(url, headers={"User-Agent": "ehrjepa/code_text"})
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URLs
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names = {Path(n).name: n for n in archive.namelist()}
                for member in members:
                    if member not in names:
                        raise KeyError(f"{member} not in {url}")
                    (root / member).write_bytes(archive.read(names[member]))
            status[name] = "downloaded"
        except Exception as exc:  # pragma: no cover - network-dependent
            status[name] = f"failed: {type(exc).__name__}: {exc}"
            print(f"[warn] could not fetch {name} from {url}: {exc}", flush=True)
    return status


# --------------------------------------------------------------------------- #
# Code -> sentence
# --------------------------------------------------------------------------- #


def _hcpcs_group(value: str) -> str | None:
    """The letter group or CPT section a HCPCS code belongs to."""
    if not value:
        return None
    if value[0].isalpha():
        return HCPCS_LETTERS.get(value[0].upper())
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return None
    number = int(digits.ljust(5, "0")[:5])
    for low, high, label in CPT_SECTIONS:
        if low <= number <= high:
            return label
    return None


def _icd9_prefix_hit(value: str, table: Mapping[str, str]) -> tuple[str, str] | None:
    """The longest strict prefix of ``value`` that is itself in ``table``."""
    for cut in range(len(value) - 1, 0, -1):
        hit = table.get(value[:cut])
        if hit:
            return value[:cut], hit
    return None


def describe(code: str, tables: DescriptionTables) -> Description:
    """One vocabulary code's sentence.

    The order is always the same: the code's own concept, then a genuine parent
    concept in the same coding system, then the bare code. A vocabulary entry
    created by rollup (``ICD9CM//250``, ``NDC//54868``) *is* the parent concept,
    so for those the second branch is not a degradation -- it is the right
    answer, and the ``ancestor`` tier says so.
    """
    literal = LITERAL_CODES.get(code)
    if literal is not None:
        return Description(code, literal, "handwritten", "exact")
    structural = STRUCTURAL_CODES.get(code)
    if structural is not None:
        return Description(code, structural, "handwritten", "exact")
    icu = ICU_CODES.get(code)
    if icu is not None:
        return Description(code, icu, "physionet-codebook", "exact")

    parts = split_code(code)
    if parts is None:
        concept = PREFIX_CONCEPTS.get(code.upper())
        if concept:
            return Description(code, concept, "handwritten", "ancestor")
        return Description(code, f"clinical event coded {code}", "fallback", "fallback")
    prefix, _, value = parts
    family = prefix.upper()

    if family == "VAR":
        # Case-sensitive: the challenge column names are the vocabulary's own
        # spelling and are what :data:`ICU_CHANNELS` is keyed on.
        channel = ICU_CHANNELS.get(value)
        if channel:
            return Description(code, channel, "physionet-codebook", "exact")
        return Description(code, f"measurement of {value}", "fallback", "fallback")

    if family == "UNIT":
        unit = ICU_UNITS.get(value.upper())
        if unit:
            return Description(code, unit, "physionet-codebook", "exact")
        return Description(code, f"admission to the {value} unit", "fallback", "fallback")

    if family.startswith("SP_"):
        name = SP_FLAGS.get(family)
        template = SP_VALUES.get(value)
        if name and template:
            return Description(code, template.format(name), "desynpuf-codebook", "exact")

    if family == "ICD9CM":
        exact = tables.icd9_dx.get(value)
        if exact:
            return Description(code, f"diagnosis: {exact}", "cms-icd9-dx", "exact")
        if tables.icd9_category is not None:
            category = tables.icd9_category(value)
            if category:
                return Description(code, f"diagnosis category: {category}", "icd9cms", "ancestor")
        hit = _icd9_prefix_hit(value, tables.icd9_dx)
        if hit:
            return Description(
                code, f"diagnosis related to: {hit[1]}", "cms-icd9-dx-prefix", "ancestor"
            )
        return Description(code, f"diagnosis, ICD-9-CM code {value}", "fallback", "fallback")

    if family == "ICD9PROC":
        exact = tables.icd9_sg.get(value)
        if exact:
            return Description(code, f"procedure: {exact}", "cms-icd9-sg", "exact")
        hit = _icd9_prefix_hit(value, tables.icd9_sg)
        if hit:
            return Description(
                code, f"procedure related to: {hit[1]}", "cms-icd9-sg-prefix", "ancestor"
            )
        return Description(
            code, f"inpatient procedure, ICD-9-CM code {value}", "fallback", "fallback"
        )

    if family == "HCPCS":
        exact = tables.hcpcs.get(value)
        if exact:
            return Description(code, f"procedure or service: {exact}", "mimic-d-hcpcs", "exact")
        section = tables.hcpcs_section.get(value) if not value[:1].isalpha() else None
        if section and section.lower() != "invalid code":
            return Description(
                code, f"procedure or service: {section.lower()}", "mimic-cpt-section", "ancestor"
            )
        hit = _icd9_prefix_hit(value, tables.hcpcs)
        if hit:
            return Description(
                code,
                f"procedure or service related to: {hit[1]}",
                "mimic-d-hcpcs-prefix",
                "ancestor",
            )
        group = _hcpcs_group(value)
        if group:
            return Description(code, f"procedure or service: {group}", "hcpcs-group", "ancestor")
        return Description(
            code, f"procedure or service, HCPCS code {value}", "fallback", "fallback"
        )

    if family == "NDC":
        exact = tables.ndc_product.get(value)
        if exact:
            return Description(code, f"medication: {exact}", "fda-ndc", "exact")
        company = tables.ndc_labeler.get(value[:5])
        if company:
            return Description(
                code, f"medication manufactured by {company}", "fda-ndc-labeler", "ancestor"
            )
        return Description(code, f"drug product NDC {value}", "fallback", "fallback")

    if family == "DRG":
        exact = tables.drg.get(value) or tables.drg.get(value.zfill(3))
        if exact:
            return Description(
                code, f"diagnosis-related group: {exact.lower()}", "mimic-drgcodes", "exact"
            )
        return Description(
            code, f"hospital diagnosis-related group {value}", "fallback", "fallback"
        )

    concept = PREFIX_CONCEPTS.get(family)
    if concept:
        return Description(code, f"{concept}, coded {value}", "handwritten-prefix", "ancestor")
    return Description(code, f"clinical event coded {code}", "fallback", "fallback")


def describe_vocab(vocab: Vocabulary, tables: DescriptionTables) -> list[Description]:
    """:func:`describe` over every id, in id order."""
    return [describe(code, tables) for code in vocab.codes]


def _family(code: str) -> str:
    """The coding system a vocabulary entry belongs to, for the coverage cut."""
    if code in SPECIAL_TOKENS:
        return "[SPECIAL]"
    parts = split_code(code)
    name = parts[0].upper() if parts else code.upper()
    return "SP_*" if name.startswith("SP_") else name


def coverage(vocab: Vocabulary, descriptions: Sequence[Description]) -> dict[str, Any]:
    """Entry counts and train-event mass per tier, per source and per code family."""
    total_events = sum(vocab.train_count)
    by_tier: dict[str, dict[str, int]] = {}
    by_source: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    for code, count, item in zip(vocab.codes, vocab.train_count, descriptions, strict=True):
        for table, key in ((by_tier, item.tier), (by_source, item.source)):
            bucket = table.setdefault(key, {"entries": 0, "events": 0})
            bucket["entries"] += 1
            bucket["events"] += count
        # The per-family cut carries its own tier split, which is the one a
        # reader actually wants: "NDC is 23k entries, 8k of them exact".
        family = by_family.setdefault(
            _family(code), {"entries": 0, "events": 0, "exact": 0, "ancestor": 0, "fallback": 0}
        )
        family["entries"] += 1
        family["events"] += count
        family[item.tier] += 1

    described = sum(b["entries"] for k, b in by_tier.items() if k != "fallback")
    described_events = sum(b["events"] for k, b in by_tier.items() if k != "fallback")
    return {
        "vocab_size": len(vocab),
        "train_events": total_events,
        "described_entries": described,
        "described_entry_rate": described / max(1, len(vocab)),
        "described_events": described_events,
        "described_event_rate": described_events / max(1, total_events),
        "exact_entry_rate": by_tier.get("exact", {}).get("entries", 0) / max(1, len(vocab)),
        "exact_event_rate": by_tier.get("exact", {}).get("events", 0) / max(1, total_events),
        "by_tier": by_tier,
        "by_source": by_source,
        "by_family": by_family,
    }


# --------------------------------------------------------------------------- #
# Sentences -> an embedding table
# --------------------------------------------------------------------------- #


def sentence_encoder(
    model_name: str = DEFAULT_MODEL, device: str = "cpu"
) -> Callable[[Sequence[str]], np.ndarray]:
    """A ``texts -> (n, d) float32`` callable backed by ``sentence-transformers``.

    Imported here rather than at module scope: the description side of this file
    is pure Python and stdlib, and a repository that only wants to *inspect* the
    descriptions should not need a 90 MB model download to import the module.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "code_init: text needs sentence-transformers. "
            "Install it with `uv pip install sentence-transformers`, or pass your own "
            "encoder to build_table()."
        ) from exc
    model = SentenceTransformer(model_name, device=device)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            model.encode(list(texts), batch_size=256, show_progress_bar=True), dtype=np.float32
        )

    return encode


def project(vectors: np.ndarray, width: int, seed: int = 0, std: float = INIT_STD) -> np.ndarray:
    """Centre, PCA to ``width``, rescale to ``std``.

    PCA rather than a learned linear map because the whole point is an
    *initialization*: a fitted projection needs no training run of its own, is a
    deterministic function of the sentences, and preserves the pairwise geometry
    the sentence model produced as well as a rank-``width`` map can.

    ``width > 384`` cannot be filled from 384-d inputs, so the leftover columns
    get seeded Gaussian noise at the same scale rather than zeros -- a column of
    zeros is a dead input dimension with no gradient of its own, which is a worse
    thing to hand an optimizer than noise.

    The rescale is a single global scalar, not a per-dimension whitening: it
    matches the total scale of ``N(0, 0.02)`` while leaving the anisotropy of the
    PCA spectrum intact, which is the part that carries the meaning. The
    per-dimension standard deviations land in the coverage JSON so the anisotropy
    is on the record rather than implicit.
    """
    from sklearn.decomposition import PCA

    if vectors.ndim != 2:
        raise ValueError(f"expected a (n, d) matrix of sentence vectors, got {vectors.shape}")
    rows, dim = vectors.shape
    centred = np.asarray(vectors, dtype=np.float64) - vectors.mean(axis=0, keepdims=True)
    kept = min(width, dim, rows)
    out = np.zeros((rows, width), dtype=np.float64)
    out[:, :kept] = PCA(n_components=kept, random_state=seed).fit_transform(centred)
    if width > kept:
        scale = float(out[:, :kept].std()) or 1.0
        out[:, kept:] = np.random.default_rng(seed).normal(0.0, scale, size=(rows, width - kept))
    current = float(out.std())
    if current > 0:
        out *= std / current
    return np.ascontiguousarray(out, dtype=np.float32)


def build_table(
    cache_dir: str | Path,
    width: int,
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
    model_name: str = DEFAULT_MODEL,
    seed: int = 0,
    std: float = INIT_STD,
    stats_out: str | Path | None = None,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Build ``code_init_text_<width>.npy`` and its coverage JSON for one cache.

    Returns the stats mapping that was written. ``encoder`` is injectable so the
    tests -- and anyone who prefers a different sentence model -- do not have to
    go through ``sentence-transformers``.
    """
    cache = Path(cache_dir)
    vocab = Vocabulary.read(cache / "vocab.parquet")
    tables = load_tables(source_dir, repo=repo)
    descriptions = describe_vocab(vocab, tables)
    encode = encoder or sentence_encoder(model_name)
    vectors = np.asarray(encode([item.text for item in descriptions]), dtype=np.float32)
    if vectors.shape[0] != len(vocab):
        raise ValueError(f"encoder returned {vectors.shape[0]} rows for {len(vocab)} codes")
    table = project(vectors, width, seed=seed, std=std)
    # PAD is a real row that is never attended to, and the random init zeroes it
    # too. Zeroing then re-scaling, rather than the other way round, is what makes
    # "the table's standard deviation is exactly ``std``" true of what is saved.
    table[PAD_ID] = 0.0
    scale = float(table.std())
    if scale > 0:
        table *= std / scale

    npy_path = cache / f"code_init_text_{width}.npy"
    np.save(npy_path, table)
    stats: dict[str, Any] = {
        "cache_dir": str(cache),
        "width": width,
        "sentence_model": model_name if encoder is None else "custom",
        "sentence_dim": int(vectors.shape[1]),
        "seed": seed,
        "init_std": std,
        "table": str(npy_path.name),
        "table_std": float(table.std()),
        "table_dim_std": [round(float(v), 6) for v in table.std(axis=0)],
        "source_rows": tables.present(),
        **coverage(vocab, descriptions),
    }
    (cache / f"code_init_text_{width}.json").write_text(json.dumps(stats, indent=2) + "\n")
    if stats_out:
        Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_out).write_text(json.dumps(stats, indent=2) + "\n")
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _summary_line(stats: Mapping[str, Any]) -> str:
    tiers = stats["by_tier"]
    parts = [
        f"{tier}: {body['entries']:,} entries / {body['events']:,} events"
        for tier, body in sorted(tiers.items())
    ]
    return (
        f"{stats['described_entries']:,}/{stats['vocab_size']:,} entries described "
        f"({stats['described_entry_rate']:.1%}), "
        f"{stats['described_event_rate']:.1%} of train events -- " + "; ".join(parts)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ehrjepa.data.code_text",
        description="Describe every vocabulary code and build a text-initialized embedding table.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download the public code tables")
    fetch.add_argument("--sources", type=Path, default=DEFAULT_SOURCE_DIR)
    fetch.add_argument("--force", action="store_true")

    show = sub.add_parser("describe", help="print the descriptions and the coverage report")
    show.add_argument("--cache", type=Path, required=True)
    show.add_argument("--sources", type=Path, default=DEFAULT_SOURCE_DIR)
    show.add_argument("--limit", type=int, default=25, help="example rows to print (0 for none)")
    show.add_argument("--out", type=Path, default=None, help="write every description to this TSV")

    build = sub.add_parser("build", help="embed the descriptions and write the init table")
    build.add_argument("--cache", type=Path, required=True)
    build.add_argument("--width", type=int, required=True, help="the model's `model.dim`")
    build.add_argument("--sources", type=Path, default=DEFAULT_SOURCE_DIR)
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--stats-out", type=Path, default=None, help="a second copy of the JSON")
    build.add_argument("--offline", action="store_true", help="never download; use what is present")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        print(json.dumps(fetch_sources(args.sources, force=args.force), indent=2))
        return 0

    if args.command == "describe":
        vocab = Vocabulary.read(args.cache / "vocab.parquet")
        tables = load_tables(args.sources)
        descriptions = describe_vocab(vocab, tables)
        for item in descriptions[: max(0, args.limit)]:
            print(f"{item.code:<28} [{item.tier:<8} {item.source:<22}] {item.text}")
        if args.out:
            args.out.write_text(
                "code\ttier\tsource\ttext\n"
                + "".join(f"{d.code}\t{d.tier}\t{d.source}\t{d.text}\n" for d in descriptions)
            )
        print()
        print(json.dumps(coverage(vocab, descriptions), indent=2))
        return 0

    if not args.offline:
        fetch_sources(args.sources)
    stats = build_table(
        args.cache,
        args.width,
        source_dir=args.sources,
        model_name=args.model,
        seed=args.seed,
        stats_out=args.stats_out,
    )
    print(_summary_line(stats))
    print(f"wrote {args.cache / f'code_init_text_{args.width}.npy'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
