"""Reading a subject's pre-anchor history out of the tensor cache.

Everything downstream of a label -- count features, encoder embeddings -- goes
through :class:`HistoryReader`, so there is exactly one place where the
"strictly before the anchor" rule is implemented and exactly one place a test
has to attack. The cut itself is :meth:`~ehrjepa.data.dataset.EventSequenceDataset.windows_at`,
which binary-searches ``time_min`` with ``side="left"``: an event whose
timestamp *equals* the anchor is excluded along with everything after it.

``max_len=None`` means unbounded history (what the count baselines want); an
integer keeps the most recent ``max_len`` events (what the encoder wants,
because that is the window it was pretrained on).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from ehrjepa.data.dataset import EventSequenceDataset

__all__ = ["HistoryReader", "anchor_minutes"]

#: Stand-in for "no window limit" -- larger than any subject's event count.
_UNBOUNDED = 1 << 40


def anchor_minutes(times: pl.Series) -> np.ndarray:
    """Anchor timestamps as int64 minutes since epoch, the cache's ``time_min`` unit."""
    return (times.dt.epoch(time_unit="us") // 60_000_000).to_numpy().astype(np.int64)


class HistoryReader:
    """Per-split views on one tensor cache, keyed by ``(subject_id, split)``.

    Parameters
    ----------
    cache_dir:
        A cache directory written by ``python -m ehrjepa.data.tokenize build``.
    max_len:
        Events kept per window, most recent first-truncated; ``None`` for all.
    splits:
        Which splits to open. Defaults to the three standard ones.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_len: int | None = None,
        splits: Iterable[str] = ("train", "tuning", "held_out"),
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_len = max_len
        self._datasets: dict[str, EventSequenceDataset] = {}
        for split in splits:
            if (self.cache_dir / split).is_dir():
                self._datasets[split] = EventSequenceDataset(
                    self.cache_dir,
                    split,
                    max_len=_UNBOUNDED if max_len is None else max_len,
                    min_len=1,
                    sampling="full",
                )

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._datasets)

    @property
    def vocab_size(self) -> int:
        return len(next(iter(self._datasets.values())).vocab.codes)

    def dataset(self, split: str) -> EventSequenceDataset:
        if split not in self._datasets:
            raise KeyError(f"split {split!r} is not in cache {self.cache_dir}")
        return self._datasets[split]

    def history(self, subject_id: int, split: str, anchor_min: int) -> dict[str, np.ndarray]:
        """Numpy arrays for the subject's events strictly before ``anchor_min``."""
        window = self.dataset(split).windows_at(int(subject_id), int(anchor_min))
        return {
            name: value.numpy()
            for name, value in window.items()
            if name not in ("length", "subject_id", "start")
        }

    def has_subject(self, subject_id: int, split: str) -> bool:
        try:
            self.dataset(split)
        except KeyError:
            return False
        return int(subject_id) in self.dataset(split)._row_of_subject

    def filter_present(self, anchors: pl.DataFrame) -> pl.DataFrame:
        """Drop anchor rows whose subject is not in the cache split.

        The cache is built from the same MEDS extract, so this only ever removes
        subjects with zero events; it is here so a mismatch is a dropped row with
        a count rather than a ``KeyError`` halfway through a run.
        """
        keep = [
            self.has_subject(row["subject_id"], row["split"])
            for row in anchors.select("subject_id", "split").iter_rows(named=True)
        ]
        return anchors.filter(pl.Series(keep))
