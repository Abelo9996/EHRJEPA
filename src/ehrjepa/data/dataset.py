"""Torch dataset and collate function over the flat tensor cache.

:class:`EventSequenceDataset` is one item per *subject*, not per window: each
``__getitem__`` draws a fresh contiguous window from that subject's slice of the
memmapped arrays, so an epoch sees a different crop of every patient. Nothing
here touches a device -- tensors come out on CPU and the training loop moves
them -- which is what keeps the same code running on MPS and CUDA.

The arrays are opened with ``mmap_mode="r"``, so the 14M-event DE-SynPUF cache
costs a few megabytes of RSS regardless of how much of it a run actually reads,
and worker processes share the page cache instead of each holding a copy.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ehrjepa.data.cache import FEATURES
from ehrjepa.data.tokenize import PAD_ID, Vocabulary

__all__ = ["EventSequenceDataset", "collate_events"]

#: Features that are integer-valued and become ``torch.long`` (embedding inputs).
_LONG_FEATURES = ("code_id", "value_bin", "time_min")

_EPOCH = dt.datetime(1970, 1, 1)


class EventSequenceDataset(torch.utils.data.Dataset):
    """Windows over one split of a tensor cache built by :mod:`ehrjepa.data.cache`.

    Parameters
    ----------
    cache_dir:
        Directory written by ``python -m ehrjepa.data.tokenize build``.
    split:
        ``train``, ``tuning`` or ``held_out``.
    max_len:
        Maximum events per window.
    min_len:
        Subjects with fewer events than this are dropped; the count is available
        as :attr:`n_dropped`.
    sampling:
        ``random_window`` draws a fresh uniformly-random contiguous window each
        time an item is requested. ``full`` is deterministic and returns the
        subject's most recent ``max_len`` events, which is what evaluation and
        the ``inspect`` CLI want.
    seed:
        Base seed for window sampling. The per-worker torch seed is mixed in, so
        DataLoader workers do not draw identical windows.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        max_len: int = 512,
        min_len: int = 16,
        sampling: str = "random_window",
        seed: int = 0,
    ) -> None:
        if sampling not in ("random_window", "full"):
            raise ValueError(f"sampling must be 'random_window' or 'full', got {sampling!r}")
        if min_len < 1 or max_len < 1:
            raise ValueError("max_len and min_len must be positive")
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.max_len = max_len
        self.min_len = min_len
        self.sampling = sampling
        self.seed = seed

        split_dir = self.cache_dir / split
        index = pl.read_parquet(split_dir / "subjects.parquet")
        self.n_subjects_total = index.height
        self.index = index.filter(pl.col("length") >= min_len)
        self.n_dropped = self.n_subjects_total - self.index.height

        self._offset = self.index["offset"].to_numpy()
        self._length = self.index["length"].to_numpy()
        self._subject_id = self.index["subject_id"].to_numpy()
        self._row_of_subject = {int(s): i for i, s in enumerate(self._subject_id)}
        self._arrays = {
            name: np.load(split_dir / f"{name}.npy", mmap_mode="r") for name in FEATURES
        }
        self._rng: np.random.Generator | None = None
        self._vocab: Vocabulary | None = None

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.index.height

    @property
    def n_events(self) -> int:
        """Total events in the split, including those of dropped subjects."""
        return int(self._arrays["code_id"].shape[0])

    @property
    def vocab(self) -> Vocabulary:
        """The cache's vocabulary, loaded on first use."""
        if self._vocab is None:
            self._vocab = Vocabulary.read(self.cache_dir / "vocab.parquet")
        return self._vocab

    def decode(self, code_ids: Sequence[int] | torch.Tensor) -> list[str]:
        """Code strings for a tensor of ids, for inspection and error messages."""
        ids = code_ids.tolist() if isinstance(code_ids, torch.Tensor) else list(code_ids)
        return [self.vocab.codes[int(i)] for i in ids]

    # ------------------------------------------------------------------ #

    def _generator(self) -> np.random.Generator:
        if self._rng is None:
            # torch.initial_seed() differs per DataLoader worker and per epoch.
            self._rng = np.random.default_rng((self.seed, torch.initial_seed()))
        return self._rng

    def rng_state(self) -> dict[str, object]:
        """The window-sampler RNG state, so a run can be checkpointed and resumed.

        Only meaningful with ``num_workers=0``; worker processes have their own
        generators, seeded per worker and per epoch, which the parent cannot see.
        """
        return dict(self._generator().bit_generator.state)

    def set_rng_state(self, state: Mapping[str, object]) -> None:
        """Restore a state returned by :meth:`rng_state`."""
        self._generator().bit_generator.state = dict(state)

    def _slice(self, start: int, stop: int, subject_id: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {}
        for name, array in self._arrays.items():
            values = np.array(array[start:stop])
            tensor = torch.from_numpy(values)
            item[name] = tensor.long() if name in _LONG_FEATURES else tensor.float()
        item["length"] = torch.tensor(stop - start, dtype=torch.long)
        item["subject_id"] = torch.tensor(subject_id, dtype=torch.long)
        item["start"] = torch.tensor(start, dtype=torch.long)
        return item

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        offset = int(self._offset[i])
        length = int(self._length[i])
        window = min(length, self.max_len)
        if self.sampling == "random_window" and length > window:
            start = offset + int(self._generator().integers(0, length - window + 1))
        else:
            # "full" keeps the most recent events, which is the causal context.
            start = offset + length - window
        return self._slice(start, start + window, int(self._subject_id[i]))

    # ------------------------------------------------------------------ #

    def windows_at(
        self, subject_id: int, end_time: dt.datetime | np.datetime64 | int
    ) -> dict[str, torch.Tensor]:
        """The subject's last ``max_len`` events **strictly before** ``end_time``.

        ``end_time`` may be a ``datetime``, a ``numpy.datetime64`` or an int in
        cache units (minutes since epoch). An event whose timestamp equals
        ``end_time`` is excluded, so label windows never see the labelled event.
        Deterministic, and independent of ``sampling``.
        """
        row = self._row_of_subject.get(int(subject_id))
        if row is None:
            raise KeyError(f"subject {subject_id} is not in split {self.split!r}")
        cutoff = _to_minutes(end_time)
        offset = int(self._offset[row])
        length = int(self._length[row])
        times = self._arrays["time_min"][offset : offset + length]
        stop = offset + int(np.searchsorted(times, cutoff, side="left"))
        start = max(offset, stop - self.max_len)
        return self._slice(start, stop, int(subject_id))


def _to_minutes(end_time: dt.datetime | np.datetime64 | int) -> int:
    """Coerce a cutoff to int64 minutes since epoch, matching ``time_min``."""
    if isinstance(end_time, (int, np.integer)):
        return int(end_time)
    if isinstance(end_time, np.datetime64):
        return int(end_time.astype("datetime64[m]").astype("int64"))
    if isinstance(end_time, dt.datetime):
        delta = end_time.replace(tzinfo=None) - _EPOCH
        return int(delta.total_seconds() // 60)
    raise TypeError(f"unsupported end_time type: {type(end_time)!r}")


def collate_events(batch: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a batch of windows to the batch maximum and build the attention mask.

    Sequence features are right-padded (``code_id`` with :data:`~.tokenize.PAD_ID`,
    everything else with zero) and ``attention_mask`` is 1 on real events.
    """
    if not batch:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    width = int(lengths.max())
    out: dict[str, torch.Tensor] = {}
    for name in FEATURES:
        pad = PAD_ID if name == "code_id" else 0
        rows = [
            torch.nn.functional.pad(item[name], (0, width - int(item["length"])), value=pad)
            for item in batch
        ]
        out[name] = torch.stack(rows)
    out["attention_mask"] = (torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)).to(torch.long)
    out["length"] = lengths
    out["subject_id"] = torch.stack([item["subject_id"] for item in batch])
    out["start"] = torch.stack([item["start"] for item in batch])
    return out
