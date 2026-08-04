from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from ..cli import _train
from .ablation import implementation_hash
from .walk_forward import TimestampSplit


def main(argv=None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise ValueError("ablation worker 需要一个 job JSON")
    job = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    if job.get("implementation_sha256") != implementation_hash():
        raise RuntimeError("worker 训练实现 hash 与冻结合同不一致")
    with np.load(job["dataset"], allow_pickle=False) as data:
        timestamps = data["timestamps"]
    boundary = job["split"]

    def select(first_name: str, last_name: str):
        first, last = boundary[first_name], boundary[last_name]
        selected = np.flatnonzero((timestamps >= first) & (timestamps <= last))
        times = np.unique(timestamps[selected])
        if times.size == 0 or int(times[0]) != first or int(times[-1]) != last:
            raise ValueError("worker fold 边界无法在数据集中精确重建")
        return selected, times

    train, train_times = select("train_first", "train_last")
    validation, validation_times = select("validation_first", "validation_last")
    test, test_times = select("test_first", "test_last")
    split = TimestampSplit(
        train=train, validation=validation, test=test,
        train_timestamps=train_times,
        validation_timestamps=validation_times,
        test_timestamps=test_times,
    )
    _train(job["config"], job["dataset"], job["output"], _split_override=split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
