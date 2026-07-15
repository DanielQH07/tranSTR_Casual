import shutil
from pathlib import Path

import kagglehub


DATASET_HANDLE = "danielq07/gdinofrcnn-ncod-hum-model"
CHECKPOINT_FILENAME = (
    "best_model_gdinofrcnn_ncod_hum_"
    "run1_generic_safe_lora_hn_ema_cos.ckpt"
)
DESTINATION = Path(__file__).resolve().parent / "weights" / CHECKPOINT_FILENAME


def main():
    try:
        dataset_root = Path(kagglehub.dataset_download(DATASET_HANDLE))
    except Exception as exc:
        raise RuntimeError(
            'Cannot download the Kaggle model dataset. Ensure your account has access '
            'and configure KAGGLE_API_TOKEN before running this script.'
        ) from exc
    candidates = list(dataset_root.rglob(CHECKPOINT_FILENAME))
    if not candidates:
        available = sorted(path.name for path in dataset_root.rglob("*.ckpt"))
        raise FileNotFoundError(
            f"Checkpoint {CHECKPOINT_FILENAME!r} was not found under {dataset_root}. "
            f"Available .ckpt files: {available}"
        )
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple matching checkpoints found: {candidates}")

    source = candidates[0]
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if not DESTINATION.exists() or DESTINATION.stat().st_size != source.stat().st_size:
        shutil.copy2(source, DESTINATION)

    if not DESTINATION.is_file() or DESTINATION.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint copy failed: {DESTINATION}")
    print(f"Checkpoint ready: {DESTINATION}")
    print(f"Size: {DESTINATION.stat().st_size / (1024 ** 2):.2f} MiB")


if __name__ == "__main__":
    main()
