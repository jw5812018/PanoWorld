# PanoWorld-LRM Inference

This package contains the inference-only release of the PanoWorld-LRM component used for whole-house reconstruction from panorama inputs.

## What is included

- Inference entrypoint: `inference.py`
- Model definition and runtime dependencies required by inference
- Two example configs:
  - `configs/inference_1024_512.yaml`
  - `configs/inference_2048_1024.yaml`
- Evaluation file lists under `data_realsee3D/`

Training scripts, WandB setup, API keys, and other non-inference artifacts have been removed.

## Quick start

1. Install the dependencies listed in `requirements.txt`.
2. Update `ckpt_path`, `data.data_path`, and `inference.out_dir` in the chosen config if needed.
3. Run inference:

```bash
bash infer_512.sh
```

or

```bash
bash infer_1024.sh
```

You can also launch directly:

```bash
python inference.py --config configs/inference_1024_512.yaml
```

## Notes

- `training.train_stage` is still kept in the config because the checkpoint layout depends on it.
- The code path is intentionally kept numerically aligned with the original inference implementation; the cleanup focuses on packaging, unused code removal, and inference-side runtime simplification.
