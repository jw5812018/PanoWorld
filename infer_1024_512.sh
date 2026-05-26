#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1 \
python inference.py --config configs/inference_1024_512.yaml "$@"
