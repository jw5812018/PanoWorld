# PanoWorld: A Generative Spatial World Model for Consistent Whole-House Panorama Synthesis

<p align="center">
  <strong>Jinrang Jia, Zhenjia Li, Yijiang Hu, Yifeng Shi</strong>
</p>

<p align="center">
  <strong>Ke Holdings Inc.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2605.17916"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.17916-b31b1b.svg"></a>
  <a href="https://jjrcn.github.io/PanoWorld-project-home/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2f80ed.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

PanoWorld is a generative spatial world model for consistent whole-house panorama synthesis. Given a floorplan and a style reference, it autoregressively generates node-based 360-degree panoramas that align with practical VR-tour navigation while preserving cross-view geometry and material consistency across an entire house.

This repository currently releases the **PanoWorld-LRM inference code**, together with model checkpoints and evaluation data links. More components of the full PanoWorld pipeline will be released progressively.

<p align="center">
  <img src="assets/img1_good2.png" alt="PanoWorld main figure" width="95%">
</p>

## Overview

- Whole-house synthesis is formulated as autoregressive generation over discrete panorama viewpoints, matching real VR-tour navigation.
- A floorplan-derived 3D shell provides global structural guidance for multi-room layout consistency.
- A dynamic 3DGS cache serves as renderable spatial memory, preserving cross-node geometry and material identity.
- PanoWorld-LRM reconstructs metric-scale multi-room geometry from panoramic observations for high-quality whole-house rendering and evaluation.

## News

- `2026-05-19`: Paper released and project page launched.
- `2026-05-25`: Open-sourced the PanoWorld-LRM inference code, checkpoints (including `1024x512` and `2048x1024` model weights), and evaluation data (`50` RealSee3D scenes).
- `Coming Soon`: PanoWorld 2D generator inference code and checkpoints.
- `Coming Soon`: Private scene data for evaluating PanoWorld panorama synthesis.
- `Coming Soon`: PanoWorld-LRM training code.
- `Coming Soon`: PanoWorld 2D generator training code.
- `Coming Soon`: Full PanoWorld pipeline, visualization, and evaluation code.

## Inference

### Quick Start

#### PanoWorld-LRM

1. Install dependencies:

```bash
pip install -r requirements.txt
```

The released inference package is tested with
`Python 3.10.18`, `PyTorch 2.3.1`, `TorchVision 0.18.1`, and `CUDA 12.1`.

2. Download the prepared RealSee3D inference and evaluation data ([Download](https://huggingface.co/datasets/JiaJinrang/PanoWorld/tree/main)):

3. Check the selected config and update `data.root_data_dir`, `data.data_path`, `inference.ckpt_path`, and `inference.out_dir` if needed.

4. Launch inference with one of the provided scripts:

```bash
bash infer_512.sh
```

or

```bash
bash infer_1024.sh
```

You can also run inference directly with:

```bash
python inference.py --config configs/inference_1024_512.yaml
```

5. If you would like to run inference on your own data, please refer to the dataset format description ([Here](https://huggingface.co/datasets/JiaJinrang/PanoWorld)):

You may reorganize your own data into the same format or modify `dataset.py` and other related files to adapt to your data.

<p align="center"><strong>Inference GPU Memory Usage</strong></p>

<table align="center">
  <thead>
    <tr>
      <th align="center"></th>
      <th align="center">1024x512</th>
      <th align="center">2048x1024</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">8-views</td>
      <td align="center">28061MiB</td>
      <td align="center">117143MiB</td>
    </tr>
    <tr>
      <td align="center">12-views</td>
      <td align="center">45449MiB</td>
      <td align="center">OOM</td>
    </tr>
  </tbody>
</table>

<p align="center"><sub><em>Tested on NVIDIA H200. The paper uses <code>1024x512</code> for experiments and metric computation.</em></sub></p>

#### PanoWorld 2D Generator

Coming Soon

#### PanoWorld

Coming Soon

### Released Files

- `inference.py`: main inference entrypoint
- `model.py`, `transformer.py`, `dpt_head.py`, `prope_custom.py`: model definition
- `dataset.py`, `utils.py`, `metric_utils.py`: dataset loading and evaluation helpers
- `configs/`: released inference configs for `1024x512` and `2048x1024`
- `data_realsee3D/`: released RealSee3D evaluation file lists

## Model Checkpoints

| Component | Resolution | Link | Notes |
| --- | --- | --- | --- |
| PanoWorld-LRM | `1024x512` | [Checkpoint](https://huggingface.co/JiaJinrang/PanoWorld/blob/main/model_ckpt/ckpt_panoworld_lrm_1024_512.pt) | Released |
| PanoWorld-LRM | `2048x1024` | [Checkpoint](https://huggingface.co/JiaJinrang/PanoWorld/blob/main/model_ckpt/ckpt_panoworld_lrm_2048_1024.ckpt) | Released |
| PanoWorld 2D Generator | Coming Soon | Coming Soon | Coming Soon |

## Data

| Split | Dataset | Usage | Link | Notes |
| --- | --- | --- | --- | --- |
| Training | 3D Front | Train LRM and 2D generator | [Download](https://tianchi.aliyun.com/dataset/65347) | Data processing scripts: Coming Soon |
| Training | RealSee3D | Train LRM and 2D generator | [Download](https://github.com/realsee-developer/RealSee3D) | Data processing scripts: Coming Soon |
| Training | Private 2D panoramas | 2D generator only | - | Private |
| Evaluation | RealSee3D | Evaluate LRM | [Download](https://huggingface.co/datasets/JiaJinrang/PanoWorld/tree/main) | Released, including `50` RealSee3D scenes |
| Evaluation | Private scene data | Evaluate PanoWorld panorama synthesis | Coming Soon | Coming Soon |

## Citation

If you find this project useful, please cite:

```bibtex
@misc{jia2026panoworldgenerativespatialworld,
      title={PanoWorld: A Generative Spatial World Model for Consistent Whole-House Panorama Synthesis},
      author={Jinrang Jia and Zhenjia Li and Yijiang Hu and Yifeng Shi},
      year={2026},
      eprint={2605.17916},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.17916},
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

We would like to thank [Gynjn/MVP](https://github.com/Gynjn/MVP), [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image), [realsee-developer/RealSee3D](https://github.com/realsee-developer/RealSee3D), and [3D Front](https://tianchi.aliyun.com/dataset/65347) for their inspiring open-source contributions.
