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

To combine the visual quality of 2D generation with the spatial consistency of 3D scene modeling, PanoWorld uses a floorplan-derived 3D shell as global geometric guidance and a dynamic 3D Gaussian Splatting cache as renderable spatial memory. The framework further introduces a feed-forward panoramic LRM for metric-scale multi-room 360-degree inputs, Room-aware Group Attention to suppress cross-room interference, and a topology-aware progressive caching strategy that avoids repeatedly reconstructing the full scene history.

This repository is the official project for the paper. Training and inference code, together with evaluation data, will be released soon.

<p align="center">
  <img src="assets/img1_good2.png" alt="PanoWorld main figure" width="95%">
</p>

## Overview

- Whole-house synthesis is formulated as autoregressive generation over discrete panorama viewpoints, matching real VR-tour navigation.
- A floorplan-derived 3D shell provides global structural guidance for multi-room layout consistency.
- A dynamic 3DGS cache serves as renderable spatial memory, preserving cross-node geometry and material identity.
- Room-aware panoramic LRM and topology-aware progressive caching improve scalability for metric-scale, multi-room synthesis.

## News

- `2026-05-19`: Paper released and project page launched. Training/inference code and evaluation data will be open-sourced soon.

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
