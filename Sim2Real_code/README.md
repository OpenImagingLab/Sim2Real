# Sim2Real: Per-Scene Optimization for Event-based Low-Light Frame Interpolation

This repository contains the official implementation of the paper **"From Sim to Real: Toward General Event-based Low-Light Frame Interpolation with Per-scene Optimization"**, presented at **SIGGRAPH Asia 2024**.

Our method introduces a **per-scene optimization strategy** tailored for **Video Frame Interpolation (VFI)** under low-light conditions, leveraging event-based cameras. This approach adapts a pre-trained model to scene-specific lighting and camera settings, mitigating artifacts and improving interpolation quality.

![Visual Comparison](Sim2Real_code/image.png)

## Key Features

- **Per-Scene Optimization**: Fine-tunes a pre-trained model for each scene, significantly enhancing interpolation quality.
- **Low-light Event Correction**: Resolves trailing artifacts and signal latency caused by low-light conditions.
- **EVFI-LL Dataset**: Includes RGB+Event sequences captured in low-light environments, providing a new benchmark.

## Quick Start: Per-Scene Optimization

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/sim2real_code.git
   cd sim2real_code
   ```

2. Download the necessary pre-trained weights and place them in the following directories:
   - **Main model weights**: Place in `pretrained_weights/`
   - **DISTS loss function weights**: Place in `losses/DISTS/weights/`

   **[Download Link Placeholder for Weights](your_download_link_here)**

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run per-scene optimization:
   ```bash
   bash perscene.sh
   ```

This will fine-tune the pre-trained model on your data and perform frame interpolation optimized for each scene.

## Optional: Pretraining the Model

If you wish to pretrain the model from scratch using simulated data, you can do so by following these additional steps:

1. Pretrain the model:
   ```bash
   bash pretrain.sh
   ```

2. After pretraining, proceed with the per-scene optimization as described above.

## Repository Structure

- `dataset/`: Utilities for loading and processing the EVFI-LL dataset.
- `losses/`: Custom loss functions and weights.
- `models/`: Core neural networks for event-based VFI.
- `params/`: Configuration files for different training setups.
- `tools/`: Utility scripts for visualization and debugging.
- `pretrained_weights/`: Pretrained models ready for per-scene optimization.
- `run_network.py`: The main script for training and inference.
- `pretrain.sh`: Script to pretrain the model.
- `perscene.sh`: Script for per-scene optimization.
- `requirements.txt`: Required dependencies.

## Benchmarking with EVFI-LL

The `EVFI-LL` dataset, captured under low-light conditions, is available [here](dataset_link). Please set up the dataset in the `dataset/` folder before running any scripts.

## Results

Our model achieves state-of-the-art performance on the EVFI-LL dataset, outperforming existing methods such as RIFE and TimeLens:

| Method            | PSNR ↑   | SSIM ↑   | LPIPS ↓  |
|-------------------|----------|----------|----------|
| RIFE              | 31.143   | 0.8846   | 0.1420   |
| TimeLens          | 30.548   | 0.8673   | 0.1691   |
| **Ours (Optimized)** | **32.762** | **0.8972** | **0.1172** |

## Citation

If you use this code, please cite our paper:
```bibtex
@inproceedings{Sim2Real2024,
  title={From Sim to Real: Toward General Event-based Low-light Frame Interpolation with Per-scene Optimization},
  author={Anonymous},
  booktitle={SIGGRAPH Asia},
  year={2024}
}
```

## License

This repository is released under the MIT License. See `LICENSE` for more details.

---

