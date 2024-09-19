## Brief introduction
[Project Page](https://openimaginglab.github.io/Sim2Real/) | Video | [Paper](https://arxiv.org/pdf/2406.08090) | Data | [Weight](https://drive.google.com/drive/folders/1sugfKHUswxqUNQKit1G5W8dF1rAKijxq?usp=drive_link)

This repository contains the official implementation of **"From Sim to Real: Toward General Event-based Low-Light Frame Interpolation with Per-scene Optimization"**, presented at **SIGGRAPH Asia 2024**.

Our approach leverages **event cameras** and focuses on enhancing **Video Frame Interpolation (VFI)** in challenging **low-light conditions** by utilizing a **per-scene optimization** strategy. This method adapts the model to specific lighting and camera settings, addressing the trailing artifacts and signal degradation commonly seen in low-light scenarios.
![Visual Comparison](Sim2Real_code/image.png)
## Key Features

- **Per-Scene Optimization**: Fine-tunes a pre-trained model for each scene, significantly improving interpolation results under varying lighting conditions.
- **Low-light Event Correction**: Effectively corrects event-based signal latency and noise in low-light conditions.
- **EVFI-LL Dataset**: Provides RGB+Event sequences captured in low-light conditions for benchmarking.

## Quick Start: Per-Scene Optimization

To immediately apply per-scene optimization with pre-trained models, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/OpenImagingLab/Sim2Real.git
   cd Sim2Real/Sim2Real_code
   ```

2. **Download and place the required weights**:
   - **Main model weights**: Place in `pretrained_weights/`
   - **DISTS loss function weights**: Place in `losses/DISTS/weights/`

   **[Download Weights Link](https://drive.google.com/drive/folders/1sugfKHUswxqUNQKit1G5W8dF1rAKijxq?usp=drive_link)**

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run per-scene optimization**:
   ```bash
   bash perscene.sh
   ```

The above steps will allow you to fine-tune the pre-trained model on specific scenes and perform frame interpolation optimized for each setting.

## Optional: Pretraining the Model

If you wish to pretrain the model from scratch using simulated data:

1. **Pretrain the model**:
   ```bash
   bash pretrain.sh
   ```

2. After pretraining, you can proceed with **per-scene optimization** as described above.

## Directory Structure

- `dataset/`: Code and utilities for dataset preparation and loading.
- `losses/`: Custom loss functions and weights for training.
- `models/`: Neural network models for Sim2Real frame interpolation tasks.
- `params/`: Configuration files for training and evaluation settings.
- `tools/`: Utility scripts for preprocessing and postprocessing tasks.
- `pretrained_weights/`: Place for storing pretrained models.
- `run_network.py`: Main script for training and evaluation.
- `pretrain.sh`: Script for model pretraining.
- `perscene.sh`: Script for per-scene optimization.
- `requirements.txt`: Required Python dependencies.

## EVFI-LL Dataset

The **EVFI-LL** dataset consists of RGB+Event sequences captured in low-light conditions, providing a challenging benchmark for evaluating event-based VFI performance. Download and place the dataset in the `dataset/` directory.

## License Information

### Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />The website content used in this project is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

### Code License
The code in this repository is licensed under the MIT License. See `LICENSE` for more details.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{zhang2024sim,
  title={From Sim-to-Real: Toward General Event-based Low-light Frame Interpolation with Per-scene Optimization},
  author={Zhang, Ziran and Ma, Yongrui and Chen, Yueting and Zhang, Feng and Gu, Jinwei and Xue, Tianfan and Guo, Shi},
  journal={arXiv preprint arXiv:2406.08090},
  year={2024}
}
```
