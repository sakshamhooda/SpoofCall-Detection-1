# SafeCall: AI-Powered Spoof Call Detection System

This repository contains the implementation of an AI-powered spoof call detection system as described in the paper "SafeCall: An AI-Powered Spoof Call Detection System and a National Call Routing Framework to combat digital arrests".

## Project Structure

```
SpoofCall-Detection-1/
├── configs/
│   └── config.yaml          # Configuration file
├── src/
│   ├── data/
│   │   └── data_loader.py   # Data loading utilities
│   ├── evaluation/
│   │   └── metrics.py       # Evaluation metrics
│   ├── models/
│   │   ├── base_model.py    # Base model class
│   │   ├── cycle_gan.py     # CycleGAN implementation
│   │   ├── star_gan.py      # StarGAN implementation
│   │   ├── spectrogram_cnn.py  # CNN for spectrogram analysis
│   │   └── sequential_lstm.py  # LSTM for sequential analysis
│   ├── training/
│   │   └── trainer.py       # Training utilities
│   └── train.py            # Main training script
└── tests/                  # Unit tests

## Features

### Voice Spoof Detection
- CycleGAN for voice conversion detection
- StarGAN for multi-domain voice conversion detection
- CNN-based spectrogram analysis
- LSTM-based sequential analysis with attention
- Advanced evaluation metrics including:
  - t-DCF (tandem Detection Cost Function)
  - SRR (Spoofing Recognition Rate)
  - LLRC (Log-Likelihood Ratio Cost)

### Video Deepfake Detection
- Support for FaceForensics++ dataset
- Vision Transformer (ViT) implementation
- EfficientNet-based detection
- Metrics for visual analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SpoofCall-Detection-1.git
cd SpoofCall-Detection-1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a specific model using the provided script:

```bash
python src/train.py --config configs/config.yaml --model [cyclegan|stargan|cnn|lstm] --data_dir path/to/data
```

### Configuration

Modify `configs/config.yaml` to adjust model parameters, training settings, and data processing options.

### Evaluation

The system provides comprehensive evaluation metrics including:
- Classification metrics (Accuracy, Precision, Recall, F1-Score)
- GAN-specific metrics (FID, IS, Wasserstein Distance)
- Voice spoofing metrics (t-DCF, SRR, LLRC)
- Vision Transformer metrics (Attention Distance, Cross-Attention Consistency)

## Model Architectures

### CycleGAN
- Utilizes cycle consistency loss for unpaired voice conversion detection
- Generator and discriminator networks with residual blocks
- Identity mapping preservation

### StarGAN
- Multi-domain voice conversion detection
- Domain classification with auxiliary classifier
- Adaptive instance normalization

### Spectrogram CNN
- Deep convolutional architecture for spectrogram analysis
- Batch normalization and dropout for regularization
- Adaptive pooling for variable input sizes

### Sequential LSTM
- Bidirectional LSTM with attention mechanism
- Multi-layer architecture with dropout
- Self-attention for temporal dependencies

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hooda2024safecall,
  title={SafeCall: An AI-Powered Spoof Call Detection System and a National Call Routing Framework to combat digital arrests},
  author={Hooda, Saksham},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.