# MVPinn: Integrating Milne-Eddington Inversion with Physics-Informed Neural Networks for GST/NIRIS Observations

## Overview

This project implements a Physics-Informed Neural Network (PINN) for performing Milne-Eddington inversion of solar Stokes profiles. The neural network learns to predict ME parameters (magnetic field strength, inclination, azimuth, etc.) from observed Stokes I, Q, U, V profiles, while being constrained by the physics of the ME forward model.

## Key Features

- **Physics-Informed Architecture**: The ME forward model is integrated into the neural network, allowing end-to-end training with physics constraints
- **Efficient Training**: Uses PyTorch for GPU-accelerated training
- **Batch Processing**: Supports batch inference for large datasets
- 
### Data Description

- **Input**: Flattened Stokes profiles `[batch_size, 4*n_wavelengths]`
- **Output**: 9 ME parameters per pixel:
  - B: Magnetic field strength 
  - θ: Inclination 
  - χ: Azimuth
  - η₀: Line-to-continuum opacity ratio 
  - ΔλD: Doppler width 
  - a: Damping parameter
  - λ₀: Line center shift 
  - B₀, B₁: Source function parameters

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU 
- Conda (for environment management)
- torch >= 1.10.0
### Setup

1. Clone the repository:
```bash
git clone https://github.com/Reenneer/MVPinn.git
cd MVPinn
```

2. Create and activate the conda environment:
```bash
conda create -n MVPinn-torch python=3.8
conda activate MVPinn-torch
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
from src import train_me_pinn
from pathlib import Path

# Train a new model
model = train_me_pinn(
    data_file='path/to/stokes_data.fts',
    n_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    optimizer_type='adam'
)
```

### Inference

```python
from src import MEInversionPINN, infer_with_pinn
import torch

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MEInversionPINN(n_wavelengths=50).to(device)
model.load_state_dict(torch.load('path/to/model.pt'))
model.eval()

# Perform inference
parameters_map, stokes_fitted = infer_with_pinn(
    model, 
    'path/to/stokes_data.fts',
    output_dir='results/'
)
```

### Complete Workflow

See `examples/example.ipynb` for a complete example of training and inference.

## Project Structure

```
PINN4ME/
├── src/                    # Source code
│   ├── Training.py        # Model definition and training
│   ├── DataLoader.py      # Data loading and preprocessing
│   ├── Infer.py           # Inference functions
│   ├── process_main.py    # Main processing pipeline
│   ├── ME_utils.py        # ME model utilities
│   └── visualization.py   # Visualization utilities
├── examples/              # Example notebooks
│   └── example.ipynb      # Main example
├── docs/                  # Documentation
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Citation

If you use this code in your research, please cite:

```bibtex
@ARTICLE{2025arXiv250709430L,
       author = {{Li}, Qin and {Shen}, Bo and {Jiang}, Haodi and {Yurchyshyn}, Vasyl B. and {Baildon}, Taylor and {Yi}, Kangwoo and {Cao}, Wenda and {Wang}, Haimin},
        title = "{MVPinn: Integrating Milne-Eddington Inversion with Physics-Informed Neural Networks for GST/NIRIS Observations}",
      journal = {arXiv e-prints},
     keywords = {Solar and Stellar Astrophysics, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = jul,
          eid = {arXiv:2507.09430},
        pages = {arXiv:2507.09430},
          doi = {10.48550/arXiv.2507.09430},
archivePrefix = {arXiv},
       eprint = {2507.09430},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250709430L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- BBSO/NIRIS for observational data
- PyTorch team for the deep learning framework

## Contact

For questions or issues, please contact [ql47@njit.edu].

# MVPinn
