# FuseML Environment Setup

## Quick Start

### 1. Create Conda Environment

```bash
conda create -n fuseml python=3.10 -y
conda activate fuseml
```

### 2. Install Dependencies

```bash
# PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Development tools
pip install numpy pytest pytest-cov black mypy flake8
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch.fx; print('torch.fx available')"
```

## Platform-Specific Notes

### Windows (CUDA 11.8+)
- PyTorch: Installed via official wheels ✓
- Triton: Requires compilation from source or WSL2/Linux environment
  - Option A: Use WSL2 with Linux environment
  - Option B: Install pre-built Windows wheel if available from conda-forge
  - For now, development proceeds without Triton; fusion phase will add it

### Linux/Mac
```bash
pip install triton>=2.0.0
```

## Testing the Environment

Run the backend smoke test:

```bash
python fuseml_backend.py
```

Expected output:
```
[FuseML] INFO — Captured FX graph with 6 nodes — scanning for fusion candidates …
[FuseML] INFO — Fusion candidate found: [input_1 (Linear) -> input_2 (GeLU)]
[FuseML] INFO — Validation passed — compiled output matches eager output
```

## Installed Packages

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Language runtime |
| torch | 2.7.1+cu118 | Deep learning framework |
| numpy | 2.2.6+ | Numerical computing |
| pytest | 7.0+ | Unit testing |
| black | 22.0+ | Code formatting |
| mypy | 0.990+ | Type checking |
| flake8 | 4.0+ | Linting |

## Development Workflow

```bash
# Activate environment
conda activate fuseml

# Format code
black .

# Run type checks
mypy fuseml_backend.py

# Run tests
pytest tests/

# Run linting
flake8 .
```

## Troubleshooting

### ImportError: torch.fx not available
- Ensure torch>=2.0.0 is installed
- `python -c "import torch.fx; print(torch.fx.__version__)"`

### CUDA not detected
- Check NVIDIA driver: `nvidia-smi`
- Verify CUDA version matches wheel: `python -c "import torch; print(torch.cuda.is_available())"`

### Triton compilation fails on Windows
- Triton currently has limited Windows support
- Recommended: Use WSL2 with Linux environment for Triton development
- Or deploy on Linux/Mac for kernel generation phase
