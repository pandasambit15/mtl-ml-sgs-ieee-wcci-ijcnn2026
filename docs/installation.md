# Installation

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CPU or CUDA)
- 8 GB RAM minimum; 32 GB recommended for full MONC datasets
- GPU with ≥ 8 GB VRAM recommended for training

## From source (recommended)

```bash
git clone https://github.com/your-org/ml-sgs-turbulence.git
cd ml-sgs-turbulence

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# CPU-only
pip install -r requirements.txt
pip install -e .

# GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Development install

```bash
pip install -e ".[dev]"
pre-commit install
```

## Verify installation

```bash
python -c "import ml_sgs; print(ml_sgs.__version__)"
pytest tests/ -x -q
```
