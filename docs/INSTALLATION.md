# Installation Guide

## System Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended (CUDA-capable GPU or Apple Silicon)
- **Disk Space**: ~10GB for models and dependencies

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/SirAlchemist1/edge-slm-ace.git
cd edge-slm-ace
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### 4. Verify Installation

```bash
# Run smoke test
python -m scripts.smoke_test
```

## Device-Specific Setup

### CUDA (NVIDIA GPU)

1. Install CUDA toolkit (11.8 or higher)
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### MPS (Apple Silicon)

1. macOS 12.3+ required
2. PyTorch automatically uses MPS if available
3. No additional setup needed

### CPU Only

1. Install CPU-only PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

## Troubleshooting

### Common Issues

**Issue**: `torch.cuda.is_available()` returns False
- **Solution**: Verify CUDA installation and PyTorch CUDA version compatibility

**Issue**: Out of memory errors
- **Solution**: Use smaller models or reduce batch size

**Issue**: Model download fails
- **Solution**: Check internet connection and HuggingFace access

## Development Setup

For development, install additional dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- `pytest` - Testing framework
- `black` - Code formatter
