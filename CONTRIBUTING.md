# Contributing to TinyACE

Thank you for your interest in contributing to TinyACE! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/edge-slm-ace.git
   cd edge-slm-ace
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install with dev dependencies
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting: `black src/ scripts/`
- Maximum line length: 100 characters
- Type hints are encouraged but not required

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests if applicable
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request with a clear description

## Project Structure

- `src/edge_slm_ace/` - Core package code
- `scripts/` - CLI scripts and utilities
- `tests/` - Test suite
- `docs/` - Documentation
- `configs/` - Configuration files

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Error messages and stack traces
- Steps to reproduce
- Expected vs. actual behavior

## Questions?

Open an issue on GitHub for questions or discussions.
