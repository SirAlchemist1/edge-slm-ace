# GitHub Repository Organization

This directory contains GitHub-specific configuration files.

## Files

- `CODEOWNERS` - Code ownership and review assignments
- `workflows/ci.yaml` - Continuous Integration workflow

## Workflow

The CI workflow runs on pull requests and pushes to main branch, executing:
- Code formatting checks (black)
- Test suite execution (pytest)
- Linting checks
