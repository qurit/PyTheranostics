# Contributing to PyTheranostics

We love your input! We want to make contributing to PyTheranostics as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Setting Up Your Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/qurit/PyTheranostics.git
   cd PyTheranostics
   ```

2. **Create a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install development dependencies:**
   ```bash
   python setup_dev.py
   ```

   This script will:
   - Verify you're in a virtual environment
   - Install the package in editable mode
   - Install all development dependencies (pytest, black, flake8, mypy, etc.)
   - Set up pre-commit hooks for automated code quality checks

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks when you commit code. They only check **files you've modified**, making them non-disruptive to existing code:

- **Code formatting** (black, isort)
- **Linting** (flake8) 
- **Type checking** (mypy)
- **Basic file hygiene** (trailing whitespace, end-of-file)
- **Smoke tests** (quick functionality checks)

**Manual run:** `pre-commit run --all-files` (checks entire codebase)

### Test Categories

We use pytest markers to categorize tests:

```python
# Smoke test (critical test, fails fast if system is not correctly configured)
@pytest.mark.smoke
def test_basic_case():
    # Fast test

# No marker = regular tests
def test_detailed_calculation():
    # Standard or slow test

# Slow tests should only be run occasionally, during merges
@pytest.mark.slow
def test_big_simulation():
    # Slow test
```

### Development Workflow

Once your environment is set up, you can use these commands:

```bash
# Run all tests
pytest

# Run only smoke tests (fast)
pytest -m smoke

# Format code
black .

# Lint code  
flake8

# Type check
mypy pytheranostics

# Run all pre-commit checks manually
pre-commit run --all-files

# Run quality checks (combination)
pytest -m smoke && black --check . && flake8 && mypy pytheranostics
```

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the CHANGES.md with details of your changes.
3. The PR will be merged once you have the sign-off of at least one other developer.

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/qurit/PyTheranostics/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/qurit/PyTheranostics/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under its MIT License. 