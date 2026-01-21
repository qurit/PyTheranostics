#!/usr/bin/env python3
"""
Development environment setup script.

Activate a virtual environment or conda environment before running.
"""

import os
import subprocess
import sys


def is_virtual_env():
    """Check if we're running in a virtual environment or conda environment."""
    return (
        hasattr(sys, "real_prefix")  # virtualenv
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)  # venv
        or "VIRTUAL_ENV" in os.environ  # venv environment variable
        or "CONDA_DEFAULT_ENV" in os.environ  # conda environment
        or "CONDA_PREFIX" in os.environ  # conda environment
    )


def main():
    """Set up the development environment."""
    print("PyTheranostics Development Setup")
    print("=" * 35)

    # Check if we're in a virtual environment
    if not is_virtual_env():
        print("❌ ERROR: Not running in a virtual or conda environment!")
        print("\nPlease activate your environment first:")
        print("  Virtual env (Windows): .venv\\Scripts\\activate")
        print("  Virtual env (Linux/Mac): source .venv/bin/activate")
        print("  Conda: conda activate <env_name>")
        sys.exit(1)

    # Detect environment type
    env_type = "conda" if "CONDA_DEFAULT_ENV" in os.environ else "virtual"
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "virtual environment")

    print(f"✅ {env_type.capitalize()} environment detected: {env_name}")
    print(f"Python executable: {sys.executable}")

    # Upgrade pip first
    print("\n📦 Upgrading pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True
    )

    # Install package in editable mode with dev dependencies (includes segmentation tools)
    print("\n🔧 Installing PyTheranostics in editable mode with dev dependencies...")
    print("   This includes segmentation tools (TotalSegmentator, torch, torchvision)")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], check=True)

    # Install and setup pre-commit hooks
    print("\n🪝 Installing pre-commit...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], check=True)

    print("🪝 Setting up pre-commit hooks...")
    subprocess.run(["pre-commit", "install"], check=True)

    # Install missing mypy stubs (optional - requires internet)
    print("\n🎯 Installing missing mypy type stubs...")
    try:
        subprocess.run(["mypy", "--install-types", "--non-interactive"], check=True)
        print("✅ Type stubs installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Could not install some type stubs (this is usually OK)")

    print("\n✅ Development environment setup complete!")
    print("\nYou can now run:")
    print("  pytest                 # Run tests")
    print("  pytest -m smoke        # Run smoke tests only")
    print("  black .                # Format code")
    print("  flake8                 # Lint code")
    print("  pydocstyle pytheranostics  # Check docstring style (NumPy format)")
    print("  mypy pytheranostics    # Type check")
    print("  pre-commit run --all-files  # Run all pre-commit checks")
    print("\n🪝 Pre-commit hooks are now active and will run on every commit")
    print("    To disable pre-commit hooks, run:")
    print("      pre-commit uninstall")


if __name__ == "__main__":
    main()
