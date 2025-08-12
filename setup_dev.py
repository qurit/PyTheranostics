#!/usr/bin/env python3
"""
Development environment setup script.
Activate a virtual environment before running.
"""

import sys
import subprocess
from pathlib import Path


def is_virtual_env():
    """Check if we're running in a virtual environment."""
    return (
        hasattr(sys, 'real_prefix') or  # virtualenv
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or  # venv
        'VIRTUAL_ENV' in os.environ  # environment variable
    )


def main():
    """Set up the development environment."""
    print("PyTheranostics Development Setup")
    print("=" * 35)
    
    # Check if we're in a virtual environment
    if not is_virtual_env():
        print("❌ ERROR: Not running in a virtual environment!")
        print("\nPlease activate your virtual environment first:")
        print("  Windows: .venv\\Scripts\\activate")
        print("  Linux/Mac: source .venv/bin/activate")
        sys.exit(1)
    
    print("✅ Virtual environment detected")
    print(f"Python executable: {sys.executable}")
    
    # Upgrade pip first
    print("\n📦 Upgrading pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install package in editable mode with dev dependencies
    print("\n🔧 Installing PyTheranostics in editable mode with dev dependencies...")
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
    print("  mypy pytheranostics    # Type check")
    print("  pre-commit run --all-files  # Run all pre-commit checks")


if __name__ == "__main__":
    import os  # Import here to avoid issues if script fails early
    main()
