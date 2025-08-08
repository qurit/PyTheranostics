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
    
    print("\n✅ Development environment setup complete!")
    print("\nYou can now run:")
    print("  pytest                 # Run tests")
    print("  black .                # Format code")
    print("  flake8                 # Lint code")
    print("  mypy pytheranostics    # Type check")


if __name__ == "__main__":
    import os  # Import here to avoid issues if script fails early
    main()
