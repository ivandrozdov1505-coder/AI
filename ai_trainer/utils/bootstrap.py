"""
Bootstrap module for AI Trainer Platform
Automatically sets up virtual environment and installs dependencies
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def is_venv() -> bool:
    """Check if running inside a virtual environment"""
    return (hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


def get_venv_path(project_root: Path) -> Path:
    """Get the virtual environment path"""
    if os.name == 'nt':  # Windows
        return project_root / ".venv" / "Scripts"
    else:  # Linux/Mac
        return project_root / ".venv" / "bin"


def check_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available via nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def get_gpu_name() -> str:
    """Get NVIDIA GPU name if available"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return ""


def run_command(cmd: list, cwd: Path = None) -> bool:
    """Run a command and return success status"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False


def bootstrap():
    """Main bootstrap function"""
    # Set UTF-8 encoding for Windows
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    project_root = Path(__file__).parent.parent
    venv_python = get_venv_path(project_root) / ("python.exe" if os.name == 'nt' else "python")
    
    # Check if already bootstrapped
    if os.environ.get('AI_TRAINER_BOOTSTRAPPED') == '1':
        print("✓ Running in bootstrapped environment")
        return True
    
    # Check if already in venv
    if is_venv():
        print("✓ Already running in virtual environment")
        os.environ['AI_TRAINER_BOOTSTRAPPED'] = '1'
        return True
    
    print("=" * 60)
    print("AI Trainer Platform - Bootstrap")
    print("=" * 60)
    
    # Create virtual environment if not exists
    if not venv_python.exists():
        print("\n📦 Creating virtual environment...")
        if not run_command([sys.executable, '-m', 'venv', str(project_root / '.venv')]):
            print("❌ Failed to create virtual environment")
            return False
        print("✓ Virtual environment created")
    else:
        print("✓ Virtual environment exists")
    
    # Upgrade pip
    print("\n⬆️  Upgrading pip...")
    run_command([str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # Check for NVIDIA GPU
    has_nvidia = check_nvidia_gpu()
    gpu_name = get_gpu_name() if has_nvidia else ""
    
    if has_nvidia:
        print(f"\n🎮 NVIDIA GPU detected: {gpu_name}")
        print("📦 Installing CUDA dependencies...")
        requirements_file = project_root / 'requirements-cuda.txt'
    else:
        print("\n💻 No NVIDIA GPU detected, using CPU mode")
        print("📦 Installing CPU dependencies...")
        requirements_file = project_root / 'requirements.txt'
    
    if requirements_file.exists():
        if not run_command([str(venv_python), '-m', 'pip', 'install', '-r', str(requirements_file)]):
            print("⚠️  Some dependencies may have failed to install")
    else:
        print(f"⚠️  Requirements file not found: {requirements_file}")
    
    # Mark as bootstrapped and restart
    os.environ['AI_TRAINER_BOOTSTRAPPED'] = '1'
    
    print("\n🔄 Restarting application from virtual environment...")
    main_py = project_root / 'main.py'
    
    # Restart with venv python
    if os.name == 'nt':
        # Windows
        os.execv(str(venv_python), [str(venv_python), str(main_py)] + sys.argv[1:])
    else:
        # Linux/Mac
        os.execv(str(venv_python), [str(venv_python), str(main_py)] + sys.argv[1:])
    
    return True


if __name__ == '__main__':
    bootstrap()
