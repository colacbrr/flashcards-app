#!/usr/bin/env python3
"""
Build script for creating standalone executables of FlashCard Studio
This script uses PyInstaller to create platform-specific executables
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import platform

def print_header():
    """Print build header"""
    print("=" * 60)
    print("🔨 FlashCard Studio - Executable Builder")
    print("=" * 60)
    print()

def check_pyinstaller():
    """Check if PyInstaller is available"""
    try:
        import PyInstaller
        print("✅ PyInstaller is available")
        return True
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
            print("✅ PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install PyInstaller")
            return False

def clean_build_dirs():
    """Clean previous build directories"""
    print("🧹 Cleaning previous build directories...")
    
    dirs_to_clean = ["build", "dist", "__pycache__"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   Removed {dir_name}/")
    
    # Remove .spec files
    for spec_file in Path(".").glob("*.spec"):
        spec_file.unlink()
        print(f"   Removed {spec_file}")

def create_spec_file():
    """Create PyInstaller spec file with custom configuration"""
    print("📝 Creating PyInstaller spec file...")
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['flashcard_app_improved.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('README.md', '.'),
    ],
    hiddenimports=[
        'PyQt5.QtCore',
        'PyQt5.QtGui', 
        'PyQt5.QtWidgets',
        'ollama',
        'pymupdf',
        'docx',
        'aiohttp',
        'requests',
        'json',
        'hashlib',
        'asyncio',
        'concurrent.futures',
        'logging'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'PIL',
        'tkinter'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FlashCardStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    cofile_version_info=None,
    icon=None,
    version='version_info.txt'
)
'''

    with open("flashcard_studio.spec", "w") as f:
        f.write(spec_content)
    
    print("✅ Spec file created: flashcard_studio.spec")

def create_version_info():
    """Create version info file for Windows builds"""
    print("📋 Creating version info...")
    
    version_info = '''# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(2, 0, 0, 0),
    prodvers=(2, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'FlashCard Studio'),
        StringStruct(u'FileDescription', u'AI-powered flashcard generator'),
        StringStruct(u'FileVersion', u'2.0.0.0'),
        StringStruct(u'InternalName', u'FlashCardStudio'),
        StringStruct(u'LegalCopyright', u'Copyright (C) 2024 FlashCard Studio'),
        StringStruct(u'OriginalFilename', u'FlashCardStudio.exe'),
        StringStruct(u'ProductName', u'FlashCard Studio'),
        StringStruct(u'ProductVersion', u'2.0.0.0')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    
    with open("version_info.txt", "w") as f:
        f.write(version_info)
    
    print("✅ Version info created: version_info.txt")

def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building executable...")
    print("   This may take several minutes...")
    
    try:
        # Build command
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--onefile",
            "--windowed",
            "--name", "FlashCardStudio",
            "--distpath", "dist",
            "--workpath", "build",
            "flashcard_studio.spec"
        ]
        
        # Add platform-specific options
        system = platform.system()
        if system == "Windows":
            cmd.extend(["--version-file", "version_info.txt"])
        
        # Run PyInstaller
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Executable built successfully!")
            return True
        else:
            print("❌ Build failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def create_installer_package():
    """Create installation package with all necessary files"""
    print("📦 Creating installation package...")
    
    # Create package directory
    package_dir = Path("FlashCard_Studio_v2.0")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    
    package_dir.mkdir()
    
    # Copy executable
    executable_name = "FlashCardStudio.exe" if platform.system() == "Windows" else "FlashCardStudio"
    exe_path = Path("dist") / executable_name
    
    if exe_path.exists():
        shutil.copy2(exe_path, package_dir / executable_name)
        print(f"   ✅ Copied executable: {executable_name}")
    else:
        print(f"   ❌ Executable not found: {exe_path}")
        return False
    
    # Copy documentation
    files_to_copy = [
        ("README.md", "README.md"),
        ("requirements.txt", "requirements.txt"),
        ("install.py", "install.py")
    ]
    
    for src, dst in files_to_copy:
        if Path(src).exists():
            shutil.copy2(src, package_dir / dst)
            print(f"   ✅ Copied: {dst}")
    
    # Create quick start guide
    quick_start = package_dir / "QUICK_START.txt"
    quick_start_content = """
==========================================================
🚀 FlashCard Studio v2.0 - Quick Start Guide
==========================================================

PREREQUISITES:
1. Install Ollama from: https://ollama.ai
2. Download a model: ollama pull mistral
3. Make sure Ollama service is running

GETTING STARTED:
1. Double-click FlashCardStudio.exe (or FlashCardStudio on Mac/Linux)
2. Select a document (PDF, DOCX, or TXT)
3. Configure your settings
4. Click "Generate Flashcards"
5. Start studying!

TROUBLESHOOTING:
- If you get "Ollama not available", install Ollama first
- The app works in mock mode for testing without Ollama
- Check README.md for detailed instructions
- For support, visit our GitHub page

SYSTEM REQUIREMENTS:
- Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- 4GB RAM (8GB recommended)
- 2GB free disk space

Happy studying! 🎓
==========================================================
"""
    quick_start.write_text(quick_start_content)
    print("   ✅ Created: QUICK_START.txt")
    
    # Create batch file for Windows
    if platform.system() == "Windows":
        batch_file = package_dir / "Run_FlashCard_Studio.bat"
        batch_content = '''@echo off
title FlashCard Studio v2.0
echo.
echo ========================================
echo   🚀 Starting FlashCard Studio v2.0
echo ========================================
echo.
echo Please wait while the application loads...
echo.
start "" "FlashCardStudio.exe"
echo.
echo FlashCard Studio is now running!
echo You can close this window.
echo.
pause
'''
        batch_file.write_text(batch_content)
        print("   ✅ Created: Run_FlashCard_Studio.bat")
    
    print(f"✅ Installation package created: {package_dir}")
    return True

def create_zip_archive():
    """Create ZIP archive of the installation package"""
    print("🗜️ Creating ZIP archive...")
    
    package_dir = "FlashCard_Studio_v2.0"
    if not Path(package_dir).exists():
        print("❌ Package directory not found")
        return False
    
    system = platform.system()
    arch = platform.machine()
    zip_name = f"FlashCard_Studio_v2.0_{system}_{arch}"
    
    try:
        shutil.make_archive(zip_name, 'zip', package_dir)
        print(f"✅ ZIP archive created: {zip_name}.zip")
        return True
    except Exception as e:
        print(f"❌ Failed to create ZIP archive: {e}")
        return False

def cleanup_build_files():
    """Clean up intermediate build files"""
    print("🧹 Cleaning up build files...")
    
    files_to_clean = [
        "flashcard_studio.spec",
        "version_info.txt"
    ]
    
    dirs_to_clean = [
        "build",
        "__pycache__"
    ]
    
    for file_name in files_to_clean:
        if Path(file_name).exists():
            Path(file_name).unlink()
            print(f"   Removed {file_name}")
    
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"   Removed {dir_name}/")

def show_build_summary():
    """Show build summary and instructions"""
    print("\n" + "=" * 60)
    print("🎉 Build Complete!")
    print("=" * 60)
    
    system = platform.system()
    arch = platform.machine()
    
    print(f"\n📊 Build Summary:")
    print(f"   Platform: {system} {arch}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Package: FlashCard_Studio_v2.0_{system}_{arch}.zip")
    
    print(f"\n📁 Files Created:")
    package_dir = Path("FlashCard_Studio_v2.0")
    if package_dir.exists():
        for item in sorted(package_dir.iterdir()):
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   {item.name} ({size_mb:.1f} MB)")
    
    print(f"\n🚀 Distribution:")
    print(f"   1. Share the ZIP file: FlashCard_Studio_v2.0_{system}_{arch}.zip")
    print(f"   2. Users should extract and run the executable")
    print(f"   3. Make sure users have Ollama installed for full functionality")
    
    print(f"\n⚠️ Important Notes:")
    print(f"   • The executable is platform-specific ({system})")
    print(f"   • Users need Ollama installed separately")
    print(f"   • First run may be slower due to initialization")
    print(f"   • Recommend testing on clean system before distribution")

def main():
    """Main build process"""
    print_header()
    
    # Check main script exists
    if not Path("flashcard_app_improved.py").exists():
        print("❌ Main script 'flashcard_app_improved.py' not found")
        return 1
    
    # Check and install PyInstaller
    if not check_pyinstaller():
        return 1
    
    # Clean previous builds
    clean_build_dirs()
    
    # Create build files
    create_version_info()
    create_spec_file()
    
    # Build executable
    if not build_executable():
        print("\n❌ Build failed!")
        return 1
    
    # Create installation package
    if not create_installer_package():
        print("\n❌ Package creation failed!")
        return 1
    
    # Create ZIP archive
    create_zip_archive()
    
    # Cleanup
    cleanup_build_files()
    
    # Show summary
    show_build_summary()
    
    print("\n✅ Build process completed successfully!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        input("\nPress Enter to exit...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)