#!/usr/bin/env python3
"""
Easy installation script for FlashCard Studio
Run this script to automatically install the application and its dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print installation header"""
    print("=" * 60)
    print("🚀 FlashCard Studio - Easy Installation Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available. Please install pip first.")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    requirements = [
        "PyQt5>=5.15.0",
        "ollama>=0.2.0", 
        "pymupdf>=1.23.0",
        "python-docx>=1.1.0",
        "aiohttp>=3.9.0",
        "requests>=2.31.0"
    ]
    
    for req in requirements:
        print(f"   Installing {req}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", req
            ], check=True, capture_output=True)
            print(f"   ✅ {req}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {req}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def check_ollama():
    """Check if Ollama is installed and running"""
    print("\n🤖 Checking Ollama installation...")
    
    try:
        # Check if ollama command exists
        subprocess.run(["ollama", "--version"], 
                      capture_output=True, check=True)
        print("✅ Ollama is installed")
        
        # Check if Ollama service is running
        try:
            import ollama
            models = ollama.list()
            if models['models']:
                print(f"✅ Ollama is running with {len(models['models'])} model(s)")
                print("   Available models:")
                for model in models['models'][:5]:  # Show first 5 models
                    print(f"     - {model['name']}")
                return True
            else:
                print("⚠️  Ollama is running but no models are installed")
                print("   Run 'ollama pull mistral' to install a model")
                return True
        except:
            print("⚠️  Ollama is installed but may not be running")
            print("   Start Ollama service and try again")
            return True
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama is not installed")
        print("   Please install Ollama from: https://ollama.ai")
        print("   The application will run in mock mode without Ollama")
        return True  # Continue anyway

def create_desktop_shortcut():
    """Create desktop shortcut (Windows/Linux)"""
    print("\n🔗 Creating desktop shortcut...")
    
    try:
        script_path = Path(__file__).parent / "flashcard_app_improved.py"
        if not script_path.exists():
            print("⚠️  Could not create shortcut - main script not found")
            return False
        
        system = platform.system()
        
        if system == "Windows":
            # Create Windows shortcut
            desktop = Path.home() / "Desktop"
            shortcut_path = desktop / "FlashCard Studio.bat"
            
            batch_content = f'''@echo off
cd /d "{script_path.parent}"
python "{script_path}"
pause
'''
            shortcut_path.write_text(batch_content)
            print(f"✅ Desktop shortcut created: {shortcut_path}")
            
        elif system == "Linux":
            # Create Linux desktop entry
            desktop = Path.home() / "Desktop"
            desktop.mkdir(exist_ok=True)
            
            shortcut_path = desktop / "FlashCard-Studio.desktop"
            desktop_content = f'''[Desktop Entry]
Version=1.0
Type=Application
Name=FlashCard Studio
Comment=AI-powered flashcard generator
Exec=python3 "{script_path}"
Icon=applications-education
Terminal=false
Categories=Education;Office;
'''
            shortcut_path.write_text(desktop_content)
            shortcut_path.chmod(0o755)
            print(f"✅ Desktop shortcut created: {shortcut_path}")
            
        else:
            print("⚠️  Desktop shortcut creation not supported on this platform")
            
        return True
        
    except Exception as e:
        print(f"⚠️  Could not create desktop shortcut: {e}")
        return False

def create_launcher_script():
    """Create a launcher script for easy execution"""
    print("\n📜 Creating launcher script...")
    
    try:
        script_dir = Path(__file__).parent
        launcher_path = script_dir / "run_flashcard_studio.py"
        
        launcher_content = '''#!/usr/bin/env python3
"""
FlashCard Studio Launcher
Double-click this file to run FlashCard Studio
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Import and run the main application
    from flashcard_app_improved import main
    
    print("🚀 Starting FlashCard Studio...")
    sys.exit(main())
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure all dependencies are installed.")
    print("Run the install.py script first.")
    input("Press Enter to exit...")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    input("Press Enter to exit...")
    sys.exit(1)
'''
        
        launcher_path.write_text(launcher_content)
        launcher_path.chmod(0o755)
        
        print(f"✅ Launcher created: {launcher_path}")
        print("   Double-click 'run_flashcard_studio.py' to start the application")
        
        return True
        
    except Exception as e:
        print(f"❌ Could not create launcher script: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions"""
    print("\n" + "=" * 60)
    print("🎉 Installation Complete!")
    print("=" * 60)
    print()
    print("📋 How to use FlashCard Studio:")
    print()
    print("1. 🤖 Make sure Ollama is running:")
    print("   • Install Ollama from: https://ollama.ai")
    print("   • Download a model: ollama pull mistral")
    print("   • Start Ollama service")
    print()
    print("2. 🚀 Start the application:")
    print("   • Double-click: run_flashcard_studio.py")
    print("   • Or run: python flashcard_app_improved.py")
    print("   • Or use desktop shortcut (if created)")
    print()
    print("3. 📚 Generate flashcards:")
    print("   • Select a document (PDF, DOCX, or TXT)")
    print("   • Choose your settings")
    print("   • Click 'Generate Flashcards'")
    print("   • Start studying!")
    print()
    print("🔧 Troubleshooting:")
    print("   • If Ollama is not available, the app runs in mock mode")
    print("   • Check logs in: ~/.flashcard_studio/")
    print("   • For support, check the README.md file")
    print()
    print("Happy studying! 🎯")

def main():
    """Main installation process"""
    print_header()
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    if not check_pip():
        return 1
    
    # Install packages
    if not install_requirements():
        print("\n❌ Installation failed. Please check the errors above.")
        return 1
    
    # Check Ollama
    check_ollama()
    
    # Create shortcuts and launchers
    create_desktop_shortcut()
    create_launcher_script()
    
    # Show final instructions
    show_usage_instructions()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        input("\nPress Enter to exit...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)