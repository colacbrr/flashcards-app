#!/usr/bin/env python3
"""
FlashCard Studio Launcher
Double-click this file to run FlashCard Studio
"""

import sys
import os
from pathlib import Path

def main():
    """Launch FlashCard Studio with proper error handling"""
    print("🚀 Starting FlashCard Studio...")
    print("=" * 50)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        # Check if main script exists
        main_script = current_dir / "flashcard_app.py"
        if not main_script.exists():
            print("❌ Main application file not found!")
            print(f"   Looking for: {main_script}")
            print("   Please ensure all files are in the same directory.")
            input("Press Enter to exit...")
            return 1
        
        # Import and run the main application
        from flashcard_app import main as app_main

        
        print("✅ Application loaded successfully!")
        print("   Starting FlashCard Studio...")
        print()
        
        # Run the application
        return app_main()
        
    except ImportError as e:
        print("❌ Import error - Missing dependencies!")
        print(f"   Error: {e}")
        print()
        print("🔧 To fix this issue:")
        print("   1. Run: python install.py")
        print("   2. Or install manually: pip install -r requirements.txt")
        print("   3. Make sure Ollama is installed for full functionality")
        print()
        input("Press Enter to exit...")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Check that all files are present")
        print("   2. Verify Python version (3.8+ required)")
        print("   3. Run install.py to check dependencies")
        print("   4. Check the log file for detailed error information")
        print()
        input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Application closed by user")
        sys.exit(0)
    except SystemExit:
        # Normal exit - don't catch this
        raise
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("The application encountered an unexpected error.")
        input("Press Enter to exit...")
        sys.exit(1)