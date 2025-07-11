"""Setup script for Archangel AI Monetization System."""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3.8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")

def create_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created")
    else:
        print("✓ Virtual environment already exists")

def install_requirements():
    """Install required packages."""
    print("📚 Installing requirements...")
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"
    
    # Upgrade pip first
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    print("✓ Requirements installed")

def setup_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed",
        "logs",
        "backups",
        "uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_environment_file():
    """Create .env file from template."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy example to .env
        env_content = env_example.read_text()
        env_file.write_text(env_content)
        print("✓ Created .env file from template")
        print("⚠️  Please edit .env file with your actual configuration")
    elif env_file.exists():
        print("✓ .env file already exists")
    else:
        print("⚠️  No .env.example found, skipping .env creation")

def initialize_database():
    """Initialize the database."""
    print("🗄️  Initializing database...")
    
    # Determine python path
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_path = "venv/bin/python"
    
    try:
        result = subprocess.run([python_path, "main.py", "db", "create"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Database initialized successfully")
        else:
            print(f"⚠️  Database initialization warning: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Could not initialize database: {e}")
        print("   You can run 'python main.py db create' manually later")

def create_admin_user():
    """Create initial admin user."""
    print("👤 Creating admin user...")
    
    email = input("Enter admin email: ").strip()
    if not email:
        print("⚠️  Skipping admin user creation")
        return
    
    password = input("Enter admin password: ").strip()
    if not password:
        print("⚠️  Skipping admin user creation")
        return
    
    # Determine python path
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_path = "venv/bin/python"
    
    try:
        result = subprocess.run([
            python_path, "main.py", "user", "create",
            "--email", email,
            "--password", password,
            "--name", "System Administrator",
            "--admin"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Admin user created successfully")
            print(result.stdout)
        else:
            print(f"❌ Error creating admin user: {result.stderr}")
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")

def test_installation():
    """Test if the installation works."""
    print("\n🧪 Testing installation...")
    
    # Determine python path
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_path = "venv/bin/python"
    
    try:
        # Test imports
        result = subprocess.run([
            python_path, "-c", 
            "from src.core.config import get_settings; from src.api.main import app; print('✓ All imports successful')"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✓ Installation test passed!")
            return True
        else:
            print(f"❌ Installation test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🚀 Archangel AI Monetization System Setup Complete!")
    print("="*60)
    
    # Determine activation command
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        python_cmd = "venv/bin/python"
    
    print("\n📋 Next steps:")
    print("1. Configure your environment:")
    print("   - Edit .env file with your API keys and configuration")
    print("   - Set up OpenAI/Anthropic API keys")
    print("   - Configure Stripe for payments (optional)")
    
    print("\n2. Activate virtual environment:")
    print(f"   {activate_cmd}")
    
    print("\n3. Start the server:")
    print(f"   {python_cmd} main.py server")
    print("   # Or with auto-reload for development:")
    print(f"   {python_cmd} main.py server --reload")
    
    print("\n4. Access the API:")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    print("   - API Base: http://localhost:8000/v1/")
    
    print("\n5. Useful commands:")
    print(f"   {python_cmd} main.py user create --email user@example.com --password secret")
    print(f"   {python_cmd} main.py analytics --days 30")
    print(f"   {python_cmd} main.py db reset")
    
    print("\n💡 Tips:")
    print("   - Check logs/ directory for application logs")
    print("   - Use /v1/models endpoint to see available AI models")
    print("   - Monitor usage with /v1/usage endpoint")
    print("   - Set up webhooks for payment processing")
    
    print("\n🔒 Security reminders:")
    print("   - Change JWT_SECRET in .env file")
    print("   - Use strong passwords")
    print("   - Enable HTTPS in production")
    print("   - Configure proper CORS settings")

def main():
    """Main setup function."""
    print("🏗️  Archangel AI Monetization System Setup")
    print("="*50)
    
    try:
        check_python_version()
        create_virtual_environment()
        install_requirements()
        setup_directories()
        setup_environment_file()
        initialize_database()
        
        # Ask if user wants to create admin user
        create_admin = input("\nCreate admin user now? (y/N): ").strip().lower()
        if create_admin == 'y':
            create_admin_user()
        
        if test_installation():
            print_next_steps()
        else:
            print("\n❌ Setup completed with errors. Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()