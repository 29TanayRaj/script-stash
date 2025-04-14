import subprocess
import sys
import os

VENV_DIR = input('Enter your enviroment name:')
PYTHON_VERSION = input("Enter the desired Python version (e.g., python3.11 or python3.9): ")

# Function to check if the python version is installed or not on the system
def find_python_executable(version_cmd):
    try:
        result = subprocess.run(
            [version_cmd, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            return version_cmd
    except FileNotFoundError:
        pass
    return None

# Function to create virtual environment 
def create_venv_env():
    
    print(f"Creating virtual environment in '{VENV_DIR}'....")

    # Create the virtual environment
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)

    print(f"Virtual environment '{VENV_DIR}' created successfully!")

    #prints the scipt path
    activate_script = os.path.join(VENV_DIR, "Scripts", "activate") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "activate")
    print(f"To activate the environment, run:\nsource {activate_script}")

# Function to install libraries from a requierments.txt file.
def install_requirements():

    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "python")

    # installs libraries from any requiements.text file
    print('Installing python packages.....')
    subprocess.run([venv_python, "-m", "pip", "install", "-r","requirements.txt"], check=True)

# Gives the information about the environment 
def print_info():

    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "python")

    # Python version in the environment
    print("Checking Python version in the virtual environment...")
    subprocess.run([venv_python, "--version"], check=True)

    # libraries installed in the virtual environment
    print(f"Python libraries installed in the environment '{VENV_DIR}' are: ")
    subprocess.run([venv_python,'-m','pip','list'])

if __name__ == "__main__":
    create_venv_env()
    install_requirements()
    print_info()