#!/bin/bash

# Exit on error
set -e

# Function to print colored output
print_status() {
    echo -e "\033[1;34m==>\033[0m $1"
}

print_error() {
    echo -e "\033[1;31mError:\033[0m $1"
}

print_success() {
    echo -e "\033[1;32mSuccess:\033[0m $1"
}

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS"

# Install system dependencies based on OS
if [ "$OS" = "linux" ]; then
    print_status "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y $(cat system_requirements.txt | grep -v '^#' | tr '\n' ' ')
    print_success "System dependencies installed"
fi

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv lightberry
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source lightberry/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install Python requirements
print_status "Installing Python requirements..."
pip install -r requirements.txt
print_success "Python requirements installed"

# Check if device is registered
print_status "Checking device registration..."
if [ ! -f "device_keys/device_id" ] || [ ! -f "device_keys/private_key.pem" ] || [ ! -f "device_keys/public_key.pem" ]; then
    print_status "Device not registered. Running setup..."
    python setup_device.py
    print_success "Local device setup complete, follow above instructions to register device online"
else
    print_success "Device already registered"
fi

# Add lightberry to PATH permanently
echo 'export PATH="$PATH:$(pwd)/lightberry/bin"' >> ~/.bashrc
echo 'export PATH="$PATH:$(pwd)/lightberry/bin"' >> ~/.bash_profile
echo 'export PATH="$PATH:$(pwd)/lightberry/bin"' >> ~/.profile
print_success "Added lightberry to PATH permanently"

print_success "Installation complete!"
print_status "To activate the virtual environment in the future, just type: lightberry" 

