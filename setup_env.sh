<<<<<<< HEAD
# setup_env.sh
#!/usr/bin/env bash
set -euo pipefail

# Default MOSEK license file path
DEFAULT_LICENSE_PATH="$HOME/Documents/METAWIRELESS/Research-Codes/mosek_lic"

# Utility: ensure command exists
check_command() {
    command -v "$1" >/dev/null 2>&1 || { echo "Error: '$1' not found. Install it and retry." >&2; exit 1; }
}

echo "Checking for required commands..."
check_command python3
check_command pip
check_command tmux || echo "Warning: tmux not installed; session launcher may fail."

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate
# shellcheck disable=SC1091
source venv/bin/activate

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing Python dependencies..."
pip install numpy scipy matplotlib cvxpy pytest wheel

# Install Mosek: try PyPI, then official wheel index
if pip install mosek; then
    echo "Mosek installed via PyPI."
else
    echo "Installing Mosek from official wheel index..."
    pip install -f https://download.mosek.com/stable/wheel/index.html Mosek
fi

echo "Freezing requirements to requirements.txt..."
pip freeze > requirements.txt

# License path setup
echo "Configuring MOSEK license..."
read -rp "Enter MOSEK license file path [${DEFAULT_LICENSE_PATH}]: " LICENSE_PATH
LICENSE_PATH="${LICENSE_PATH:-$DEFAULT_LICENSE_PATH}"
if [ ! -f "$LICENSE_PATH" ]; then
    echo "Warning: license file not found at $LICENSE_PATH" >&2
else
    echo "Found license at $LICENSE_PATH"
fi
export MOSEKLM_LICENSE_FILE="$LICENSE_PATH"

# Persist in .env
if grep -q "^MOSEKLM_LICENSE_FILE" .env 2>/dev/null; then
    sed -i.bak "s|^MOSEKLM_LICENSE_FILE.*|MOSEKLM_LICENSE_FILE=${LICENSE_PATH}|" .env
else
    echo "MOSEKLM_LICENSE_FILE=${LICENSE_PATH}" >> .env
fi

echo "Environment setup complete!"
echo "To run your simulation in tmux, use:"
echo "  python session.py [--session my_session]"













# #!/bin/bash

# # Create virtual environment
# python -m venv venv

# # Activate virtual environment
# source venv/bin/activate

# # Upgrade pip and setuptools
# pip install --upgrade pip setuptools

# # Install packages
# pip install numpy scipy matplotlib cvxpy mosek pytest

# # pip install -f https://download.mosek.com/stable/wheel/index.html Mosek

# # Freeze requirements
# pip freeze > requirements.txt

# # Set MOSEK license path
# export MOSEKLM_LICENSE_FILE="/Users/robertkukufotock/Documents/METAWIRELESS/Research-Codes/mosek_lic" # "/Users/apple/Documents/Metawireless Researcher/Research-Codes/mosek_lic" # Update this path

# echo "Environment setup complete. Don't forget to update the MOSEK license path!"
=======
#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Install packages
pip install numpy scipy matplotlib cvxpy mosek pytest

# pip install -f https://download.mosek.com/stable/wheel/index.html Mosek

# Freeze requirements
pip freeze > requirements.txt

# Set MOSEK license path
export MOSEKLM_LICENSE_FILE="/Users/apple/Documents/Metawireless Researcher/Research-Codes/mosek_lic" # Update this path

echo "Environment setup complete. Don't forget to update the MOSEK license path!"
>>>>>>> origin/main
