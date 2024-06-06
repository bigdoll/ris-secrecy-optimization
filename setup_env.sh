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
