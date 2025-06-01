# RIS-Based Secrecy Optimization Project

Welcome to the RIS-based Secrecy Optimization Project! This project focuses on optimizing the secrecy performance of a wireless communication system using reconfigurable intelligent surfaces (RIS). The project involves simulating and optimizing communication environments, leveraging advanced algorithms and techniques.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Running Simulations](#running-simulations)
  - [Running Unit Tests](#running-unit-tests)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [References](#references)

## Overview

This project proposes a low complexity optimization algorithm to maximize the secrecy energy efficiency (SEE) in the uplink of a wireless network aided by an RIS. The project optimizes the transmit powers of mobile users and the RIS reflection coefficients. The primary focus is on both active and passive RIS configurations to achieve energy-efficient secure communications.

## Features

- **Channel Environment Simulation**: Simulate different channel environments with various configurations.
- **Secrecy Optimization**: Optimize RIS configurations to improve secrecy.
- **Visualization**: Visualize results using 2D and 3D plots.
- **OOP Design**: Efficient code management using Object-Oriented Programming principles.
- **Unit Testing**: Comprehensive unit tests to ensure code reliability.

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8+
- Virtualenv
- MOSEK Solver (for optimization tasks)

### Setup

1. **Clone the Repository**

   ```sh
   git clone https://github.com/your-repo/ris-secrecy-optimization.git
   cd ris-secrecy-optimization
   ```

2. **Setup the Virtual Environment**

   ```sh
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

3. **Run the Setup Script**

   ```sh
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

4. **Set the MOSEK License Path**
   Add the following line to your .bashrc or .zshrc file:

   ```sh
   export MOSEKLM_LICENSE_FILE="/path/to/your/mosek/license"
   ```

   Then, source your updated profile

   ```sh
   source ~/.bashrc # or `source ~/.zshrc`
   ```

## Usage

### Running Simulations

To start the simulation, run:

```sh
python main.py
```

### Running Unit Tests

To run all the unit tests:

```sh
python -m unittest discover tests
```

To run a specific test:

```sh
python -m unittest tests.test_visualization.TestPlotter.test_plot_results
```

## Project Structure

ris-secrecy-optimization/
├── main.py
├── README.md
├── requirements.txt
├── setup_env.sh
├── docs
├── data/
│ ├── channel_samples/
│ ├── ris_coefficients/
│ └── outputs/
├── src/
│ ├── \_\_init\_\_.py
│ ├── sysconfig.py
│ ├── utils.py
│ ├── gamma_utils.py
│ ├── power_utils.py
│ ├── optimizers.py
│ └── visualization.py
└── tests/
├── \_\_init\_\_.py
├── test_sysconfig.py
├── test_utils.py
├── test_gamma_utils.py
├── test_power_utils.py
├── test_optimizers.py
└── test_visualization.py

## Documentation

For detailed documentation on each module and its functions, please refer to the docstrings within the source code files in the src directory.

## References

For more details on the concepts and algorithms used in this project, please refer to the source research paper provided in the repository (inside the doc folder!).
