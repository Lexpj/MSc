#!/bin/bash

# Additional arguments can be applied for installation
# For tianshou, setup via poetry

cd tianshou
conda run -n tianshou pip install poetry
conda run -n tianshou poetry install --extras "classic_control mujoco argparse"

# Tianshou has old (numpy < 2) binaries, which for ease of use are not recomputed for numpy >= 2. However, the poetry install includes a numpy version >= 2. This manually forces an install for a lower python version.
conda run -n tianshou pip install "numpy<2"

