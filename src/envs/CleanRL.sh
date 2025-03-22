#!/bin/bash

# Additional arguments can be applied for installation
# For CleanRL, setup the packages via their requirement files. 


conda run -n CleanRL pip install -r "./envs/req_cleanrl/requirements.txt"
conda run -n CleanRL pip install -r "./envs/req_cleanrl/requirements-atari.txt"
conda run -n CleanRL pip install -r "./envs/req_cleanrl/requirements-mujoco.txt"
