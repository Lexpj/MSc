#!/bin/bash

# Additional arguments can be applied for installation
# For tianshou, setup the packages via their requirement files. 

cd tianshou
conda run -n tianshou poetry install --extras "atari classic_control mujoco  argparse"

