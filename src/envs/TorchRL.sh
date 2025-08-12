#!/bin/bash

# Additional arguments can be applied for installation
# For TorchRL installation of torch after env setup, along with omegaconf

conda run -n TorchRL pip install torch torchrl omegaconf tqdm "gymnasium[mujoco]" tensorboard
