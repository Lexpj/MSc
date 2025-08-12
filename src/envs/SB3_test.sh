#!/bin/bash

# Additional arguments can be applied for installation
# For SB3 not needed

conda run -n SB3_test conda install -c conda-forge glew
conda run -n SB3_test conda install -c conda-forge mesalib
conda run -n SB3_test conda install -c anaconda mesa-libgl-cos6-x86_64
conda run -n SB3_test conda install -c menpo glfw3
conda run -n SB3_test conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl


conda run -n SB3_test conda env config vars set MJLIB_PATH=$HOME/.mujoco/mujoco210/bin/libmujoco210.so LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210 
conda run -n SB3_test conda env config vars set MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt

conda run -n SB3_test pip install mujoco-py

