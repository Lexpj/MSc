# How Stable Are Baselines? An analysis on the consistency of RL baseline frameworks
This is the repository used for my thesis MSc Computer Science: Artificial Intelligence. This repository contains an installation guide along with the files to reproduce the experiments done in my thesis.

## Installation
Since each baseline framework has variable dependencies, and only works on a subset of (non-overlapping) Python versions, the installation is tedious. However, Conda has made this easy to set up via independent environments for each framework. The installation is self-contained, and can be started via
```
bash setup.sh -all
```
to install all conda environments such that all frameworks can be used. If you only want a subset of environments, you can specify this via individually specifying which environments you want to install:
```
bash setup.sh SB3 Tianshou CleanRL TorchRL
```

## Experimentation

### SLURM jobs
If you want to execute a SLURM job of an experiment, you can via

```
cd src
bash job.sh [--fw FW] [--env ENV] [--steps STEPS] [--rep REP] [--time TIME] [--par PAR] [--mem MEM] [--hps HPS]
```

The frameworks `--fw` you may specify are:
- SB3 (Stable Baselines)
- CleanRL

Environments `--env` may be specified, but should be supported by Gymnasium, Mujoco or Atari. 

Steps `--steps` is the total amount of timesteps the model is trained for. This should be an integer input

Repetitions `--rep` is the amount of individual independent runs done. This should be an integer. By default, 5 repetitions are done.

The time `--time` argument specifies the total runtime the experiment may take (individually accounted for, not accumulated). If the time limit is exceeded, the experiment will stop. 

The partition `--par` can be specified to some node of a HPC. Note that some nodes may have time limits built in, i.e. the use of some node may not take more than 1 hour. 

The memory `--mem` is the maximum memory the experiment may use. Using 4G should be enough for all experiments.env

The hyperparameters `--hps` may be specified by passing the path to this YML file. This is an optional argument. If nothing is passed, the default parameters will be used of that specific algorithm of that specific framework.

Logging `--log` as well as saving the model after training via `--save_model` are automatically flagged when applying a SLURM job. This is not the case for individual runs (read below), and may be used as optional flags.

When applying a SLURM job, an initial run will be done on 1 timestep as a control measure to see whether the environment(s) are correctly set up. This allows crash reports to be available before the HPC is accessed, avoiding queuing for errors.

### Individual runs
Individual runs may be performed as well via 

```
conda activate [FW]
python ./[FW]/train_[ALG].py [--env=ENV] [--steps=STEPS] [--hps] [--log] [--save_model]
```
where the parameters in caps should be replaced (and square brackets removed). See a description of the parameters in SLURM jobs.