# How Stable Are Baselines? An analysis on the consistency of RL baseline frameworks
This is the repository used for my thesis MSc Computer Science: Artificial Intelligence. This repository contains an installation guide along with the files to reproduce the experiments done in my thesis.

## Setup
Every framework has its own requirements, and we adhered to these requirements in this work. This means that we cannot use a single collection of packages and the Python version, but we have used Conda, an environment management system. To set up the environments, simply run 
```
bash setup.sh -all
```
which sets up Stable Baselines 3, CleanRL, and TorchRL in Python 3.9.0, Python 3.8.3, and Python 3.9.21, respectively. Furthermore, it installs the required packages provided by the installation guide for each framework, respectively. The sequential use of `yml` files and `sh` files allows the setting up of environments beyond a simple \texttt{requirements.txt} file, such as set up via Poetry. This mechanism can also be used to set up Conda environments for new frameworks beyond the scope of this research. To make specific algorithms work, a stub between the automatically called file `train\_[alg].py` and the baseline framework's implementation of said algorithm has to be hand-coded.

## Usage
This work was performed using the compute resources from the Academic Leiden Interdisciplinary Cluster Environment (ALICE) provided by Leiden University. Through SLURM jobs, we gained access to perform repeated experiments in parallel. To set up a job, we have made a simple file that does most of the work. To run a job, write
```
bash job.sh [--fw FW] [--alg ALG] [--env ENV] [--steps STEPS]
    [--rep REP] [--time TIME] [--par PAR] [--mem MEM] [--hps HPS]
```

### Frameworks and algorithms
The parameters that can be supplied are $fw$, which specifies the framework used. This includes the frameworks that are set up, i.e. $fw \in \{SB3, CleanRL, TorchRL\}$. Similarly, the algorithm $alg$ must be an algorithm set up within the framework, like within this work $alg \in \{ppo, td3, sac\}$. Note that setting up the algorithm means that the corresponding file can be run when called. This is the file `./[fw]/train\_[alg].py`. 

### Environment
The environment `env` specifies which environment is used. In this work, we mainly use MuJoCo environments, but any will do as long as it is supported by Gymnasium and compatible with the framework that is used. For example, the ALE suite requires some additional setup, so it cannot be used out of the box. Furthermore, the version of the environment must be specified, that is for example, `env=HalfCheetah-v4`. The parameter `steps` specifies the total number of timesteps taken within the environment. 

### SLURM specifications
Additionally, the parameter `rep` sets up repeated experiments in the form of an SLURM Array Job, where [1-`rep`] jobs are executed. The parameters `time`, `par`, and `mem` have to do with the SLURM job itself: it defines the maximum amount of time allowed for this experiment, the partition it runs on, and the maximum amount of memory this job may use. A good starting point is 24:00:00, 48:00:00, and 72:00:00 for the algorithms ppo, td3, and sac, respectively, regardless of the framework used. This directly implies that ppo-experiments can be run on `par=cpu-medium`, since it adheres to the time limit constraint, while the remaining experiments must be run on `par=cpu-long`. All frameworks have enough memory when `mem=4G` is used. 

### Hyperparameters
Lastly, `hps` refers to the path to the hyperparameter YAML file. This flexibility of providing the path of the hyperparameters instead of the individual hyperparameters allows for high maintenance and the option to keep a clear overview of what hyperparameters are being used in each experiment. All experiments are run on a single CPU \\

### Individual runs
As a sidenote, each experiment can be run individually and outside of the SLURM job context. Run
```
conda activate [fw]
python ./[fw]/train_[alg].py [--env ENV] [--steps STEPS] [--hps HPS]
```
where the parameters within brackets are similar to the aforementioned parameters. 

## Results and plotting
When called, each experiment runs in the Conda environment of the specified framework. The results of each experiment can be found within `./[fw]/results/[alg]\_[env]\_[steps]/[hps]`, where each experiment is placed within a different folder, which could also contain trained models at the end of training. Each of the SLURM files also contains a printed selection of the hyperparameters that are being used, such that it can be traced back to what the experiment contained. Denote that `steps` is part of the path of the results. To effectively make use of this design consideration, when debugging, one could use `steps+1` along with the same settings to separate the experiment from other experiments without the need for complicated workarounds. A collection of jobs can be found in `./exps`.

We provide some utility whilst experimenting, such as formatting YAML hyperparameter files, environment wrapper functions, and job generators. SLURM jobs tend to be unstable, which could cause runs to fail or re-queue jobs. Furthermore, in large-scale experiments, it could become unnoticed that some runs failed instantly. The file `checkruns.py` reads out, in a brute-force manner, each of the experiments to validate that these runs are complete. In this way, it is clear to the researcher which runs should be checked and, if necessary, rerun. 

The plots of the results within this research can be generated via the respective plotting files. When reading out an experiment, a cache is saved. This allows access to the results of the experiments in a faster way, avoiding the need to repeatedly read out many files. Beware that if experiments are rerun, this cache must be manually removed, or it will use the old results when plotting. 