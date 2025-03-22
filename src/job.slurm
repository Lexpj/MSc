#!/bin/bash -l
#SBATCH --mail-user="lex.janssens@live.nl"
#SBATCH --mail-type="ALL"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# EXPERIMENT VARIABLES
fw=$1
env=$2
steps=$3
rep=$4
alg=$5
hps=$6

# Source the Python virtual environment
cd $HOME/Documents/MSc/${fw}
conda activate ${fw}

echo "[$SHELL] $$############################################"
echo "[$SHELL] ## Starting Python $(python --version)"
echo "[$SHELL] ## Training using algorithm ${alg} from framework ${fw}"
echo "[$SHELL] ## On environment ${env} with ${steps} timesteps"
echo "[$SHELL] ## This is repetition ${SLURM_ARRAY_TASK_ID}/${rep}"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
echo "[$SHELL] ##############################################"
# get the current working directory

export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD
echo "[$SHELL] ## current python version: "$(python --version)
echo "[$SHELL] ## python from path: "$(which python)
echo "[$SHELL] ## list of packages: "$(conda list)
echo "[$SHELL] ##############################################"
# Run the file
echo "[$SHELL] ## Start run script"
python train_${alg}.py --env=${env} --steps=${steps} --alg=${alg} --rep=${SLURM_ARRAY_TASK_ID} --fw=${fw} --hps=${hps} --save_model --log

echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"
