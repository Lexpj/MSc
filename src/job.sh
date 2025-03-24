args=()

for argument in "$@"
do
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}"
    fi
done

args+=('--fq' ${fw})
args+=('--env' ${env})
args+=('--steps' ${steps})
args+=('--alg' ${alg})
args+=('--par' ${par})
args+=('--time' ${time})
args+=('--mem' ${mem})

# Repetitions by default 5
if [[ ${rep} == "" ]]; then
    key="--rep"
    value=5
    v="${key/--/}"
    declare $v="${value}"
else
    args+=('--rep' ${rep})
fi
# Hyperparameters by default the file corresponding to FW and ALG default hps of specific environment set
if [[ ${hps} == "" ]]; then
    key="--hps"
    value="./hps/${fw}_${alg}_default"
    value+=$(python -c "import env_utils; print(env_utils.identify_env(\"${env}\"))")
    value+=".yml"
    v="${key/--/}"
    declare $v="${value}"
else
    args+=('--hps' ${hps})
fi

# If the environment is an Atari environment, it should be preprocessed such that ALE/[env] is not a folder
env=$(python -c "import env_utils; print(env_utils.format_atari(\"${env}\"))")

# Initial run to catch errors before sending it to HPC

if [[ $( python ./check.py --steps=${steps} --env=${env} --fw=${fw} --rep=${rep} --alg=${alg} --hps=${hps} ) ]]; then
    echo "Check successful, starting SBATCH"
    hpsconfig="${hps##*_}"
    hpsconfig="${hpsconfig%.yml}"
    sbatch -J "${fw}_${env}_${steps}" -o "./${fw}/results/${alg}_${env}_${steps}/${hpsconfig}/%x_%j.out" -a "1-${rep}" -t "${time}" --mem "${mem}" -p "${par}" job.slurm "${fw}" "${env}" "${steps}" "${rep}" "${alg}" "${hps}"
    # By default, SLURM jobs have the --log and --save_model flag
else
    echo "Check unsuccessful, starting SBATCH aborted."
fi

