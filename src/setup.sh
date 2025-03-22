#!/bin/bash

echo "Setting up Conda environments..."

# Define available environments and their YAML files
declare -A env_files
env_files["SB3"]="./envs/SB3.yml"
env_files["CleanRL"]="./envs/CleanRL.yml"

# Function to install an environment
install_env() {
    local env_name=$1
    local env_file=${env_files[$env_name]}

    if [[ -z "$env_file" ]]; then
        echo "ERROR: No environment named '$env_name'"
        return 1
    fi

    echo "Setting up environment: $env_name"
    conda env remove -n "$env_name" -y
    conda env create -f "$env_file"
    bash "./envs/${env_name}.sh"

    if conda info --envs | grep -q "$env_name"; then
        echo "Environment $env_name created successfully."
    else
        echo "Failed to create environment $env_name."
        return 1
    fi
}

# Check if no arguments are provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [-all] [fw1 fw2 ...]"
    exit 1
fi

# Process command-line arguments
if [[ "$1" == "-all" ]]; then
    for env_name in "${!env_files[@]}"; do
        install_env "$env_name"
    done
else
    for env_name in "$@"; do
        install_env "$env_name"
    done
fi

echo "Setup complete!"

