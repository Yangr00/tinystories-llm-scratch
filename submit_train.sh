#!/bin/bash

# --- SLURM JOB SUBMISSION SCRIPT ---
#
# This script is designed to be submitted with `sbatch`a
# sbatch submit_train.sh

# --- RESOURCE REQUESTS ---
#
#SBATCH --job-name=llm-scratch-train  # A name for your job
#SBATCH --output=slurm-%j.out         # Standard output and error log (%j is the job ID)
#SBATCH --error=slurm-%j.err          # Separate error log
#SBATCH --partition=interactive-gpu   # The partition to run on (gpu is a common one, but you can also use interactive-gpu)
#SBATCH --gres=gpu:l40s:1             # Request one L40S GPU
#SBATCH --nodes=1                     # Request a single node
#SBATCH --ntasks-per-node=1           # Request a single task (process)
#SBATCH --mem-per-cpu=48gb            # Memory per CPU core
#SBATCH --time=4:00:00                # Time limit (HH:MM:SS). 30 minutes is plenty for this test.

# --- MODIFICATION: Add Email Notifications ---
#SBATCH --mail-user=ruan0073@umn.edu   # Your email address
#SBATCH --mail-type=ALL               # Send email on job ALL events (BEGIN, END, FAIL)

# --- COMMANDS TO RUN ON THE COMPUTE NODE ---

echo "--- Job Information ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "-----------------------"

# --- MODIFICATION: Set up the Conda Environment ---
# Initialize Conda for non-interactive shells
eval "$(conda shell.bash hook)"
# Activate your specific Conda environment
conda activate py-h100
echo "Conda environment activated: $(which python)"
echo "-----------------------"

# Navigate to the directory where the job was submitted from.
# This makes file paths (like data/tinystories_valid.npy) work correctly.
cd $SLURM_SUBMIT_DIR

echo "--- GPU Information ---"
nvidia-smi
echo "-----------------------"


echo "--- Starting Training Script ---"

# Run the training script using uv.
# We are training and validating on the small validation set to test the pipeline.
uv run python train.py \
    --train_data_path data/tinystories_train.npy \
    --val_data_path data/tinystories_valid.npy

echo "--- Training Script Finished ---"
