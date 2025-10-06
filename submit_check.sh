#!/bin/bash
#
#SBATCH --job-name=l40_gpu_test         # A name for your job
#SBATCH --account=ventz001              # The account/group name we found
#SBATCH --mail-type=ALL                 # Send email on job start, end, and failure
#SBATCH --mail-user=<your_email@umn.edu>   # Your email address
#
##### RESOURCE REQUESTS #####
#SBATCH -t 00:15:00                     # Time limit (HH:MM:SS). 15 minutes is fine for a test.
#SBATCH --mem=8gb                       # Memory request (8 GB).
#SBATCH --ntasks=1                      # Number of tasks (usually 1)
#SBATCH --cpus-per-task=2               # Number of CPU cores per task
#
##### GPU REQUESTS #####
#SBATCH --partition=interactive-gpu     # Partition for L40s
#SBATCH --gres=gpu:l40s:1               # Request one (1) L40s GPU
#
##### JOB OUTPUT #####
#SBATCH -o slurm-l40-%j.out             # Standard output file
#SBATCH -e slurm-l40-%j.err             # Standard error file

# --- Your commands go below this line ---

echo "Loading conda module..."
module load conda

echo "Activating conda environment..."
source activate pytorch-h100

echo "Running Python GPU check script..."
python check_gpu.py

echo "Script finished."
