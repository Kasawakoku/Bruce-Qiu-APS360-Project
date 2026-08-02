#!/bin/bash
#SBATCH --job-name=airliner_primary         # Name of your job
#SBATCH --output=logs/%x_%j.out             # Standard output log (%x is job-name, %j is JobID)
#SBATCH --error=logs/%x_%j.err              # Standard error log
#SBATCH --time=06:00:00                     # Time limit (HH:MM:SS) - Be realistic!
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                          # Number of tasks (usually 1 for basic PyTorch)
#SBATCH --cpus-per-task=4                   # CPU cores (match your DataLoader num_workers + a bit of overhead)
#SBATCH --gpus-per-node=1                            # Request 1 GPU (Trillium uses NVIDIA H100s)
#SBATCH --partition=compute                     # (Optional/Depends on cluster) Specify the GPU queue

# MUST RUN WITH TRAINING WORKING DIRECTORY AS CURRENT DIRECTORY

# 1. Load the necessary modules (SciNet specific)
module purge
module load StdEnv/2023
module load python/3.11.5

module load cuda   # Load CUDA if required by SciNet's PyTorch setup

# 2. Activate your environment
source .venv/bin/activate

# 3. Execute your Python script
echo "Starting Run 1 training job..."
python ./main.py \
    --mode train \
    --model primary \
    --epochs 40 \
    --batch_size 16 \
    --lr 0.001 \
    --checkpoint_freq 10 \
    --checkpoint_dir "./checkpoints/checkpoints_primary_bs16_lr0.001" \
    
echo "Run 1 Training complete!"

echo "Starting Run 2 training job..."
python ./main.py \
    --mode train \
    --model primary \
    --epochs 40 \
    --batch_size 32 \
    --lr 0.001 \
    --checkpoint_freq 10 \
    --checkpoint_dir "./checkpoints/checkpoints_primary_bs32_lr0.001" \
    
    
echo "Run 2 Training complete!"

echo "Starting Run 3 training job..."
python ./main.py \
    --mode train \
    --model primary \
    --epochs 40 \
    --batch_size 64 \
    --lr 0.001 \
    --checkpoint_freq 10 \
    --checkpoint_dir "./checkpoints/checkpoints_primary_bs64_lr0.001" \
    
echo "Run 3 Training complete!"

