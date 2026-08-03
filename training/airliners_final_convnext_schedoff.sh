#!/bin/bash
#SBATCH --job-name=airliner_final_convnext_schedoff         # Name of your job
#SBATCH --output=logs/%x_%j.out             # Standard output log (%x is job-name, %j is JobID)
#SBATCH --error=logs/%x_%j.err              # Standard error log
#SBATCH --time=03:00:00                     # Time limit (HH:MM:SS) - Be realistic!
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                          # Number of tasks (usually 1 for basic PyTorch)
#SBATCH --cpus-per-task=8                   # CPU cores (match your DataLoader num_workers + a bit of overhead)
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
export TORCH_HOME=~/links/scratch/Bruce-Qiu-APS360-Project/.torch_cache

# 3. Execute your Python script



echo "Starting ConvNeXt training job..."
python ./main.py \
    --mode train \
    --model primary_convnext \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.000075 \
    --weight_decay 0.05 \
    --dropout_rate 0.3 \
    --hidden_dim 512 \
    --checkpoint_freq 20 \
    --checkpoint_dir "./checkpoints/final/primary_convnext_final_schedOFF" \
    --image_size 300 \
    --num_workers 8 
    
echo "ConvNeXt Training complete!"