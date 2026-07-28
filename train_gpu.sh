#!/bin/bash
#SBATCH --job-name=yolo-train-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=ngpuashort,ngpualong
#SBATCH --output=./logs/train_GPU-%j.txt
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL

hostname
module purge
module load cuda/12.6.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Réduit les OOM dus à la fragmentation du cache CUDA (utile sur GPU/MIG à mémoire limitée)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job lancé sur $(hostname) ==="
echo "Date : $(date)"
echo "CPUs alloués : $SLURM_CPUS_PER_TASK"
echo "GPUs alloués : $SLURM_GPUS_ON_NODE"
echo "Mémoire allouée : $SLURM_MEM_PER_NODE"

echo ""
echo "=== Démarrage de l'entraînement sur GPU ==="
.venv/bin/python entrainement_masks.py

echo ""
echo "=== Entraînement terminé ==="
echo "Date : $(date)"
