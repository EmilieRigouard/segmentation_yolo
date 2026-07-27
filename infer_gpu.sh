#!/bin/bash
#SBATCH --job-name=yolo-inference
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ngpuashort,ngpualong
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/users/bonaime/logs/inference-%j.txt
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL

echo "=== Job lancé sur $(hostname) ==="
echo "Date : $(date)"

# Charger CUDA
module purge
module load cuda/12.6.2

# Unset CUDA_VISIBLE_DEVICES pour laisser PyTorch voir seulement ce qui est réellement accessible
unset CUDA_VISIBLE_DEVICES

# Ajouter les librairies CUDA à LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Vérifier CUDA
echo "=== Vérification GPU ==="
nvidia-smi --query-gpu=name --format=csv

# Vérifier PyTorch
echo "=== Vérification PyTorch ==="
.venv/bin/python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPUs: {torch.cuda.device_count()}')"

# Lancer l'inférence
echo ""
echo "=== Lancement de l'inférence SAHI ==="
.venv/bin/python segmentation_yolo.py

echo ""
echo "=== Inférence terminée ==="
echo "Date : $(date)"
