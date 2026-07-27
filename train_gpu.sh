#!/bin/bash
#SBATCH --job-name=yolo-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ngpualong,ngpuashort
#SBATCH --output=/gpfs/users/bonaime/logs/train-%j.txt
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks-per-node=8


cd /gpfs/scratch/bonaime/git/segmentation_yolo
hostname
module purge

echo "=== Job lancé sur $(hostname) ==="
echo "Date : $(date)"


# Charger CUDA
module load cuda/12.6.2

# Unset CUDA_VISIBLE_DEVICES pour laisser PyTorch voir seulement ce qui est réellement accessible
unset CUDA_VISIBLE_DEVICES

# Ajouter les librairies CUDA à LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Vérifier CUDA
echo "=== Vérification CUDA ==="
nvidia-smi --query-gpu=name --format=csv

# IMPORTANT : Réinstaller PyTorch maintenant que CUDA est chargé
echo "=== Réinstallation PyTorch avec CUDA ==="
uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126 -q

# Tester
#echo "=== Test PyTorch ==="
#.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"


.venv/bin/python  entrainement_masks.py