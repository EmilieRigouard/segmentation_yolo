#!/bin/bash
#SBATCH --job-name=yolo-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=ngpuashort,ngpualong
#SBATCH --output=/gpfs/users/bonaime/logs/train-%j.txt
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks-per-node=32

echo "=== Job lancé sur $(hostname) ==="
echo "Date : $(date)"

# Charger CUDA
module purge
module load cuda/12.6.2

# Ajouter les librairies CUDA à LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Vérifier CUDA
echo "=== Vérification GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

# Réinstaller PyTorch avec CUDA si nécessaire
echo "=== Vérification PyTorch CUDA ==="
.venv/bin/pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126 -q
.venv/bin/python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPUs: {torch.cuda.device_count()}')"

# Lancer l'entraînement
echo ""
echo "=== Lancement du pipeline YOLO ==="
.venv/bin/python entrainement_masks.py

echo ""
echo "=== Entraînement terminé ==="
echo "Date : $(date)"
