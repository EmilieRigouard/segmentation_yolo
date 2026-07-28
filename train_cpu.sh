#!/bin/bash
#SBATCH --job-name=yolo-train-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=ncpu,ncpum,ncpulong
#SBATCH --output=./logs/train-%j.txt
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL


hostname
module purge

echo "=== Job lancé sur $(hostname) ==="
echo "Date : $(date)"
echo "CPUs alloués : $SLURM_CPUS_PER_TASK"
echo "Mémoire allouée : $SLURM_MEM_PER_NODE"

# Configuration CPU
# yolov8n-seg est un petit modèle : au-delà de quelques threads, le calcul
# CPU (intra-op OpenMP/MKL) est dominé par l'overhead de synchronisation,
# pas par le calcul lui-même. Plus de threads = plus lent (mesuré : 2
# threads ~1h48/epoch, 32 threads ~2h05/epoch, 128 threads ~3h55/epoch).
# On garde donc un nombre de threads faible.
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=2

echo ""
echo "=== Démarrage de l'entraînement sur CPU ==="
.venv/bin/python entrainement_masks.py

echo ""
echo "=== Entraînement terminé ==="
echo "Date : $(date)"