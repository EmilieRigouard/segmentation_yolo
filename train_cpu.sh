#!/bin/bash
#SBATCH --job-name=yolo-train-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=ncpu,ncpum,ncpumshort,ncpulong
#SBATCH --output=./logs/train-%j.txt
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL


cd /gpfs/scratch/bonaime/git/segmentation_yolo
hostname
module purge

echo "=== Job lancé sur $(hostname) ==="
echo "Date : $(date)"
echo "CPUs alloués : $SLURM_CPUS_PER_TASK"
echo "Mémoire allouée : $SLURM_MEM_PER_NODE"

# Configuration CPU
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=2

exit
echo ""
echo "=== Démarrage de l'entraînement sur CPU ==="
.venv/bin/python entrainement_masks.py

echo ""
echo "=== Entraînement terminé ==="
echo "Date : $(date)"