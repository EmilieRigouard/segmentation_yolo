#!/bin/bash
#SBATCH --job-name=yolo-inference-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ncpu,ncpum,ncpumshort,ncpulong
#SBATCH --output=./logs/infer-%j.txt
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL

echo "=== Job lancé sur $(hostname) ==="
echo "Date : $(date)"
echo "CPUs alloués : $SLURM_CPUS_PER_TASK"
echo "Mémoire allouée : $SLURM_MEM_PER_NODE"

# Configuration CPU
module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "=== Démarrage de l'inférence sur CPU ==="
.venv/bin/python segmentation_yolo.py

echo ""
echo "=== Inférence terminée ==="
echo "Date : $(date)"
