#!/bin/bash
#SBATCH --job-name=yolo-train-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
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
# Le process principal (forward/backward PyTorch) doit pouvoir utiliser
# tous les CPUs alloués. Les workers du DataLoader se limitent déjà
# automatiquement à 1 thread chacun (comportement PyTorch par défaut),
# donc pas de risque de sur-souscription en augmentant OMP_NUM_THREADS.
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "=== Démarrage de l'entraînement sur CPU ==="
.venv/bin/python entrainement_masks.py

echo ""
echo "=== Entraînement terminé ==="
echo "Date : $(date)"