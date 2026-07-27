# Pipeline YOLO Segmentation - Guide SCAPA GPU

## Installation

### 1. Créer l'environnement virtuel

```bash
uv venv --python 3.11
uv sync
```

### 2. Charger CUDA et installer PyTorch

D'abord, sur un **nœud GPU** :

```bash
module load cuda/12.6.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv sync
```

### 3. Vérifier que CUDA marche

```bash
module load cuda/12.6.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

## Utilisation

### Entraînement sur GPU (4 × A100 40GB)

```bash
sbatch train_gpu.sh
```

Monitorer :
```bash
tail -f /gpfs/users/bonaime/logs/train-*.txt
```

### Inférence sur GPU

```bash
sbatch infer_gpu.sh
```

## Configuration

### Variables d'environnement `.env`

```ini
IMAGES_SRC=/chemin/vers/images
JSON_DIR=/chemin/vers/annotations
DATASET_DIR=./training/estran/dataset
SPLIT_DIR=./training/estran/split
TUILES_DIR=./training/estran/tuiles
YAML_PATH=./training/estran/dataset_tuiles.yaml
TRAIN_DIR=./runs/detect
DOSSIER_IMG=/chemin/vers/test/images
RESULTATS=./resultats
BEST_MODEL=./runs/detect/train/weights/best.pt
```

### Performance

- **Entraînement** : batch=64 sur A100 40GB → ~15h pour 100 epochs
- **Inférence** : 2048×2048 slices → 50-100% plus rapide qu'1024×1024

## Troubleshooting

### CUDA not available dans l'env local (macOS/Windows)

C'est normal ! PyTorch CPU-only. Sur SCAPA avec GPU, ça marche.

### ImportError: libcudnn.so.9

Solution : Toujours charger les modules **AVANT** Python

```bash
module load cuda/12.6.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
.venv/bin/python entrainement_masks.py
```

### PyTorch voit 4 GPUs mais CUDA: False

Besoin de `--gres=gpu:4` dans le script SLURM pour réserver les GPUs !

## Architecture

```
entrainement_masks.py    → 7 étapes du pipeline (CPU + GPU)
segmentation_yolo.py     → Inférence batch SAHI (GPU)
pyproject.toml          → Dépendances (torch, ultralytics, sahi, etc.)
train_gpu.sh            → SLURM job entraînement
infer_gpu.sh            → SLURM job inférence
```

## Détails techniques

### Batch Size automatique

Le script détecte la GPU et adapte le batch size :
- **A100** → batch=64
- **V100** → batch=32
- **RTX 4080** → batch=16

### Slice Size inférence

Adapté à la GPU pour maximiser throughput :
- **A100** → 2048×2048 slices
- Autres → 1024×1024 slices
