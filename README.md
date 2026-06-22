# Pipeline YOLO — Détection de cuvettes de dégazage

Pipeline de segmentation YOLO pour la détection de cuvettes sur images drone haute résolution (5472×3648).

## Étapes
`entrainement_masks.py` - script d'entrainement sur un set de masques annotés + images correspondantes

1. Conversion annotations COCO → YOLO segmentation
2. Split train/val (80/20)
3. Découpe en tuiles 1024×1024 (overlap 70%)
4. Ajout de backgrounds négatifs
5. Génération du `dataset.yaml`
6. Entraînement YOLOv8n-seg
7. Inférence avec SAHI (image unique)

`segmentation_yolo.py` — script indépendant pour appliquer le modèle sur un dossier entier d'images

## Installation

```bash
# Installer uv si pas déjà fait
pip install uv

# Créer l'environnement et installer les dépendances
uv sync
```

## Configuration

```bash
cp .env .env
# Éditer .env avec les chemins locaux
```

## Lancement

```bash
uv run python pipeline.py
```
