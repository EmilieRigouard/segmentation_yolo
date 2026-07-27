# -*- coding: utf-8 -*-
"""
Inférence SAHI par lot — détection de cuvettes de dégazage
Applique le modèle YOLO sur toutes les images d'un dossier
"""

from multiprocessing import cpu_count, Pool
from socket import gethostname

import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path
from dotenv import load_dotenv
import psutil
import cv2
import os

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# CHEMINS
# ═══════════════════════════════════════════════════════════════
BEST_MODEL  = os.getenv("BEST_MODEL")
DOSSIER_IMG = Path(os.getenv("DOSSIER_IMG"))
RESULTATS   = Path(os.getenv("RESULTATS"))
RESULTATS.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# CPU
# ═══════════════════════════════════════════════════════════════

# Nb de CPU à utiliser
# Stella
if "ncpu" in gethostname():
    cpu_nb = len(psutil.Process().cpu_affinity())
# macseb
elif "mac" in gethostname():
    cpu_nb = cpu_count()
# Windows
else:
    cpu_nb = cpu_count() - 1

print(f"Using {cpu_nb=} CPU")

# Déterminer le device (GPU ou CPU)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPU(s) détecté(s) : {num_gpus}")
    device = "0"  # SAHI utilise le GPU 0 pour l'inférence

    # Adapter les paramètres d'inférence selon la GPU
    # (sur cluster : A100 probable)
    # Note : on peut pas lire le nom de la GPU de façon fiable sur cluster
    # donc on utilise une heuristique : plus de 40GB = A100-like
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory GPU disponible : {gpu_memory_gb:.1f} GB")
        if gpu_memory_gb > 35:  # A100 40GB
            slice_height = 2048
            slice_width = 2048
            overlap_ratio = 0.15
        else:
            slice_height = 1024
            slice_width = 1024
            overlap_ratio = 0.2
    except:
        # Fallback : utiliser les valeurs standard
        print(f"Could not read GPU memory, using default slice parameters")
        slice_height = 1024
        slice_width = 1024
        overlap_ratio = 0.2

    print(f"Slices optimisées : {slice_height}x{slice_width}, overlap={overlap_ratio}")

    # Réduire légèrement le parallélisme CPU quand on utilise GPU
    cpu_nb = max(1, cpu_nb // 2)
    print(f"Mode GPU : workers CPU réduits à {cpu_nb}")
else:
    print("Pas de GPU disponible, utilisation du CPU")
    device = "cpu"
    slice_height = 1024
    slice_width = 1024
    overlap_ratio = 0.2

# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ═══════════════════════════════════════════════════════════════
# Re-vérifier le device au moment du chargement du modèle
# (important en environnement MIG où le nombre de GPUs peut changer)
print(f"\n=== Vérification avant chargement du modèle ===")
if torch.cuda.is_available():
    num_gpus_now = torch.cuda.device_count()
    print(f"  torch.cuda.device_count(): {num_gpus_now}")
    if num_gpus_now == 1:
        device_for_model = "0"
    else:
        device_for_model = ",".join(str(i) for i in range(num_gpus_now))
    print(f"  Device pour inférence: {device_for_model}")
else:
    device_for_model = "cpu"
    print(f"  Pas de GPU, utilisation CPU")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=BEST_MODEL,
    confidence_threshold=0.3,
    device=device_for_model,  # Utiliser le device re-vérifié
)


# ═══════════════════════════════════════════════════════════════
# FONCTION DE TRAITEMENT D'UNE IMAGE
# ═══════════════════════════════════════════════════════════════
def process_image(img_path, index, total_images, slice_h, slice_w, overlap_r):
    """Traite une seule image et retourne le nombre de détections."""
    print(f"[{index+1}/{total_images}] {img_path.name} ...", end=" ")

    result = get_sliced_prediction(
        str(img_path),
        detection_model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap_r,
        overlap_width_ratio=overlap_r,
    )

    # Dessiner les cercles
    img = cv2.imread(str(img_path))
    for pred in result.object_prediction_list:
        x1 = pred.bbox.minx
        y1 = pred.bbox.miny
        x2 = pred.bbox.maxx
        y2 = pred.bbox.maxy
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        r  = int(max(x2 - x1, y2 - y1) / 2)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), thickness=6)

    output_path = RESULTATS / img_path.name
    cv2.imwrite(str(output_path), img)

    nb_detections = len(result.object_prediction_list)
    print(f"{nb_detections} détections")
    return nb_detections


# ═══════════════════════════════════════════════════════════════
# INFÉRENCE SUR TOUTES LES IMAGES DU DOSSIER EN PARALLÈLE
# ═══════════════════════════════════════════════════════════════
images = list(DOSSIER_IMG.glob("*.JPG")) + list(DOSSIER_IMG.glob("*.jpg"))
print(f"Images trouvées : {len(images)}")

# Préparer les arguments pour chaque image
args = [(img, i, len(images), slice_height, slice_width, overlap_ratio) for i, img in enumerate(images)]

# Traitement parallèle sur tous les CPU disponibles
with Pool(processes=cpu_nb) as pool:
    results = pool.starmap(process_image, args)

print(f"\nTerminé ! Résultats sauvegardés dans : {RESULTATS}")
print(f"Total détections : {sum(results)}")
