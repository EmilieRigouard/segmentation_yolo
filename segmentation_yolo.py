# -*- coding: utf-8 -*-
"""
Inférence SAHI par lot — détection de cuvettes de dégazage
Applique le modèle YOLO sur toutes les images d'un dossier
"""

from multiprocessing import cpu_count, Pool
from socket import socket

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
if "ncpu" in socket.gethostname():
    cpu_nb = len(psutil.Process().cpu_affinity())
# macseb
elif "mac" in socket.gethostname():
    cpu_nb = cpu_count()
# Windows
else:
    cpu_nb = cpu_count() - 1

print(f"Using {cpu_nb=} CPU")

# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ═══════════════════════════════════════════════════════════════
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=BEST_MODEL,
    confidence_threshold=0.3,
    device="cpu",
)


# ═══════════════════════════════════════════════════════════════
# FONCTION DE TRAITEMENT D'UNE IMAGE
# ═══════════════════════════════════════════════════════════════
def process_image(img_path, index, total_images):
    """Traite une seule image et retourne le nombre de détections."""
    print(f"[{index+1}/{total_images}] {img_path.name} ...", end=" ")

    result = get_sliced_prediction(
        str(img_path),
        detection_model,
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
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
args = [(img, i, len(images)) for i, img in enumerate(images)]

# Traitement parallèle sur tous les CPU disponibles
with Pool(processes=cpu_nb) as pool:
    results = pool.starmap(process_image, args)

print(f"\nTerminé ! Résultats sauvegardés dans : {RESULTATS}")
print(f"Total détections : {sum(results)}")
