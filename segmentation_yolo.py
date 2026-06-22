# -*- coding: utf-8 -*-
"""
Inférence SAHI par lot — détection de cuvettes de dégazage
Applique le modèle YOLO sur toutes les images d'un dossier
"""

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path
from dotenv import load_dotenv
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
# CHARGEMENT DU MODÈLE
# ═══════════════════════════════════════════════════════════════
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=BEST_MODEL,
    confidence_threshold=0.3,
    device="cpu",
)

# ═══════════════════════════════════════════════════════════════
# INFÉRENCE SUR TOUTES LES IMAGES DU DOSSIER
# ═══════════════════════════════════════════════════════════════
images = list(DOSSIER_IMG.glob("*.JPG")) + list(DOSSIER_IMG.glob("*.jpg"))
print(f"Images trouvées : {len(images)}")

for i, img_path in enumerate(images):
    print(f"[{i+1}/{len(images)}] {img_path.name} ...", end=" ")
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
    print(f"{len(result.object_prediction_list)} détections")

print(f"\nTerminé ! Résultats sauvegardés dans : {RESULTATS}")
