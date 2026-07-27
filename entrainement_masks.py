# -*- coding: utf-8 -*-
"""
Pipeline YOLO segmentation pour détection de cuvettes de dégazage
Images drone haute résolution (5472x3648) → découpe en tuiles → entraînement
"""

from socket import gethostname

import cv2
import shutil
import random
import numpy as np
import sys
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count
import psutil
import torch
import os

load_dotenv()


def format_elapsed_time(seconds):
    """Formate le temps écoulé en h:mm:ss"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def glob_images(directory):
    """Retourne les images d'un dossier, en gérant .jpg/.JPG/.jpeg/.JPEG."""
    directory = Path(directory)
    exts = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    found = []
    for ext in exts:
        found.extend(directory.glob(ext))

    if not found:
        print(f"⚠️ AVERTISSEMENT: Aucune image trouvée dans {directory}")
        sys.exit(1)

    return found


def decoupe_tuile(img_path, labels_dir, output_dir, tile_size=1024, overlap=0.7):
    """Traite une seule image pour la découpe en tuiles.

    Fonction globale pour être picklable par multiprocessing.
    Retourne le nombre de tuiles créées.
    """
    img_path = Path(img_path)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    label_path = labels_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        return 0

    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    h, w = img.shape[:2]

    with open(label_path) as f:
        annotations = f.readlines()

    stride = int(tile_size * (1 - overlap))
    total_tuiles = 0

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            tile_labels = []

            for ann in annotations:
                parts = ann.strip().split()
                class_id = parts[0]
                coords = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                coords_px = coords * np.array([w, h])

                in_tile = (
                    (coords_px[:, 0] >= x) & (coords_px[:, 0] < x + tile_size) &
                    (coords_px[:, 1] >= y) & (coords_px[:, 1] < y + tile_size)
                )
                if in_tile.sum() / len(in_tile) < 0.5:
                    continue

                coords_tile = coords_px - np.array([x, y])
                coords_tile = np.clip(coords_tile, 0, tile_size)
                coords_norm = coords_tile / tile_size
                flat = coords_norm.flatten()
                tile_labels.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in flat))

            if tile_labels:
                name = f"{img_path.stem}_x{x}_y{y}"
                cv2.imwrite(str(output_dir / "images" / f"{name}.jpg"), tile)
                with open(output_dir / "labels" / f"{name}.txt", "w") as f:
                    f.write("\n".join(tile_labels))
                total_tuiles += 1

    return total_tuiles


class PipelineYOLO:

    # ═══════════════════════════════════════════════════════════════
    # CHEMINS
    # ═══════════════════════════════════════════════════════════════
    def __init__(self):
        self.IMAGES_SRC  = Path(os.getenv("IMAGES_SRC"))
        self.JSON_DIR    = Path(os.getenv("JSON_DIR"))
        self.DATASET_DIR = Path(os.getenv("DATASET_DIR"))
        self.SPLIT_DIR   = Path(os.getenv("SPLIT_DIR"))
        self.TUILES_DIR  = Path(os.getenv("TUILES_DIR"))
        self.YAML_PATH   = Path(os.getenv("YAML_PATH"))

        # Vérifier que les chemins source existent
        if not self.IMAGES_SRC.exists():
            print(f"❌ ERREUR: IMAGES_SRC n'existe pas: {self.IMAGES_SRC}")
            sys.exit(1)
        if not self.JSON_DIR.exists():
            print(f"❌ ERREUR: JSON_DIR n'existe pas: {self.JSON_DIR}")
            sys.exit(1)

        # Créer les répertoires de sortie s'ils n'existent pas
        self.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        self.TUILES_DIR.mkdir(parents=True, exist_ok=True)
        self.YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

        print(f"✓ IMAGES_SRC: {self.IMAGES_SRC}")
        print(f"✓ JSON_DIR: {self.JSON_DIR}")
        print(f"✓ DATASET_DIR: {self.DATASET_DIR}")
        print(f"✓ SPLIT_DIR: {self.SPLIT_DIR}")
        print(f"✓ TUILES_DIR: {self.TUILES_DIR}")
        print(f"✓ YAML_PATH: {self.YAML_PATH}")


        # Nb de CPU à utiliser
        # Stella
        if "ncpu" in gethostname():
            self.cpu_nb = len(psutil.Process().cpu_affinity())
        # macseb
        elif "mac" in gethostname():
            self.cpu_nb = cpu_count()
        # Windows
        else:
            self.cpu_nb = cpu_count() - 1

        print(f"Using {self.cpu_nb=} CPU")

        # Déterminer le device GPU
        self.device = self._detect_device()
        print(f"Device pour entraînement : {self.device}")

        # Enregistrer le temps de démarrage
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Démarrage du pipeline à {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

    def _detect_device(self):
        """Détecte et retourne le device (GPU multi ou CPU)"""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
            print(f"  → {num_gpus} GPU(s) détecté(s) : {', '.join(gpu_names)}")
            # Retourner tous les GPUs : "0,1,2,3" ou juste "0" si un seul
            if num_gpus == 1:
                return "0"
            else:
                return ",".join(str(i) for i in range(num_gpus))
        else:
            print(f"  → Pas de GPU, utilisation du CPU")
            return "cpu"

    def log_step(self, step_name):
        """Affiche l'heure et le temps écoulé pour une étape"""
        elapsed = time.time() - self.start_time
        elapsed_str = format_elapsed_time(elapsed)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{current_time}] [{elapsed_str}] {step_name}")

    # ═══════════════════════════════════════════════════════════════
        print(f"Démarrage du pipeline à {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")


    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 1 — Convertir JSON COCO → labels YOLO segmentation
    # ═══════════════════════════════════════════════════════════════
    def convert_coco(self):
        self.log_step("ÉTAPE 1 — Convertir JSON COCO → labels YOLO segmentation")
        if self.DATASET_DIR.exists():
            shutil.rmtree(self.DATASET_DIR)

        convert_coco(
            labels_dir=str(self.JSON_DIR.parent),
            save_dir=str(self.DATASET_DIR),
            use_segments=True,
            use_keypoints=False,
        )

    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 2 — Split train/val (80/20) → dans SPLIT_DIR
    # ═══════════════════════════════════════════════════════════════
    def split_dataset(self):
        self.log_step("ÉTAPE 2 — Split train/val (80/20)")
        LABELS_SRC = self.DATASET_DIR / "labels" / "Train"

        for split in ["train", "val"]:
            (self.SPLIT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.SPLIT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

        all_images = [f for f in glob_images(self.IMAGES_SRC)
                      if (LABELS_SRC / (f.stem + ".txt")).exists()]

        random.seed(42)
        random.shuffle(all_images)
        split_idx  = int(len(all_images) * 0.8)
        train_imgs = all_images[:split_idx]
        val_imgs   = all_images[split_idx:]

        print(f"Train : {len(train_imgs)} images | Val : {len(val_imgs)} images")

        for img_list, split in [(train_imgs, "train"), (val_imgs, "val")]:
            for img_path in img_list:
                shutil.copy(img_path, self.SPLIT_DIR / "images" / split / img_path.name)
                label_path = LABELS_SRC / (img_path.stem + ".txt")
                if label_path.exists():
                    shutil.copy(label_path, self.SPLIT_DIR / "labels" / split / label_path.name)

        print("Split terminé !")

    # ═══════════════════════════════════════════════════════════════
    # STATS DIAMÈTRES
    # ═══════════════════════════════════════════════════════════════
    def print_diameter_stats(self):
        self.log_step("STATS — Calcul des diamètres")
        tailles = []
        for label_path in (self.SPLIT_DIR / "labels" / "train").glob("*.txt"):
            img_path = None
            for candidate in glob_images(self.IMAGES_SRC):
                if candidate.stem == label_path.stem:
                    img_path = candidate
                    break
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(label_path) as f:
                for line in f:
                    vals = list(map(float, line.strip().split()[1:]))
                    coords = np.array(vals).reshape(-1, 2)
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    (cx, cy), r = cv2.minEnclosingCircle(coords.astype(np.int32))
                    tailles.append(r * 2)

        if tailles:
            print(f"Diamètre moyen  : {np.mean(tailles):.1f} px")
            print(f"Diamètre min    : {np.min(tailles):.1f} px")
            print(f"Diamètre max    : {np.max(tailles):.1f} px")
        else:
            print("⚠️ Aucun label trouvé pour les stats diamètres")

    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 3 — Découpe en tuiles
    # ═══════════════════════════════════════════════════════════════
    def decoupe_tuiles(self, images_dir, labels_dir, output_dir, tile_size=1024, overlap=0.7):
        self.log_step(f"ÉTAPE 3 — Découpe en tuiles ({Path(output_dir).name})")
        output_dir = Path(output_dir)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

        images = glob_images(images_dir)
        labels_dir = Path(labels_dir)

        # Préparer les arguments pour chaque image
        args = [
            (img, labels_dir, output_dir, tile_size, overlap)
            for img in images
        ]

        # Traitement parallèle sur tous les CPU disponibles
        with Pool(processes=self.cpu_nb) as pool:
            results = pool.starmap(decoupe_tuile, args)

        total_tuiles = sum(results)
        print(f"Tuiles avec annotations ({output_dir.name}) : {total_tuiles}")


    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 4 — Ajouter des backgrounds
    # ═══════════════════════════════════════════════════════════════
    def ajouter_backgrounds(self, images_src, images_annotees, output_dir, n_backgrounds=50):
        self.log_step(f"ÉTAPE 4 — Ajouter des backgrounds ({Path(output_dir).name})")
        output_dir = Path(output_dir)
        tile_size = 1024

        annotees = {f.stem for f in glob_images(images_annotees)}
        sans_cuvettes = [f for f in glob_images(images_src) if f.stem not in annotees]

        random.seed(42)
        selection = random.sample(sans_cuvettes, min(n_backgrounds, len(sans_cuvettes)))

        for img_path in selection:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            x = random.randint(0, max(0, w - tile_size))
            y = random.randint(0, max(0, h - tile_size))
            tile = img[y:y+tile_size, x:x+tile_size]
            name = f"bg_{img_path.stem}_x{x}_y{y}.jpg"
            cv2.imwrite(str(output_dir / "images" / name), tile)

        print(f"Backgrounds ajoutés ({Path(output_dir).name}) : {len(selection)}")

    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 5 — Créer le dataset.yaml
    # ═══════════════════════════════════════════════════════════════
    def create_yaml(self):
        self.log_step("ÉTAPE 5 — Créer le dataset.yaml")
        yaml_content = f"""path: {self.TUILES_DIR.as_posix()}
train: train/images
val: val/images

nc: 1
names: ["cuvette"]
"""
        with open(self.YAML_PATH, "w") as f:
            f.write(yaml_content)

        print(f"\ndataset_tuiles.yaml créé : {self.YAML_PATH}")

    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 6 — Entraînement YOLO
    # ═══════════════════════════════════════════════════════════════
    def train(self):
        self.log_step("ÉTAPE 6 — Entraînement YOLO")
        model = YOLO("yolov8n-seg.pt")

        # Déterminer le batch size optimal selon le GPU disponible
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)

            # Adapter le batch size à la GPU
            if "A100" in gpu_name:
                batch_size = 64  # A100 40GB → très puissant !
            elif "A6000" in gpu_name or "RTX 6000" in gpu_name:
                batch_size = 48
            elif "V100" in gpu_name:
                batch_size = 32
            elif "A10" in gpu_name or "RTX 4090" in gpu_name:
                batch_size = 32
            elif "RTX 4080" in gpu_name:
                batch_size = 16
            elif "RTX 4070" in gpu_name or "RTX 3090" in gpu_name:
                batch_size = 12
            else:
                batch_size = 8  # Par défaut : conservateur

            print(f"  → Batch size : {batch_size} (optimisé pour {gpu_name})")
        else:
            batch_size = 4
            print(f"  → Batch size : {batch_size} (CPU)")

        model.train(
            data=str(self.YAML_PATH),
            epochs=100,
            imgsz=1024,
            batch=batch_size,
            device=self.device,  # GPU détecté automatiquement
            workers=self.cpu_nb,
            augment=True,
            patience=30,
            degrees=180.0,
            flipud=0.5,
        )

    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 7 — Inférence avec SAHI
    # ═══════════════════════════════════════════════════════════════
    def run_inference(self):
        self.log_step("ÉTAPE 7 — Inférence avec SAHI")
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        from PIL import Image

        BEST_MODEL = str(sorted(
            Path(os.getenv("TRAIN_DIR")).glob("train*/weights/best.pt")
        )[-1])
        print(f"\nModèle utilisé : {BEST_MODEL}")

        TEST_IMAGE = str(glob_images(self.IMAGES_SRC)[0])

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=BEST_MODEL,
            confidence_threshold=0.3,
            device="cpu",
        )

        result = get_sliced_prediction(
            TEST_IMAGE,
            detection_model,
            slice_height=1024,
            slice_width=1024,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        img = cv2.imread(TEST_IMAGE)
        for pred in result.object_prediction_list:
            x1 = pred.bbox.minx
            y1 = pred.bbox.miny
            x2 = pred.bbox.maxx
            y2 = pred.bbox.maxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            r  = int(max(x2 - x1, y2 - y1) / 2)
            cv2.circle(img, (cx, cy), r, (0, 255, 0), thickness=6)

        Path("./resultats").mkdir(exist_ok=True)
        output_path = "./resultats/resultat.jpg"
        cv2.imwrite(output_path, img)
        print(f"Détections : {len(result.object_prediction_list)}")

        Image.open(output_path).show()

    # ═══════════════════════════════════════════════════════════════
    # RUN
    # ═══════════════════════════════════════════════════════════════
    def run(self):
        # Étape 1
        self.convert_coco()

        # Étape 2
        self.split_dataset()
        self.print_diameter_stats()

        # Étape 3
        self.decoupe_tuiles(
            images_dir=self.SPLIT_DIR / "images" / "train",
            labels_dir=self.SPLIT_DIR / "labels" / "train",
            output_dir=self.TUILES_DIR / "train",
        )
        self.decoupe_tuiles(
            images_dir=self.SPLIT_DIR / "images" / "val",
            labels_dir=self.SPLIT_DIR / "labels" / "val",
            output_dir=self.TUILES_DIR / "val",
        )

        # Étape 4
        self.ajouter_backgrounds(
            images_src=self.IMAGES_SRC,
            images_annotees=self.SPLIT_DIR / "images" / "train",
            output_dir=self.TUILES_DIR / "train",
            n_backgrounds=50,
        )
        self.ajouter_backgrounds(
            images_src=self.IMAGES_SRC,
            images_annotees=self.SPLIT_DIR / "images" / "val",
            output_dir=self.TUILES_DIR / "val",
            n_backgrounds=10,
        )

        print(f"\nDataset final :")
        print(f"  Train : {len(glob_images(self.TUILES_DIR / 'train' / 'images'))} images")
        print(f"  Val   : {len(glob_images(self.TUILES_DIR / 'val'   / 'images'))} images")

        # Étape 5
        self.create_yaml()

        # Étape 6
        self.train()

        # Étape 7
        self.run_inference()

        # Rapport final
        total_time = time.time() - self.start_time
        total_time_str = format_elapsed_time(total_time)
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f"\n{'='*60}")
        print(f"Pipeline terminé avec succès !")
        print(f"Date/heure de fin : {end_time}")
        print(f"Durée totale : {total_time_str}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    PipelineYOLO().run()