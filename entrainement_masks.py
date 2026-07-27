# -*- coding: utf-8 -*-
"""
Pipeline YOLO segmentation pour détection de cuvettes de dégazage
Images drone haute résolution (5472x3648) → découpe en tuiles → entraînement
"""

import cv2
import shutil
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
from dotenv import load_dotenv
import os

load_dotenv()


def glob_images(directory):
    """Retourne les images d'un dossier, en gérant .jpg/.JPG/.jpeg/.JPEG."""
    directory = Path(directory)
    exts = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    found = []
    for ext in exts:
        found.extend(directory.glob(ext))
    return found


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

    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 1 — Convertir JSON COCO → labels YOLO segmentation
    # ═══════════════════════════════════════════════════════════════
    def convert_coco(self):
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
        output_dir = Path(output_dir)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

        stride = int(tile_size * (1 - overlap))
        total_tuiles = 0

        for img_path in glob_images(images_dir):
            label_path = Path(labels_dir) / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            with open(label_path) as f:
                annotations = f.readlines()

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
                        tile_labels.append(class_id + " " + " ".join(f"{v:.6f}" for v in flat))

                    if tile_labels:
                        name = f"{img_path.stem}_x{x}_y{y}"
                        cv2.imwrite(str(output_dir / "images" / f"{name}.jpg"), tile)
                        with open(output_dir / "labels" / f"{name}.txt", "w") as f:
                            f.write("\n".join(tile_labels))
                        total_tuiles += 1

        print(f"Tuiles avec annotations ({Path(output_dir).name}) : {total_tuiles}")

    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 4 — Ajouter des backgrounds
    # ═══════════════════════════════════════════════════════════════
    def ajouter_backgrounds(self, images_src, images_annotees, output_dir, n_backgrounds=50):
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
        model = YOLO("yolov8n-seg.pt")
        model.train(
            data=str(self.YAML_PATH),
            epochs=100,
            imgsz=1024,
            batch=4,
            device="cpu",
            workers=2,
            augment=True,
            patience=30,
            degrees=180.0,
            flipud=0.5,
        )

    # ═══════════════════════════════════════════════════════════════
    # ÉTAPE 7 — Inférence avec SAHI
    # ═══════════════════════════════════════════════════════════════
    def run_inference(self):
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


if __name__ == "__main__":
    PipelineYOLO().run()