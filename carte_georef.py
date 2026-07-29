# -*- coding: utf-8 -*-
"""
Pipeline complet — Détection de cuvettes de dégazage
======================================================

Étape 1 : Détection SAHI (YOLO) sur les images drone -> masques PNG (ronds rouges)
Étape 2 : Géoréférencement des masques via EXIF + DEM               -> GeoTIFF
Étape 3 : Vectorisation des ronds rouges géoréférencés              -> GeoPackage (.gpkg)

Tout est organisé sous un seul dossier RESULTATS (variable d'env), avec des
sous-dossiers pour chaque étape intermédiaire.

Dépendances :
    pip install sahi ultralytics torch opencv-python python-dotenv psutil \
                rasterio affine pyproj geopandas shapely

Variables d'environnement attendues (.env) :
    BEST_MODEL     chemin vers le modèle YOLO entraîné (.pt)
    DOSSIER_IMG    dossier contenant les images drone .JPG/.jpg brutes
    RESULTATS      dossier racine de sortie (créé si besoin)
    DEM_PATH       chemin vers le MNT/DEM (GeoTIFF) pour le géoréférencement
    EXIFTOOL_PATH  chemin vers l'exécutable exiftool
"""

import json
import math
import os
import re
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from socket import gethostname

import cv2
import geopandas as gpd
import numpy as np
import psutil
import rasterio
import rasterio.windows
from affine import Affine
from dotenv import load_dotenv
from pyproj import Transformer
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_gcps
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Point

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION GÉNÉRALE
# ═══════════════════════════════════════════════════════════════
BEST_MODEL    = os.getenv("BEST_MODEL")
DOSSIER_IMG   = Path(os.getenv("DOSSIER_IMG"))
RESULTATS     = Path(os.getenv("RESULTATS"))
DEM_PATH      = os.getenv("DEM_PATH")
EXIFTOOL_PATH = os.getenv("EXIFTOOL_PATH", os.getenv("Exiftool_path"))

# Sous-dossiers dérivés de RESULTATS (tout reste au même endroit)
DOSSIER_PREVIEWS = RESULTATS / "previews"   # photos + cercles, pour contrôle visuel
DOSSIER_MASKS    = RESULTATS / "masks"      # masques PNG (ronds rouges seuls, fond transparent)
DOSSIER_GEOREF   = RESULTATS / "georef"     # masques géoréférencés (.tif)
OUTPUT_GPKG      = RESULTATS / "cuvettes.gpkg"
LAYER_NAME       = "cuvettes"

for d in (RESULTATS, DOSSIER_PREVIEWS, DOSSIER_MASKS, DOSSIER_GEOREF):
    d.mkdir(parents=True, exist_ok=True)

# Paramètres caméra (à adapter à votre drone)
EPSG_CODE        = 32738
SENSOR_WIDTH_MM  = 13.2
SENSOR_HEIGHT_MM = 8.8
FOCAL_LENGTH_MM  = 8.8
SUBSAMPLE_GCP    = 200   # densité des GCPs (200 = rapide, 50 = précis)

# Vectorisation finale
GEOMETRIE  = "polygon"   # "point" ou "polygon"
MIN_PIXELS = 10

# Seuil de rouge en RGB — CORRIGÉ : R haut, G et B bas (rouge pur = 255,0,0)
# L'ancien seuil (200-255 sur les 3 canaux) détectait du blanc, pas du rouge.
ROUGE_MIN = np.array([180, 0, 0])
ROUGE_MAX = np.array([255, 80, 80])


# ═══════════════════════════════════════════════════════════════
# CPU (pour l'étape de détection, parallélisable)
# ═══════════════════════════════════════════════════════════════
def get_cpu_count():
    if "ncpu" in gethostname():
        return len(psutil.Process().cpu_affinity())
    elif "mac" in gethostname():
        return cpu_count()
    else:
        return max(1, cpu_count() - 1)


CPU_NB = get_cpu_count()
SLICE_HEIGHT, SLICE_WIDTH, OVERLAP_RATIO = 512, 512, 0.2


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 1 — DÉTECTION SAHI -> MASQUES PNG
# ═══════════════════════════════════════════════════════════════
detection_model = None  # chargé dans etape1_detection() pour ne pas polluer un import


def _detect_and_mask(img_path, index, total):
    print(f"  [{index + 1}/{total}] {img_path.name} ...", end=" ")

    result = get_sliced_prediction(
        str(img_path),
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
    )

    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]

    # Aperçu annoté : photo originale + cercles (contrôle visuel humain)
    preview = img_bgr.copy()
    # Masque pur : fond noir transparent + ronds rouges pleins (BGRA)
    mask = np.zeros((h, w, 4), dtype=np.uint8)

    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        r = int(max(x2 - x1, y2 - y1) / 2)
        cv2.circle(preview, (cx, cy), r, (0, 0, 255), thickness=6)     # BGR rouge, contour
        cv2.circle(mask, (cx, cy), r, (0, 0, 255, 255), thickness=-1)  # BGRA rouge, plein

    cv2.imwrite(str(DOSSIER_PREVIEWS / img_path.name), preview)
    cv2.imwrite(str(DOSSIER_MASKS / f"{img_path.stem}_contour.png"), mask)

    nb = len(result.object_prediction_list)
    print(f"{nb} détection(s)")
    return nb


def etape1_detection():
    global detection_model

    print(f"\n{'=' * 60}\n  ÉTAPE 1 — Détection SAHI\n{'=' * 60}")
    print(f"CPU utilisés : {CPU_NB}  |  Slices : {SLICE_HEIGHT}x{SLICE_WIDTH}, overlap={OVERLAP_RATIO}")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=BEST_MODEL,
        confidence_threshold=0.3,
        device="cpu",
    )

    images = sorted(list(DOSSIER_IMG.glob("*.JPG")) + list(DOSSIER_IMG.glob("*.jpg")))
    print(f"Images trouvées : {len(images)}")
    if not images:
        print("[ERREUR] Aucune image trouvée dans DOSSIER_IMG.")
        sys.exit(1)

    args = [(img, i, len(images)) for i, img in enumerate(images)]
    with Pool(processes=CPU_NB) as pool:
        results = pool.starmap(_detect_and_mask, args)

    print(f"Total détections : {sum(results)}")
    print(f"Masques -> {DOSSIER_MASKS}")
    print(f"Aperçus -> {DOSSIER_PREVIEWS}")


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 2 — GÉORÉFÉRENCEMENT DES MASQUES (EXIF + DEM)
# ═══════════════════════════════════════════════════════════════
class MaskGeoreferencer:
    """
    Géoréférence un masque PNG en utilisant les métadonnées
    d'une image drone correspondante (même nom, sans '_contour').
    """

    def __init__(self, mask_path, source_image_path, exiftool_path,
                 DEM_path=None, epsg_code=32738,
                 sensor_width_mm=13.2, sensor_height_mm=8.8,
                 focal_length_mm=8.8):

        self.mask_path = Path(mask_path)
        self.source_image_path = Path(source_image_path)
        self.exiftool_path = exiftool_path
        self.DEM_path = DEM_path
        self.epsg_code = epsg_code
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.focal_length_mm = focal_length_mm

        self.lat = self.lon = None
        self.altitude_absolute = self.altitude_relative = None
        self.yaw_gimbal = self.pitch_gimbal = self.roll_gimbal = 0.0
        self.yaw_drone = self.pitch_drone = self.roll_drone = 0.0
        self.dewarpflag = None
        self.k1 = self.k2 = self.k3 = 0.0
        self.p1 = self.p2 = 0.0
        self.fx_calib = self.fy_calib = None
        self.cx_calib = self.cy_calib = None

        self.lever_x = self.lever_y = self.lever_z = 0.0

        self.mask_rgb = None
        self.height_mask = None
        self.width_mask = None
        self.bands = None
        self.mask_undistorted = None
        self.height_mask_undistorted = None
        self.width_mask_undistorted = None
        self.new_K = None
        self.K_inv = None

        self.ground_elevation = None
        self.height_above_ground = None

    # ── Métadonnées ──────────────────────────────────────────────
    def extract_metadata(self):
        print(f"\n--- Métadonnées depuis : {self.source_image_path.name} ---")

        raw = subprocess.check_output([
            str(self.exiftool_path), "-json", "-G", "-n",
            "-EXIF:all", "-XMP:all", "-MakerNotes:all", "-Composite:all",
            str(self.source_image_path)
        ])
        meta = json.loads(raw)[0]

        def get(*keys):
            for k in keys:
                if k in meta:
                    return meta[k]
            return None

        def getf(*keys, default=0.0):
            for k in keys:
                if k in meta and meta[k] not in [None, ""]:
                    try:
                        return float(meta[k])
                    except Exception:
                        pass
            return default

        self.lat = get("Composite:GPSLatitude")
        self.lon = get("Composite:GPSLongitude")
        self.altitude_absolute = getf("MakerNotes:AbsoluteAltitude", "XMP:AbsoluteAltitude")
        self.altitude_relative = getf("MakerNotes:RelativeAltitude", "XMP:RelativeAltitude")

        self.yaw_gimbal = math.radians(getf("XMP:GimbalYawDegree"))
        self.pitch_gimbal = math.radians(getf("XMP:GimbalPitchDegree"))
        self.roll_gimbal = math.radians(getf("XMP:GimbalRollDegree"))

        self.yaw_drone = math.radians(getf("XMP:FlightYawDegree"))
        self.pitch_drone = math.radians(getf("XMP:FlightPitchDegree"))
        self.roll_drone = math.radians(getf("XMP:FlightRollDegree"))

        self.dewarpflag = getf("XMP:DewarpFlag")

        dewarp = meta.get("XMP:DewarpData")
        if dewarp:
            try:
                nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", dewarp)))
                if len(nums) > 9:
                    nums = nums[-9:]
                self.fx_calib, self.fy_calib, _, _, self.k1, self.k2, self.p1, self.p2, self.k3 = nums
            except Exception as e:
                print(f"[WARN] DewarpData parse error: {e}")

        self.cx_calib = getf("XMP:CalibratedOpticalCenterX")
        self.cy_calib = getf("XMP:CalibratedOpticalCenterY")

        print(f"  Lat={self.lat:.8f}, Lon={self.lon:.8f}")
        print(f"  Alt abs={self.altitude_absolute:.2f} m  |  Alt rel={self.altitude_relative:.2f} m")
        print(f"  Gimbal YPR: {math.degrees(self.yaw_gimbal):.1f}° "
              f"{math.degrees(self.pitch_gimbal):.1f}° {math.degrees(self.roll_gimbal):.1f}°")
        return True

    # ── Chargement du masque ─────────────────────────────────────
    def load_mask(self):
        img = cv2.imread(str(self.mask_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[ERR] Impossible de lire le masque : {self.mask_path}")
            return False

        if len(img.shape) == 2:
            self.mask_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            self.bands = 3
        elif img.shape[2] == 4:
            self.mask_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            self.bands = 4
        else:
            self.mask_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.bands = 3

        self.height_mask, self.width_mask = self.mask_rgb.shape[:2]
        print(f"  Masque: {self.width_mask} x {self.height_mask} px, {self.bands} bandes")
        return True

    # ── Correction distorsion ─────────────────────────────────────
    def apply_undistortion(self):
        if self.fx_calib and self.fy_calib and self.cx_calib and self.cy_calib:
            fx, fy = self.fx_calib, self.fy_calib
            cx, cy = self.cx_calib, self.cy_calib
        else:
            fx = self.focal_length_mm * self.width_mask / self.sensor_width_mm
            fy = self.focal_length_mm * self.height_mask / self.sensor_height_mm
            cx, cy = self.width_mask / 2.0, self.height_mask / 2.0

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

        h, w = self.mask_rgb.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)

        self.mask_undistorted = cv2.undistort(self.mask_rgb, K, D, None, new_K)
        self.new_K = new_K
        try:
            self.K_inv = np.linalg.inv(new_K)
        except Exception:
            self.K_inv = None

        self.height_mask_undistorted, self.width_mask_undistorted = self.mask_undistorted.shape[:2]
        print(f"  Distorsion corrigée -> {self.width_mask_undistorted} x {self.height_mask_undistorted} px")
        return True

    def skip_undistortion(self):
        self.mask_undistorted = self.mask_rgb
        self.height_mask_undistorted = self.height_mask
        self.width_mask_undistorted = self.width_mask

        fx = self.focal_length_mm * self.width_mask / self.sensor_width_mm
        fy = self.focal_length_mm * self.height_mask / self.sensor_height_mm
        cx, cy = self.width_mask / 2.0, self.height_mask / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.new_K = K
        try:
            self.K_inv = np.linalg.inv(K)
        except Exception:
            self.K_inv = None
        return True

    # ── Hauteur sol ────────────────────────────────────────────────
    def calculate_flight_height(self):
        if not self.DEM_path:
            self.ground_elevation = None
            self.height_above_ground = abs(self.altitude_relative)
            print(f"  [INFO] Pas de DEM -> Hauteur: {self.height_above_ground:.2f} m")
            return True

        try:
            with rasterio.open(self.DEM_path) as src:
                dem_crs = src.crs.to_string()
                t_to_dem = Transformer.from_crs(f"EPSG:{self.epsg_code}", dem_crs, always_xy=True)
                t_wgs84 = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg_code}", always_xy=True)

                cx, cy = t_wgs84.transform(self.lon, self.lat)
                dx, dy = t_to_dem.transform(cx, cy)
                row, col = src.index(dx, dy)
                row = max(0, min(row, src.height - 1))
                col = max(0, min(col, src.width - 1))
                elev = float(src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
                self.ground_elevation = elev
                self.height_above_ground = abs(self.altitude_absolute - elev)
        except Exception as e:
            print(f"[WARN] DEM lecture échouée: {e}")
            self.ground_elevation = None
            self.height_above_ground = abs(self.altitude_relative)

        print(f"  Élévation sol: {self.ground_elevation} m  |  "
              f"Hauteur: {self.height_above_ground:.2f} m")
        return True

    # ── Rotation ────────────────────────────────────────────────────
    def _rotation_matrix(self, yaw, pitch, roll):
        pitch += math.radians(90)
        Rz = np.array([[math.cos(yaw), math.sin(yaw), 0],
                       [-math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])
        return Rz @ Ry @ Rx

    # ── Intersection rayon / DEM ─────────────────────────────────────
    def ray_dem_intersection(self, pixel_x, pixel_y, dem_dataset, transformer_to_dem):
        if self.K_inv is None:
            return None

        R_cam = self._rotation_matrix(self.yaw_gimbal, self.pitch_gimbal, self.roll_gimbal)
        R_dro = self._rotation_matrix(self.yaw_drone, self.pitch_drone, self.roll_drone)

        pixel_h = np.array([pixel_x, pixel_y, 1.0])
        ray_cam = self.K_inv @ pixel_h
        ray_world = R_cam @ ray_cam
        ray_world /= np.linalg.norm(ray_world)

        t_wgs84 = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg_code}", always_xy=True)
        gps_x, gps_y = t_wgs84.transform(self.lon, self.lat)
        gps_z = self.altitude_absolute

        lever = R_dro @ np.array([self.lever_x, self.lever_y, self.lever_z])
        cam_x = gps_x - lever[0]
        cam_y = gps_y - lever[1]
        cam_z = gps_z - lever[2]

        if ray_world[2] <= 0:
            return None

        ground_est = self.ground_elevation if self.ground_elevation is not None \
            else (self.altitude_absolute - abs(self.altitude_relative))
        traj_ground = abs((cam_z - ground_est) / ray_world[2])

        step_size = 0.5
        num_steps = int(traj_ground / step_size) + 50
        best = None
        min_diff = float('inf')

        for i in range(num_steps):
            t = step_size * i
            px = cam_x + t * ray_world[0]
            py = cam_y + t * ray_world[1]
            pz = cam_z - t * ray_world[2]

            try:
                dx, dy = transformer_to_dem.transform(px, py)
                row, col = dem_dataset.index(dx, dy)
                rf, cf = int(np.floor(row)), int(np.floor(col))
                rfrac, cfrac = row - rf, col - cf

                if 0 <= rf < dem_dataset.height - 1 and 0 <= cf < dem_dataset.width - 1:
                    def r(rr, cc):
                        return float(dem_dataset.read(1,
                            window=rasterio.windows.Window(cc, rr, 1, 1))[0, 0])
                    z11, z21 = r(rf, cf), r(rf, cf + 1)
                    z12, z22 = r(rf + 1, cf), r(rf + 1, cf + 1)
                    dem_z = (z11 * (1 - cfrac) * (1 - rfrac) + z21 * cfrac * (1 - rfrac) +
                             z12 * (1 - cfrac) * rfrac + z22 * cfrac * rfrac)
                elif 0 <= rf < dem_dataset.height and 0 <= cf < dem_dataset.width:
                    dem_z = float(dem_dataset.read(1,
                        window=rasterio.windows.Window(cf, rf, 1, 1))[0, 0])
                else:
                    continue

                diff = abs(pz - dem_z)
                if diff < 0.1:
                    return (px, py, dem_z)
                if diff < min_diff:
                    min_diff = diff
                    best = (px, py, dem_z)
                if pz < dem_z:
                    break
            except Exception:
                continue

        return best

    # ── Géoréférencement principal ─────────────────────────────────
    def georeference(self, output_path, subsample=100):
        if not self.DEM_path:
            print("[ERR] Un DEM est requis pour le géoréférencement précis.")
            return False

        print(f"\n  [INFO] Génération des GCPs (subsample={subsample})...")

        with rasterio.open(self.DEM_path) as dem_ds:
            dem_crs = dem_ds.crs.to_string()
            t_to_dem = Transformer.from_crs(f"EPSG:{self.epsg_code}", dem_crs, always_xy=True)

            gcps = []
            h, w = self.height_mask_undistorted, self.width_mask_undistorted
            total = (h // subsample) * (w // subsample)
            processed = 0

            for row in range(0, h, subsample):
                for col in range(0, w, subsample):
                    processed += 1
                    if processed % 50 == 0:
                        print(f"  [INFO] {processed}/{total} ({100 * processed / total:.0f}%)", end='\r')

                    inter = self.ray_dem_intersection(col, row, dem_ds, t_to_dem)
                    if inter is not None:
                        xw, yw, zw = inter
                        gcps.append(GroundControlPoint(row=row, col=col, x=xw, y=yw, z=zw))

        print(f"\n  [INFO] {len(gcps)} GCPs générés")
        if len(gcps) < 4:
            print("[ERR] Pas assez de GCPs.")
            return False

        transform = from_gcps(gcps)

        img_out = np.rot90(self.mask_undistorted, k=2)
        img_out = np.fliplr(img_out)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=h, width=w,
            count=self.bands,
            dtype=img_out.dtype,
            crs=CRS.from_epsg(self.epsg_code),
            transform=transform,
            compress='lzw'
        ) as dst:
            for i in range(self.bands):
                dst.write(img_out[:, :, i], i + 1)

        print(f"  [OK] GeoTIFF masque créé : {output_path}")
        return True


def _georeference_masks_batch(masks_folder, images_folder, output_folder, exiftool_path,
                               DEM_path, epsg_code, sensor_width_mm, sensor_height_mm,
                               focal_length_mm, subsample, mask_suffix, mask_ext,
                               image_exts=(".JPG", ".jpg", ".jpeg", ".JPEG")):
    masks_folder = Path(masks_folder)
    images_folder = Path(images_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    masks = sorted(masks_folder.glob(f"*{mask_suffix}{mask_ext}"))
    if not masks:
        print(f"[WARN] Aucun masque trouvé dans {masks_folder} "
              f"(pattern: *{mask_suffix}{mask_ext})")
        return

    print(f"\n{len(masks)} masque(s) à traiter")

    ok_count = fail_count = 0

    for mask_path in masks:
        stem = mask_path.stem
        base_name = stem[: -len(mask_suffix)] if mask_suffix else stem

        source_path = None
        for ext in image_exts:
            candidate = images_folder / (base_name + ext)
            if candidate.exists():
                source_path = candidate
                break

        if source_path is None:
            print(f"\n[SKIP] Image source introuvable pour : {mask_path.name}")
            fail_count += 1
            continue

        output_path = output_folder / (base_name + "_mask_georef.tif")

        print(f"\n▶ {mask_path.name}  (source: {source_path.name})")

        try:
            geo = MaskGeoreferencer(
                mask_path=mask_path,
                source_image_path=source_path,
                exiftool_path=exiftool_path,
                DEM_path=DEM_path,
                epsg_code=epsg_code,
                sensor_width_mm=sensor_width_mm,
                sensor_height_mm=sensor_height_mm,
                focal_length_mm=focal_length_mm,
            )

            geo.extract_metadata()
            geo.load_mask()

            if geo.dewarpflag == 0:
                geo.apply_undistortion()
            else:
                geo.skip_undistortion()

            geo.calculate_flight_height()

            if geo.georeference(output_path, subsample=subsample):
                ok_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"[ERR] Échec pour {mask_path.name} : {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print(f"\n✅ {ok_count} masque(s) géoréférencé(s)  |  ❌ {fail_count} échec(s)")
    print(f"📂 Sorties : {output_folder}")


def etape2_georeferencement():
    print(f"\n{'=' * 60}\n  ÉTAPE 2 — Géoréférencement (EXIF + DEM)\n{'=' * 60}")

    if not DEM_PATH or not EXIFTOOL_PATH:
        print("[ERREUR] DEM_PATH et EXIFTOOL_PATH doivent être définis dans le .env")
        sys.exit(1)

    _georeference_masks_batch(
        masks_folder=DOSSIER_MASKS,
        images_folder=DOSSIER_IMG,
        output_folder=DOSSIER_GEOREF,
        exiftool_path=EXIFTOOL_PATH,
        DEM_path=DEM_PATH,
        epsg_code=EPSG_CODE,
        sensor_width_mm=SENSOR_WIDTH_MM,
        sensor_height_mm=SENSOR_HEIGHT_MM,
        focal_length_mm=FOCAL_LENGTH_MM,
        subsample=SUBSAMPLE_GCP,
        mask_suffix="_contour",   # correspond au suffixe créé à l'étape 1
        mask_ext=".png",
    )


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 3 — VECTORISATION -> GEOPACKAGE
# ═══════════════════════════════════════════════════════════════
def _pixel_to_coord(row, col, transform: Affine):
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y


def _vectoriser_tif(tif_path: Path) -> list:
    """
    Lit un GeoTIFF géoréférencé, détecte les ronds rouges et renvoie une liste
    de features [{"geometry": ..., "source": ..., "id_blob": ..., "aire_px": ...}]
    """
    features = []

    with rasterio.open(tif_path) as src:
        crs = src.crs
        transform = src.transform

        # Filtre "zone centrale" : ne garder que les détections dont le centroïde
        # tombe dans les 50% centraux de la tuile, pour éviter les doublons entre
        # photos qui se chevauchent (bords communs entre deux prises de vue).
        h, w = src.height, src.width
        row_min, row_max = h // 4, h - h // 4
        col_min, col_max = w // 4, w - w // 4

        count = src.count
        if count >= 3:
            r = src.read(1).astype(np.uint8)
            g = src.read(2).astype(np.uint8)
            b = src.read(3).astype(np.uint8)
            alpha = src.read(4).astype(np.uint8) if count == 4 else np.ones_like(r) * 255
        else:
            raise ValueError(f"{tif_path.name} : attendu au moins 3 bandes (R/G/B), trouvé {count}.")

        rgb = np.dstack([r, g, b])

        masque_rouge = (
            (rgb[:, :, 0] >= ROUGE_MIN[0]) & (rgb[:, :, 0] <= ROUGE_MAX[0]) &
            (rgb[:, :, 1] >= ROUGE_MIN[1]) & (rgb[:, :, 1] <= ROUGE_MAX[1]) &
            (rgb[:, :, 2] >= ROUGE_MIN[2]) & (rgb[:, :, 2] <= ROUGE_MAX[2]) &
            (alpha > 0)
        ).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        masque_rouge = cv2.morphologyEx(masque_rouge, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masque_rouge)

        for label_id in range(1, num_labels):
            area_px = stats[label_id, cv2.CC_STAT_AREA]
            if area_px < MIN_PIXELS:
                continue

            cx, cy = centroids[label_id]
            if not (col_min <= cx < col_max and row_min <= cy < row_max):
                continue

            if GEOMETRIE == "point":
                x, y = _pixel_to_coord(cy, cx, transform)
                geom = Point(x, y)
            else:
                blob_mask = (labels == label_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                (cx_c, cy_c), rayon_px = cv2.minEnclosingCircle(contour)

                x_c, y_c = _pixel_to_coord(cy_c, cx_c, transform)

                pixel_size_x = abs(transform.a)
                pixel_size_y = abs(transform.e)
                rayon_geo = rayon_px * (pixel_size_x + pixel_size_y) / 2.0

                geom = Point(x_c, y_c).buffer(rayon_geo, resolution=64)

            features.append({
                "geometry": geom,
                "source": tif_path.name,
                "id_blob": label_id,
                "aire_px": int(area_px),
                "crs": crs,
            })

    return features


def etape3_vectorisation():
    print(f"\n{'=' * 60}\n  ÉTAPE 3 — Vectorisation -> GeoPackage\n{'=' * 60}")

    tif_files = sorted(DOSSIER_GEOREF.glob("*.tif")) + sorted(DOSSIER_GEOREF.glob("*.tiff"))
    if not tif_files:
        print(f"[ERREUR] Aucun GeoTIFF trouvé dans {DOSSIER_GEOREF}")
        sys.exit(1)

    print(f"{len(tif_files)} fichier(s) TIF trouvé(s).")

    all_features = []
    crs_ref = None

    for tif in tif_files:
        print(f"  → Traitement : {tif.name} …", end=" ", flush=True)
        try:
            feats = _vectoriser_tif(tif)
            print(f"{len(feats)} rond(s) détecté(s).")
            if feats:
                crs_ref = feats[0]["crs"]
                all_features.extend(feats)
        except Exception as e:
            print(f"[ERREUR] {e}")

    if not all_features:
        print("[INFO] Aucun rond rouge détecté dans les fichiers.")
        return

    gdf = gpd.GeoDataFrame(
        [{"source": f["source"], "id_blob": f["id_blob"], "aire_px": f["aire_px"]}
         for f in all_features],
        geometry=[f["geometry"] for f in all_features],
        crs=crs_ref,
    )

    # ── Dédoublonnage : si deux ronds se chevauchent, on garde le plus grand ──
    print("\nDédoublonnage des chevauchements entre images…")
    avant = len(gdf)
    gdf = gdf.copy()
    gdf["aire_geo"] = gdf.geometry.area
    gdf = gdf.sort_values("aire_geo", ascending=False).reset_index(drop=True)

    a_supprimer = set()
    for i, row_i in gdf.iterrows():
        if i in a_supprimer:
            continue
        for j, row_j in gdf.iterrows():
            if j <= i or j in a_supprimer:
                continue
            if row_i.geometry.overlaps(row_j.geometry) or row_i.geometry.contains(row_j.geometry):
                a_supprimer.add(j)

    gdf = gdf.drop(index=list(a_supprimer)).reset_index(drop=True)
    gdf = gdf.drop(columns=["aire_geo"])
    print(f"  {avant} ronds détectés -> {len(gdf)} après dédoublonnage ({avant - len(gdf)} supprimé(s)).")

    gdf.to_file(OUTPUT_GPKG, layer=LAYER_NAME, driver="GPKG")
    print(f"\n✅ Export terminé : {OUTPUT_GPKG.resolve()}")
    print(f"   Couche    : {LAYER_NAME}")
    print(f"   Entités   : {len(gdf)}")
    print(f"   CRS       : {crs_ref}")
    print(f"   Géométrie : {GEOMETRIE}")
    print("\nOuvrez le .gpkg directement dans QGIS (Glisser-déposer ou Couche > Ajouter).")


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    etape1_detection()
    etape2_georeferencement()
    etape3_vectorisation()

    print(f"\n{'=' * 60}\n  PIPELINE TERMINÉ\n{'=' * 60}")
    print(f"  Aperçus     : {DOSSIER_PREVIEWS}")
    print(f"  Masques     : {DOSSIER_MASKS}")
    print(f"  Géoréférencé: {DOSSIER_GEOREF}")
    print(f"  GeoPackage  : {OUTPUT_GPKG}")