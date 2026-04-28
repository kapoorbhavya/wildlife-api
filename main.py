# ============================================================
# main.py — Wildlife Migration API v11.0
# Uses YOLOv8x for TIGHT bounding boxes (like car detection)
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import timm
import numpy as np
import cv2
import json
import os
import io
import time
import tempfile
import base64
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Tuple
from pydantic import BaseModel
from collections import defaultdict
from ultralytics import YOLO
# ── Download models if not present (for Railway deployment) ──
import os

def download_models_if_needed():
    import gdown
    os.makedirs("models", exist_ok=True)
    os.makedirs("data",   exist_ok=True)

    files = {
        "models/wildlife_yolov8x_v2_multidetect_best.pt": "YOUR_YOLO_GDRIVE_ID",
        "models/wildlife_efficientnet_best.pth":           "YOUR_EFFICIENTNET_GDRIVE_ID",
        "data/migration_data.json":                        "YOUR_MIGRATION_GDRIVE_ID",
    }
    for path, gdrive_id in files.items():
        if not os.path.exists(path):
            print(f"  Downloading {path}...")
            gdown.download(
                f"https://drive.google.com/uc?id={gdrive_id}",
                path, quiet=False
            )

download_models_if_needed()

# ─────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

YOLO_PATH = MODELS_DIR / "wildlife_yolov8x_v2_multidetect_best.pt"
EFFICIENTNET_PATH = MODELS_DIR / "wildlife_efficientnet_best.pth"
MIGRATION_DATA_PATH = DATA_DIR   / "migration_data.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIES_CLASSES = [
    "Asian Elephant", "Common-Myna", "House-Crow",
    "Indian-Grey-Hornbill", "donkey", "horse", "tigers", "zebra"
]
NUM_CLASSES = len(SPECIES_CLASSES)

SPECIES_COLORS_HEX = {
    "Asian Elephant":       "#FF6B6B",
    "Common-Myna":          "#4ECDC4",
    "House-Crow":           "#45B7D1",
    "Indian-Grey-Hornbill": "#96CEB4",
    "donkey":               "#FFEAA7",
    "horse":                "#DDA0DD",
    "tigers":               "#FF8C00",
    "zebra":                "#98D8C8",
}

def hex_to_bgr(h: str) -> Tuple:
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b,g,r)

SPECIES_COLORS_BGR = {sp: hex_to_bgr(c) for sp,c in SPECIES_COLORS_HEX.items()}

# Detection thresholds
DETECTION_CONF  = 0.15   # YOLOv8 detection confidence
CLASSIFY_CONF   = 0.20   # Minimum classification confidence
NMS_IOU_THRESH  = 0.40   # IoU threshold for NMS
MAX_DETECTIONS  = 20     # Max animals per image

infer_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

# ─────────────────────────────────────────────────────────────
# 2. APP SETUP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Wildlife Migration API", version="11.0.0")
# Replace your existing CORSMiddleware with this:
# Replace CORSMiddleware with:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # ← must be False when origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# 3. LOAD MODELS
# ─────────────────────────────────────────────────────────────
print("="*60)
print(f"  Loading models on {DEVICE}")

# Phase 2: YOLOv8x for TIGHT bounding boxes
print("  Loading YOLOv8x detector...")
detector = YOLO(str(YOLO_PATH))
print("  ✅ YOLOv8x loaded")
# After loading detector, add:
print(f"  YOLO model classes: {detector.names}")
# Phase 3: EfficientNet for species classification
print("  Loading EfficientNet classifier...")
classifier = timm.create_model(
    "tf_efficientnetv2_s",
    pretrained=False,
    num_classes=NUM_CLASSES,
    drop_rate=0.3,
    drop_path_rate=0.2,
)
classifier.load_state_dict(
    torch.load(str(EFFICIENTNET_PATH), map_location=DEVICE)
)
classifier = classifier.to(DEVICE).eval()
print("  ✅ EfficientNet loaded")

# Migration data
with open(MIGRATION_DATA_PATH) as f:
    migration_data: dict = json.load(f)
print(f"  ✅ Migration data: {len(migration_data)} species")
print("="*60)

# ─────────────────────────────────────────────────────────────
# 4. PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────

class Top3Prediction(BaseModel):
    species: str
    confidence: float

class SpeciesInfo(BaseModel):
    common_name: str
    scientific_name: str
    conservation_status: str
    habitat: str
    diet: str
    population_estimate: int
    fun_facts: List[str]

class MigrationData(BaseModel):
    migration_type: str
    migration_description: str
    avg_distance_km: float
    migration_route: List[dict]
    seasonal_positions: dict
    breeding_grounds: dict
    key_habitats: List[str]

class AnimalDetection(BaseModel):
    animal_id: int
    bbox: List[int]
    species: str
    confidence: float
    detection_confidence: float
    color_hex: str
    top3_predictions: List[Top3Prediction]
    species_info: SpeciesInfo
    migration_data: MigrationData

class ImageAnalysisResponse(BaseModel):
    success: bool
    filename: str
    image_size: dict
    processing_time_seconds: float
    total_animals_detected: int
    unique_species_count: int
    species_counts: dict
    detections: List[AnimalDetection]
    annotated_image_base64: str

# ─────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def classify_crop(crop: Image.Image):
    """Classify a cropped animal image using EfficientNet"""
    arr = np.array(crop.convert("RGB"))
    tensor = infer_transform(image=arr)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(
            classifier(tensor), dim=1
        )[0].cpu().numpy()

    idx = int(probs.argmax())
    return SPECIES_CLASSES[idx], round(float(probs[idx]), 4), probs


def get_top3(probs: np.ndarray) -> List[Top3Prediction]:
    idx = np.argsort(probs)[::-1][:3]
    return [Top3Prediction(
        species=SPECIES_CLASSES[i],
        confidence=round(float(probs[i]), 4)
    ) for i in idx]


def build_species_info(mig: dict, sp: str) -> SpeciesInfo:
    return SpeciesInfo(
        common_name         = mig.get("common_name", sp),
        scientific_name     = mig.get("scientific_name", "Unknown"),
        conservation_status = mig.get("conservation_status", "Unknown"),
        habitat             = mig.get("habitat", "Unknown"),
        diet                = mig.get("diet", "Unknown"),
        population_estimate = mig.get("population_estimate", 0),
        fun_facts           = mig.get("fun_facts", []),
    )


def build_migration_data(mig: dict) -> MigrationData:
    return MigrationData(
        migration_type        = mig.get("migration_type", "Resident"),
        migration_description = mig.get("migration_description", ""),
        avg_distance_km       = float(mig.get("avg_distance_km", 0)),
        migration_route       = mig.get("migration_route", []),
        seasonal_positions    = mig.get("seasonal_positions", {}),
        breeding_grounds      = mig.get("breeding_grounds", {}),
        key_habitats          = mig.get("key_habitats", []),
    )


def box_iou(a: List[int], b: List[int]) -> float:
    """Calculate IoU between two boxes"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0]) * (a[3]-a[1])
    ub = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (ua + ub - inter + 1e-6)

# ─────────────────────────────────────────────────────────────
# 6. MAIN DETECTION PIPELINE
# Uses YOLOv8x → tight boxes → EfficientNet classify each box
# ─────────────────────────────────────────────────────────────

def detect_all_animals(image: Image.Image) -> List[AnimalDetection]:
    iw, ih = image.size
    print(f"\n  Image: {iw}×{ih}")

    # ── Step 1: YOLOv8x Detection ─────────────────────────────
    img_np_rgb = np.array(image.convert("RGB"))

    results = detector.predict(
        img_np_rgb,
        conf=DETECTION_CONF,
        iou=NMS_IOU_THRESH,
        imgsz=1280,
        max_det=MAX_DETECTIONS,
        augment=True,
        verbose=False
    )[0]

    
    print(f"  Raw boxes found: {len(results.boxes)}")
    # ── Step 2: Process each detection ───────────────────────
    raw_detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        det_conf = float(box.conf)
        bw, bh = x2 - x1, y2 - y1

        # Skip tiny boxes
        if bw < iw * 0.01 or bh < ih * 0.01:
            continue

        # Skip whole-image boxes
        if bw > iw * 0.97 and bh > ih * 0.97:
            continue

        # Add padding for better classification
        pad = int(min(iw, ih) * 0.015)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(iw, x2 + pad)
        cy2 = min(ih, y2 + pad)

        # Crop and classify
        crop = image.crop((cx1, cy1, cx2, cy2))
        species, cls_conf, probs = classify_crop(crop)

        # ── Confidence rescue — retry with larger crop if unsure ──
        if cls_conf < 0.40:
            pad_large = int(min(iw, ih) * 0.05)
            crop2 = image.crop((
                max(0, x1 - pad_large),
                max(0, y1 - pad_large),
                min(iw, x2 + pad_large),
                min(ih, y2 + pad_large)
            ))
            species2, cls_conf2, probs2 = classify_crop(crop2)
            if cls_conf2 > cls_conf:
                species, cls_conf, probs = species2, cls_conf2, probs2
                print(f"  Rescue improved: {species} ({cls_conf:.1%})")

        raw_detections.append({
            'bbox': [x1, y1, x2, y2],
            'det_conf': det_conf,
            'species': species,
            'cls_conf': cls_conf,
            'probs': probs,
        })

        print(f"  Detection: {species} ({cls_conf:.1%}) bbox=[{x1},{y1},{x2},{y2}]")

    # ── Step 3: NMS — remove duplicate boxes ─────────────────
    raw_detections.sort(key=lambda x: -x['cls_conf'])

    final_detections = []
    for det in raw_detections:
        is_duplicate = False
        for kept in final_detections:
            iou = box_iou(det['bbox'], kept['bbox'])
            if iou > NMS_IOU_THRESH and det['species'] == kept['species']:
                is_duplicate = True
                break
        if not is_duplicate:
            final_detections.append(det)

    print(f"  After NMS: {len(final_detections)} unique detections")

    # ── Step 4: Build AnimalDetection objects ─────────────────
    animals = []
    for i, det in enumerate(final_detections):
        sp  = det['species']
        mig = migration_data.get(sp, {})

        animals.append(AnimalDetection(
            animal_id            = i + 1,
            bbox                 = det['bbox'],
            species              = sp,
            confidence           = det['cls_conf'],
            detection_confidence = det['det_conf'],
            color_hex            = SPECIES_COLORS_HEX.get(sp, "#FFFFFF"),
            top3_predictions     = get_top3(det['probs']),
            species_info         = build_species_info(mig, sp),
            migration_data       = build_migration_data(mig),
        ))
        print(f"  ✅ #{i+1}: {sp} {det['cls_conf']:.1%}")

    sp_counts = defaultdict(int)
    for a in animals:
        sp_counts[a.species] += 1
    print(f"\n  Final: {len(animals)} animals — {dict(sp_counts)}")

    return animals

# ─────────────────────────────────────────────────────────────
# 7. DRAWING — Tight boxes like car detection
# ─────────────────────────────────────────────────────────────

def draw_detections(
    image: Image.Image,
    animals: List[AnimalDetection],
) -> str:
    img_np = cv2.cvtColor(
        np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR
    )
    ih, iw = img_np.shape[:2]

    scale     = max(iw, ih) / 800.0
    thickness = max(2, int(3 * scale))
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fscale    = max(0.5, 0.7 * scale)

    sp_count: Dict[str, int] = defaultdict(int)
    for a in animals:
        sp_count[a.species] += 1

    for animal in animals:
        x1, y1, x2, y2 = animal.bbox
        color = SPECIES_COLORS_BGR.get(animal.species, (200, 200, 200))
        pct   = int(animal.confidence * 100)

        # ── Draw bounding box ─────────────────────────────────
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)

        # ── Measure label size first ──────────────────────────
        label = f"{animal.species} {pct}%"
        (tw, th), bl = cv2.getTextSize(label, font, fscale, 1)
        label_w = tw + 10
        label_h = th + bl + 8

        # ── Fix 1: shift label LEFT if it overflows right edge ─
        lx1 = x1
        if lx1 + label_w > iw:
            lx1 = max(0, iw - label_w)   # push left
        lx2 = min(iw, lx1 + label_w)

        # ── Fix 2: place label BELOW box if it overflows top ──
        if y1 - label_h < 0:
            # not enough room above — draw below top edge of box
            ly1 = y1
            ly2 = y1 + label_h
            text_y = ly1 + th + 2
        else:
            # normal — draw above box
            ly1 = y1 - label_h
            ly2 = y1
            text_y = ly2 - bl - 2

        # ── Draw label background ─────────────────────────────
        cv2.rectangle(img_np, (lx1, ly1), (lx2, ly2), color, -1)

        # ── Draw label text ───────────────────────────────────
        cv2.putText(
            img_np, label,
            (lx1 + 5, text_y),
            font, fscale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA
        )

        # ── Animal ID circle — also keep inside image ─────────
        radius  = max(12, int(15 * scale))
        cx_circ = max(radius, min(iw - radius, x1 + radius))
        cy_circ = max(radius, min(ih - radius, y1 - radius))
        cv2.circle(img_np, (cx_circ, cy_circ), radius, color, -1)
        cv2.circle(img_np, (cx_circ, cy_circ), radius, (255, 255, 255), 1)

        # Center the ID number text in the circle
        id_str = str(animal.animal_id)
        (iw_t, ih_t), _ = cv2.getTextSize(id_str, font, fscale * 0.8, 1)
        cv2.putText(
            img_np, id_str,
            (cx_circ - iw_t // 2, cy_circ + ih_t // 2),
            font, fscale * 0.8,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    # ── Summary panel ─────────────────────────────────────────
    total = len(animals)
    n_sp  = len(sp_count)

    pf  = max(0.4, 0.55 * scale)
    lh  = max(22, int(28 * scale))
    lns = [f"Animals: {total}  Species: {n_sp}"]
    lns.append("-" * 28)
    for sp, cnt in sp_count.items():
        lns.append(f"  {sp}: {cnt}")

    mw  = max(cv2.getTextSize(l, font, pf, 1)[0][0] for l in lns)
    pw  = mw + 30
    ph2 = lh * len(lns) + 20
    px1 = iw - pw - 15
    py1 = ih - ph2 - 15
    px2 = iw - 15
    py2 = ih - 15

    overlay = img_np.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (15, 25, 45), -1)
    cv2.addWeighted(overlay, 0.8, img_np, 0.2, 0, img_np)
    cv2.rectangle(img_np, (px1, py1), (px2, py2), (100, 120, 160), 2)

    for i, line in enumerate(lns):
        cv2.putText(
            img_np, line,
            (px1 + 10, py1 + 15 + i * lh),
            font, pf, (200, 220, 255), 1, cv2.LINE_AA
        )

    # ── Title bar ─────────────────────────────────────────────
    title = f"Wildlife Detection  |  {total} animal(s)  |  {n_sp} species"
    ts    = max(0.45, 0.65 * scale)
    (ttw, tth), _ = cv2.getTextSize(title, font, ts, 2)
    tx = max(0, (iw - ttw) // 2)
    ty = max(30, int(38 * scale))

    cv2.putText(img_np, title, (tx+2, ty+2), font, ts, (0,0,0),     2, cv2.LINE_AA)
    cv2.putText(img_np, title, (tx,   ty),   font, ts, (255,255,255), 2, cv2.LINE_AA)

    # ── Encode to base64 ──────────────────────────────────────
    rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
# ─────────────────────────────────────────────────────────────
# 8. API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status":  "running",
        "version": "11.0.0",
        "device":  str(DEVICE),
        "message": "Wildlife Migration Tracking API",
        "endpoints": [
            "GET  /health",
            "GET  /species",
            "GET  /species/{name}",
            "POST /analyze-image",
            "POST /analyze-video",
        ]
    }


@app.get("/health")
async def health():
    return {
        "status":           "healthy",
        "device":           str(DEVICE),
        "detection_conf":   DETECTION_CONF,
        "classify_conf":    CLASSIFY_CONF,
        "nms_iou":          NMS_IOU_THRESH,
        "species_count":    NUM_CLASSES,
        "species":          SPECIES_CLASSES,
        "models": {
            "detector":    "YOLOv8x (tight bounding boxes)",
            "classifier":  "EfficientNet-V2-S (species ID)",
        }
    }


@app.get("/species")
async def get_all_species():
    return {
        "total": len(SPECIES_CLASSES),
        "species": {
            sp: {
                **{k: migration_data.get(sp, {}).get(k, "")
                   for k in ["common_name", "scientific_name",
                              "conservation_status", "habitat",
                              "migration_type"]},
                "avg_distance_km": migration_data.get(sp, {}).get("avg_distance_km", 0),
                "color_hex": SPECIES_COLORS_HEX.get(sp, "#FFF"),
            } for sp in SPECIES_CLASSES
        }
    }


@app.get("/species/{name}")
async def get_species(name: str):
    mig = migration_data.get(name)
    if not mig:
        for k in migration_data:
            if k.lower() == name.lower():
                mig, name = migration_data[k], k
                break
    if not mig:
        raise HTTPException(
            404, {"error": f"'{name}' not found",
                  "available": SPECIES_CLASSES}
        )
    return {"species": name, "data": mig}


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload wildlife image → get:
    - TIGHT bounding boxes around each animal (like car detection)
    - Species name + confidence for each animal
    - Full migration data for each species
    - Base64 annotated image
    """
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Use JPEG/PNG. Got: {file.content_type}")

    try:
        start     = time.time()
        img_bytes = await file.read()

        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Cannot decode image.")

        iw, ih = image.size
        print(f"\n{'─'*55}")
        print(f"  File: {file.filename} ({iw}×{ih})")

        # Run complete pipeline
        animals = detect_all_animals(image)

        # Count species
        sp_count: Dict[str, int] = defaultdict(int)
        for a in animals:
            sp_count[a.species] += 1

        # Draw annotated image
        b64 = draw_detections(image, animals) if animals else ""

        elapsed = round(time.time() - start, 3)

        return ImageAnalysisResponse(
            success                 = True,
            filename                = file.filename or "unknown",
            image_size              = {"width": iw, "height": ih},
            processing_time_seconds = elapsed,
            total_animals_detected  = len(animals),
            unique_species_count    = len(sp_count),
            species_counts          = dict(sp_count),
            detections              = animals,
            annotated_image_base64  = b64,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    sample_rate: int = 30,
):
    """
    Upload wildlife video → analyze frames → detect species + migration data
    sample_rate: analyze every Nth frame (default 30)
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, f"Expected video. Got: {file.content_type}")

    tmp_path = None
    try:
        start  = time.time()
        suffix = Path(file.filename or "v.mp4").suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)

        agg: Dict[str, dict] = {}
        fc = ac = 0

        while cap.isOpened() and ac < MAX_DETECTIONS * 5:
            ret, frame = cap.read()
            if not ret:
                break

            if fc % sample_rate == 0:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                animals = detect_all_animals(pil)

                for a in animals:
                    sp = a.species
                    if sp not in agg:
                        agg[sp] = {
                            "species":           sp,
                            "frame_appearances": 1,
                            "max_confidence":    a.confidence,
                            "species_info":      a.species_info.dict(),
                            "migration_data":    a.migration_data.dict(),
                            "color_hex":         a.color_hex,
                        }
                    else:
                        agg[sp]["frame_appearances"] += 1
                        if a.confidence > agg[sp]["max_confidence"]:
                            agg[sp]["max_confidence"] = a.confidence
                ac += 1
            fc += 1

        cap.release()
        os.unlink(tmp_path)
        tmp_path = None

        return {
            "success":  True,
            "filename": file.filename,
            "video_info": {
                "total_frames":    total,
                "fps":             round(fps, 2),
                "analyzed_frames": ac,
                "sample_rate":     sample_rate,
            },
            "processing_time_seconds": round(time.time() - start, 3),
            "unique_species_detected": len(agg),
            "species_detections":      list(agg.values()),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),  # Railway sets PORT automatically
        reload=False,                              # Never reload in production
        timeout_keep_alive=120,                    # 2 min timeout for slow inference
    )