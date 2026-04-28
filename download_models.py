import gdown, os

os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)

# Upload your .pt files to Google Drive → Share → Anyone with link → Copy ID
# ID is the part after /d/ in the share URL

YOLO_ID        = "https://drive.google.com/file/d/1XPEwCTFc3IYNsriPqR4BsRmzvVkWEvMO/view?usp=sharing"
EFFICIENTNET_ID = "https://drive.google.com/file/d/1YSQMVSHFiUzKWQhnAmXw-OgTOfYKawyk/view?usp=sharing"
MIGRATION_ID   = "https://drive.google.com/file/d/1oEfTQOwV1HQF1fCm9MFcJ6Kchfzk--1s/view?usp=sharing"

if not os.path.exists("models/wildlife_yolov8x_v2_multidetect_best.pt"):
    print("Downloading YOLO model...")
    gdown.download(f"https://drive.google.com/uc?id={YOLO_ID}",
                   "models/wildlife_yolov8x_v2_multidetect_best.pt")

if not os.path.exists("models/wildlife_efficientnet_best.pth"):
    print("Downloading EfficientNet...")
    gdown.download(f"https://drive.google.com/uc?id={EFFICIENTNET_ID}",
                   "models/wildlife_efficientnet_best.pth")

if not os.path.exists("data/migration_data.json"):
    print("Downloading migration data...")
    gdown.download(f"https://drive.google.com/uc?id={MIGRATION_ID}",
                   "data/migration_data.json")

print("All models ready.")