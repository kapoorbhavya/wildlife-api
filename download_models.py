import gdown, os

os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)

files = {
    "models/wildlife_yolov8x_v2_multidetect_best.pt": os.environ.get("YOLO_GDRIVE_ID",        "https://drive.google.com/file/d/1XPEwCTFc3IYNsriPqR4BsRmzvVkWEvMO/view?usp=sharing"),
    "models/wildlife_efficientnet_best.pth":           os.environ.get("EFFICIENTNET_GDRIVE_ID", "https://drive.google.com/file/d/1YSQMVSHFiUzKWQhnAmXw-OgTOfYKawyk/view?usp=sharing"),
    "data/migration_data.json":                        os.environ.get("MIGRATION_GDRIVE_ID",    "https://drive.google.com/file/d/1oEfTQOwV1HQF1fCm9MFcJ6Kchfzk--1s/view?usp=sharing"),
}

for path, gdrive_id in files.items():
    if not os.path.exists(path) and gdrive_id:
        print(f"Downloading {path}...")
        gdown.download(
            f"https://drive.google.com/uc?id={gdrive_id}",
            path, quiet=False
        )
        print(f"Done: {path}")

print("All models ready.")