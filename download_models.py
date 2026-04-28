import gdown, os, sys

os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)

files = {
    "models/wildlife_yolov8x_v2_multidetect_best.pt": os.environ.get("YOLO_GDRIVE_ID"),
    "models/wildlife_efficientnet_best.pth":           os.environ.get("EFFICIENTNET_GDRIVE_ID"),
    "data/migration_data.json":                        os.environ.get("MIGRATION_GDRIVE_ID"),
}

all_ok = True
for path, gdrive_id in files.items():
    if os.path.exists(path):
        print(f"Already exists: {path}")
        continue

    if not gdrive_id:
        print(f"ERROR: No ID provided for {path}")
        all_ok = False
        continue

    # Clean ID — remove full URL if accidentally pasted
    if "drive.google.com" in gdrive_id:
        gdrive_id = gdrive_id.split("/d/")[1].split("/")[0]
        print(f"Extracted ID: {gdrive_id}")

    print(f"Downloading {path} (ID: {gdrive_id})...")
    try:
        gdown.download(
            id=gdrive_id,      # use id= parameter instead of URL
            output=path,
            quiet=False,
            fuzzy=True,        # handles various URL formats
        )
        print(f"Done: {path}")
    except Exception as e:
        print(f"ERROR downloading {path}: {e}")
        all_ok = False

if not all_ok:
    print("Some downloads failed!")
    sys.exit(1)

print("All models ready.")