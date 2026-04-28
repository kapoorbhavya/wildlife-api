import gdown, os, sys

os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)

files = {
    "models/wildlife_yolov8x_v2_multidetect_best.pt": os.environ.get("YOLO_GDRIVE_ID"),
    "models/wildlife_efficientnet_best.pth":           os.environ.get("EFFICIENTNET_GDRIVE_ID"),
    "data/migration_data.json":                        os.environ.get("MIGRATION_GDRIVE_ID"),
}

print("=== Starting model downloads ===")
print(f"YOLO ID:         {os.environ.get('YOLO_GDRIVE_ID', 'NOT SET')}")
print(f"EfficientNet ID: {os.environ.get('EFFICIENTNET_GDRIVE_ID', 'NOT SET')}")
print(f"Migration ID:    {os.environ.get('MIGRATION_GDRIVE_ID', 'NOT SET')}")

all_ok = True
for path, gdrive_id in files.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"✅ Already exists: {path} ({size/1e6:.1f} MB)")
        continue

    if not gdrive_id or gdrive_id == "None":
        print(f"❌ No ID for {path} — skipping")
        all_ok = False
        continue

    # Strip full URL if accidentally pasted
    if "drive.google.com" in gdrive_id:
        gdrive_id = gdrive_id.split("/d/")[1].split("/")[0]

    print(f"⬇️  Downloading: {path}")
    print(f"   ID: {gdrive_id}")

    try:
        result = gdown.download(
            id=gdrive_id,
            output=path,
            quiet=False,
            fuzzy=True,
        )
        if result and os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ Done: {path} ({size/1e6:.1f} MB)")
        else:
            print(f"❌ Download returned None for {path}")
            all_ok = False
    except Exception as e:
        print(f"❌ Exception: {e}")
        all_ok = False

print("=== Download complete ===" if all_ok else "=== Some downloads FAILED ===")
if not all_ok:
    sys.exit(1)

    