from pathlib import Path
import subprocess
import sys
import zipfile

from config import RAW_DIR, TRAIN_DIR


def run_command(command):
    print("Running:", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("\nDownload failed.")
        print("Make sure the Kaggle CLI is installed and authenticated.")
        print("Also make sure you have accepted the Dogs vs Cats competition rules.")
        sys.exit(result.returncode)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    train_zip = RAW_DIR / "train.zip"

    if not train_zip.exists():
        run_command([
            "kaggle",
            "competitions",
            "download",
            "dogs-vs-cats",
            "-f",
            "train.zip",
            "-p",
            str(RAW_DIR),
        ])
    else:
        print("train.zip already exists. Skipping download.")

    if TRAIN_DIR.exists() and any(TRAIN_DIR.glob("*.jpg")):
        print(f"Training images already extracted: {TRAIN_DIR}")
        return

    if not train_zip.exists():
        raise FileNotFoundError(f"Could not find {train_zip}")

    print("Extracting train.zip...")
    with zipfile.ZipFile(train_zip, "r") as zf:
        zf.extractall(RAW_DIR)

    # Kaggle's archive normally extracts to data/raw/train/
    if TRAIN_DIR.exists():
        count = len(list(TRAIN_DIR.glob("*.jpg")))
        print(f"Done. Found {count} training images.")
    else:
        raise FileNotFoundError(
            "Extraction finished, but data/raw/train was not found. "
            "Please inspect data/raw and update TRAIN_DIR if needed."
        )


if __name__ == "__main__":
    main()
