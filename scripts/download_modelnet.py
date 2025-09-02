# scripts/download_data.py

import argparse
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# Configuration for datasets
# Using a dictionary makes it easy to add more datasets later
DATASETS = {
    "ModelNet40": {
        "url": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
        "filename": "ModelNet40.zip",
        "unzipped_dir": "ModelNet40"
    },
    "ModelNet10": {
        "url": "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        "filename": "ModelNet10.zip",
        "unzipped_dir": "ModelNet10"
    }
}

def download_and_unzip(dataset_name: str, data_dir: Path):
    """Downloads, shows a progress bar, and unzips the dataset."""
    
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[dataset_name]
    url = config["url"]
    zip_path = data_dir / config["filename"]
    unzipped_path = data_dir / config["unzipped_dir"]

    # 1. Check if data is already downloaded and unzipped
    if unzipped_path.exists():
        print(f"✅ Dataset '{dataset_name}' already exists in {unzipped_path}. Skipping download.")
        return

    # 2. Create the data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{data_dir}' created.")

    # 3. Download the file with a progress bar
    print(f"Downloading {dataset_name} from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc=config["filename"],
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        print("Download complete.")

        # 4. Unzip the file
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Unzipped to {data_dir}.")

        # 5. Clean up the zip file
        zip_path.unlink()
        print(f"Removed temporary file: {zip_path}.")
        print(f"✅ Successfully downloaded and prepared {dataset_name}.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading file: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare ModelNet datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ModelNet40",
        choices=DATASETS.keys(),
        help="The dataset to download."
    )
    args = parser.parse_args()

    # Place the data in a `data/raw` directory relative to the project root
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    
    download_and_unzip(dataset_name=args.dataset, data_dir=raw_data_dir)