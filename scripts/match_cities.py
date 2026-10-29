import zipfile
import io
from PIL import Image

def inspect_zip_masks(zip_path):
    print(f"=== {zip_path} ===")
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.endswith("cm.png"):
                data = z.read(name)
                img = Image.open(io.BytesIO(data))
                city = name.split("/")[1]
                print(f"  City: {city:15s} | size: {img.size}")

inspect_zip_masks("dataset/Onera Satellite Change Detection dataset - Train Labels.zip")
inspect_zip_masks("dataset/Onera Satellite Change Detection dataset - Test Labels.zip")
