import zipfile
import io
from PIL import Image
import numpy as np
import pyarrow.parquet as pq

train_city_order = [
    "nantes", "pisa", "rennes", "cupertino", "aguasclaras",
    "bordeaux", "beihai", "hongkong", "beirut", "paris",
    "mumbai", "saclay_e", "bercy", "abudhabi"
]

test_city_order = [
    "rio", "saclay_w", "valencia", "chongqing", "montpellier",
    "brasilia", "norcia", "dubai", "milano", "lasvegas"
]

def verify_alignment():
    print("Verifying Train Scenes Alignment:")
    z_train = zipfile.ZipFile("dataset/Onera Satellite Change Detection dataset - Train Labels.zip")
    df_train = pq.read_table("dataset/train-00000-of-00001.parquet").to_pandas()
    for idx, city in enumerate(train_city_order):
        m_bytes_parquet = df_train['mask'].iloc[idx]['bytes']
        m_img_parquet = Image.open(io.BytesIO(m_bytes_parquet))
        arr_parquet = np.array(m_img_parquet)

        zip_mask_bytes = z_train.read(f"Onera Satellite Change Detection dataset - Train Labels/{city}/cm/cm.png")
        arr_zip = np.array(Image.open(io.BytesIO(zip_mask_bytes)))
        # In zip cm.png, change is in the Red channel (255 for change)
        if arr_zip.ndim == 3:
            zip_bin = (arr_zip[..., 0] > 0).astype(np.uint8)
        else:
            zip_bin = (arr_zip > 0).astype(np.uint8)
        parq_bin = (arr_parquet > 0).astype(np.uint8)
        diff = np.sum(zip_bin != parq_bin)
        print(f"  [{idx:2d}] {city:12s}: shape={arr_parquet.shape} | zip_shape={arr_zip.shape} | diff_pixels={diff}")

    print("\nVerifying Test Scenes Alignment:")
    z_test = zipfile.ZipFile("dataset/Onera Satellite Change Detection dataset - Test Labels.zip")
    df_test = pq.read_table("dataset/test-00000-of-00001.parquet").to_pandas()
    for idx, city in enumerate(test_city_order):
        m_bytes_parquet = df_test['mask'].iloc[idx]['bytes']
        m_img_parquet = Image.open(io.BytesIO(m_bytes_parquet))
        arr_parquet = np.array(m_img_parquet)

        zip_mask_bytes = z_test.read(f"Onera Satellite Change Detection dataset - Test Labels/{city}/cm/cm.png")
        arr_zip = np.array(Image.open(io.BytesIO(zip_mask_bytes)))
        if arr_zip.ndim == 3:
            zip_bin = (arr_zip[..., 0] > 0).astype(np.uint8)
        else:
            zip_bin = (arr_zip > 0).astype(np.uint8)
        parq_bin = (arr_parquet > 0).astype(np.uint8)
        diff = np.sum(zip_bin != parq_bin)
        print(f"  [{idx:2d}] {city:12s}: shape={arr_parquet.shape} | zip_shape={arr_zip.shape} | diff_pixels={diff}")

if __name__ == "__main__":
    verify_alignment()
