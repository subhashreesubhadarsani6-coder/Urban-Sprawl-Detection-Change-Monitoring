import pyarrow.parquet as pq
import io
from PIL import Image
import numpy as np

for pf in ['dataset/train-00000-of-00001.parquet', 'dataset/test-00000-of-00001.parquet']:
    table = pq.read_table(pf)
    df = table.to_pandas()
    print(f'=== {pf} ({len(df)} rows) ===')
    for idx in range(len(df)):
        i1 = df['image1'].iloc[idx]
        i2 = df['image2'].iloc[idx]
        m = df['mask'].iloc[idx]
        img1 = Image.open(io.BytesIO(i1['bytes']))
        img2 = Image.open(io.BytesIO(i2['bytes']))
        mask = Image.open(io.BytesIO(m['bytes']))
        m_arr = np.array(mask)
        uniques, counts = np.unique(m_arr, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(uniques, counts)}
        total_px = m_arr.size
        changed_px = dist.get(1, 0)
        pct_change = round((changed_px / total_px) * 100, 2)
        print(f"  Row {idx:2d}: path1={i1.get('path')} | size={img1.size} | mask_size={mask.size} | changed_px={changed_px:,} ({pct_change}%) | uniques={list(uniques)}")
