"""
Phase 3 — Generate remapped v2 splits (8-class schema).
Reads data/splits/{train,val,test}.csv and writes data/splits_v2/*.csv
"""
import os, sys
sys.path.insert(0, ".")
import pandas as pd
from pathlib import Path
from src.data.label_map_v2 import US8K_TO_8CLASS, CLASS_NAMES_V2

OUT = Path("data/splits_v2")
OUT.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/splits/{split}.csv")
    df["class_v2"] = df["class"].map(US8K_TO_8CLASS)
    missing = df[df["class_v2"].isna()]
    if not missing.empty:
        print(f"  [WARN] {split}: {len(missing)} unmapped rows:", missing["class"].unique())
        df = df.dropna(subset=["class_v2"])
    df.to_csv(OUT / f"{split}.csv", index=False)
    dist = df["class_v2"].value_counts()
    print(f"\n  {split.upper()} split ({len(df)} samples):")
    for cls in CLASS_NAMES_V2:
        n = dist.get(cls, 0)
        bar = "█" * (n // 20)
        print(f"    {cls:20s}: {n:4d}  {bar}")

print("\n  Phase 3 complete — splits_v2 written.")
