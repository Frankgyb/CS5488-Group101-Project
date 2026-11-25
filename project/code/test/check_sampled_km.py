"""Quick sanity check on the sampled_km table (tips / speed / fare ranges).

Location: project/code/test/check_sampled_km.py | PROJECT_ROOT depth: 2
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # .../project
SAMPLED_KM_PATH = PROJECT_ROOT / "data" / "output" / "sampled_km"

df = pd.read_parquet(SAMPLED_KM_PATH)
print("sampled_km shape:", df.shape)

# tips check
tips_orig = np.expm1(df["log_tips"])
print(f"tips: min={tips_orig.min():.2f}, max={tips_orig.max():.2f}, >100 count={(tips_orig>100).sum()}")

# speed check
print(f"speed_mph: min={df['speed_mph'].min():.2f}, max={df['speed_mph'].max():.2f}, "
      f"outside [1,80] count={((df['speed_mph']<1)|(df['speed_mph']>80)).sum()}")

# fare check
fare_orig = np.expm1(df["log_fare"])
print(f"fare: min={fare_orig.min():.2f}, max={fare_orig.max():.2f}, "
      f"p95={fare_orig.quantile(0.95):.2f}, p99={fare_orig.quantile(0.99):.2f}")
