import pandas as pd
import numpy as np

np.random.seed(42)
base = pd.read_csv('data/berth_occupancy_hourly.csv')
frames = []
for day in range(90):
    d = base.copy()
    d['timestamp'] = pd.to_datetime(d['timestamp']) + pd.Timedelta(days=day)
    noise = np.random.normal(0, 0.08, len(d))
    d['occupancy_rate'] = (d['occupancy_rate'] + noise).clip(0.1, 1.0).round(4)
    d['occupied_berths'] = (d['occupancy_rate'] * 8).round().astype(int)
    d['hour_of_day'] = d['timestamp'].dt.hour
    d['day_of_week'] = d['timestamp'].dt.dayofweek
    d['month'] = d['timestamp'].dt.month
    frames.append(d)

df = pd.concat(frames)
df.to_csv('data/berth_occupancy_hourly.csv', index=False)
print(f'✅ {len(df)} rows ready')