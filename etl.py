import pandas as pd
import numpy as np

print("Loading AIS data in chunks...")

PORT_LAT = (54.5, 57.8)
PORT_LON = (8.0, 15.5)

chunks = []
for chunk in pd.read_csv(
    'data/aisdk-2024-03-01.csv',
    usecols=['# Timestamp','MMSI','Latitude','Longitude','SOG','Ship type'],
    low_memory=False,
    chunksize=100_000
):
    chunk.columns = ['timestamp','mmsi','lat','lon','sog','ship_type']
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], dayfirst=True)
    chunk['ship_type_num'] = pd.to_numeric(chunk['ship_type'], errors='coerce')
    in_port = (
        chunk['lat'].between(*PORT_LAT) &
        chunk['lon'].between(*PORT_LON) &
        chunk['ship_type'].isin(['Cargo','Tanker'])
    )
    filtered = chunk[in_port]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"  chunk processed, in-port rows so far: {sum(len(c) for c in chunks)}")

if not chunks:
    print("❌ No cargo ships found — check bounding box or ship type")
    exit()

port_df = pd.concat(chunks).sort_values(['mmsi','timestamp'])
print(f"Total in-port signals: {len(port_df):,}")

events = []
for mmsi, grp in port_df.groupby('mmsi'):
    arrived  = grp['timestamp'].iloc[0]
    departed = grp['timestamp'].iloc[-1]
    duration_h = (departed - arrived).total_seconds() / 3600
    if 1 < duration_h < 72:
        events.append({'mmsi': mmsi, 'arrived': arrived,
                       'departed': departed, 'duration_hours': round(duration_h,2)})

events_df = pd.DataFrame(events)
events_df.to_csv('data/port_events.csv', index=False)
print(f"Port visits: {len(events_df)}")

N_BERTHS = 8
time_range = pd.date_range(events_df['arrived'].min(),
                           events_df['departed'].max(), freq='1h')
rows = []
for t in time_range:
    active = ((events_df['arrived'] <= t) & (events_df['departed'] >= t)).sum()
    rows.append({'timestamp': t,
                 'occupied_berths': int(min(active, N_BERTHS)),
                 'occupancy_rate': round(min(active/N_BERTHS, 1.0), 4),
                 'hour_of_day': t.hour,
                 'day_of_week': t.dayofweek,
                 'month': t.month})

occ_df = pd.DataFrame(rows)
occ_df.to_csv('data/berth_occupancy_hourly.csv', index=False)
print(f"\n✅ Done! {len(occ_df)} hourly rows")
print(f"Avg occupancy: {occ_df['occupancy_rate'].mean():.1%}")
print(occ_df.head(10))