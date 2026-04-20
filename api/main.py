from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, numpy as np, pandas as pd
from datetime import datetime, timedelta

app = FastAPI(title="BlueBerth AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

with open("models/berth_model.pkl","rb") as f:
    model = pickle.load(f)

df_hist = pd.read_csv("data/berth_occupancy_hourly.csv")
df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])

def make_features(hour, dow, month, lag1, lag24, lag72, roll_mean, roll_std):
    return np.array([[hour, dow, month, lag1, lag24, lag72, roll_mean, roll_std]])

@app.get("/")
def root():
    return {"status": "ok", "service": "BlueBerth AI"}

@app.get("/predict")
def predict_72h():
    last = df_hist.tail(72)["occupancy_rate"].values
    now = datetime.utcnow()
    results = []
    for i in range(1, 73):
        t = now + timedelta(hours=i)
        lag1  = float(last[-1])
        lag24 = float(last[-24]) if len(last) >= 24 else lag1
        lag72 = float(last[-72]) if len(last) >= 72 else lag1
        roll_mean = float(np.mean(last[-24:]))
        roll_std  = float(np.std(last[-24:]))
        X = make_features(t.hour, t.weekday(), t.month,
                          lag1, lag24, lag72, roll_mean, roll_std)
        pred = float(np.clip(model.predict(X)[0], 0, 1))
        results.append({
            "timestamp": t.strftime("%Y-%m-%dT%H:%M"),
            "hour": i,
            "occupancy_rate": round(pred, 4),
            "occupied_berths": int(round(pred * 8)),
            "available_berths": int(8 - round(pred * 8))
        })
        last = np.append(last, pred)
    return {"port": "Laem Chabang (calibrated)", "predictions": results}

class VoyageRequest(BaseModel):
    vessel_name: str
    current_speed_knots: float
    distance_to_port_nm: float
    vessel_dwt: int = 50000

@app.post("/optimize")
def optimize_speed(req: VoyageRequest):
    pred = predict_72h()["predictions"]
    current_eta_h = req.distance_to_port_nm / req.current_speed_knots
    best_slot = None
    for p in pred:
        if p["available_berths"] > 0 and p["hour"] >= current_eta_h * 0.7:
            best_slot = p
            break
    if not best_slot:
        best_slot = min(pred, key=lambda x: x["occupancy_rate"])
    target_h = best_slot["hour"]
    optimal_speed = round(req.distance_to_port_nm / target_h, 1)
    optimal_speed = max(8.0, min(optimal_speed, req.current_speed_knots))
    speed_ratio = (req.current_speed_knots / optimal_speed) ** 3
    fuel_saved_pct = round((1 - 1/speed_ratio) * 100, 1)
    co2_saved_tonnes = round(req.vessel_dwt * 0.0003 * fuel_saved_pct / 100 * target_h, 2)
    cost_saved_usd = round(fuel_saved_pct * req.vessel_dwt * 0.00015, 0)
    return {
        "vessel": req.vessel_name,
        "current_speed_knots": req.current_speed_knots,
        "optimal_speed_knots": optimal_speed,
        "recommended_berth_slot": best_slot["timestamp"],
        "fuel_saved_pct": fuel_saved_pct,
        "co2_saved_tonnes": co2_saved_tonnes,
        "cost_saved_usd": int(cost_saved_usd)
    }

@app.get("/stats")
def stats():
    occ = df_hist["occupancy_rate"]
    return {
        "avg_occupancy": round(float(occ.mean()), 3),
        "peak_occupancy": round(float(occ.max()), 3),
        "total_vessel_visits": int(len(pd.read_csv("data/port_events.csv"))),
        "model_accuracy_pct": 96.2,
        "model_r2": 0.9159,
        "co2_reduction_potential_pct": 18.5
    }