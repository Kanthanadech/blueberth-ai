import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

print("Loading data...")
df = pd.read_csv('data/berth_occupancy_hourly.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# feature engineering
df['lag_1h']  = df['occupancy_rate'].shift(1)
df['lag_24h'] = df['occupancy_rate'].shift(24)
df['lag_72h'] = df['occupancy_rate'].shift(72)
df['rolling_24h_mean'] = df['occupancy_rate'].shift(1).rolling(24).mean()
df['rolling_24h_std']  = df['occupancy_rate'].shift(1).rolling(24).std()
df = df.dropna()

FEATURES = ['hour_of_day','day_of_week','month',
            'lag_1h','lag_24h','lag_72h',
            'rolling_24h_mean','rolling_24h_std']
TARGET = 'occupancy_rate'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

print("Training model...")
model = GradientBoostingRegressor(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
acc  = (1 - mae) * 100

print(f"\n✅ Model trained!")
print(f"MAE:      {mae:.4f}")
print(f"R²:       {r2:.4f}")
print(f"Accuracy: {acc:.1f}%")

# feature importance
fi = pd.Series(model.feature_importances_, index=FEATURES)
print(f"\nTop features:\n{fi.sort_values(ascending=False).head(5)}")

# save model
with open('models/berth_model.pkl','wb') as f:
    pickle.dump(model, f)
with open('models/feature_names.pkl','wb') as f:
    pickle.dump(FEATURES, f)

print("\n✅ Model saved to models/berth_model.pkl")

# save test results for backtest proof
results = X_test.copy()
results['actual'] = y_test.values
results['predicted'] = y_pred
results.to_csv('data/backtest_results.csv', index=False)
print("✅ Backtest saved to data/backtest_results.csv")