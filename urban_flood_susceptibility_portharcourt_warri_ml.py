
# ============================================================
# Urban Flood Susceptibility Mapping in Port Harcourt & Warri
# Using GIS and Machine Learning (Synthetic Data)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

def generate_urban_flood_data(size=80, city="Port Harcourt"):
    np.random.seed(42 if city == "Port Harcourt" else 24)

    elevation = np.clip(np.random.normal(12, 3, (size, size)), 2, 25)
    slope = np.clip(np.random.normal(3.5, 1.2, (size, size)), 0.5, 12)
    distance_river = np.clip(np.random.exponential(500, (size, size)), 50, 2000)
    land_use = np.random.choice([0, 1], size=(size, size), p=[0.35, 0.65])
    drainage_density = np.clip(np.random.normal(2.8, 0.6, (size, size)), 1.0, 4.5)
    rainfall = np.clip(np.random.normal(65, 12, (size, size)), 30, 120)

    flood_risk = (
        (elevation < 8).astype(int) +
        (slope < 2).astype(int) +
        (distance_river < 400).astype(int) +
        land_use +
        (rainfall > 70).astype(int)
    )

    flood_label = (flood_risk >= 3).astype(int)
    return elevation, slope, distance_river, land_use, drainage_density, rainfall, flood_label

def prepare_ml_dataset(data):
    elevation, slope, dist_river, land_use, drainage, rainfall, label = data
    rows = []

    for i in range(elevation.shape[0]):
        for j in range(elevation.shape[1]):
            rows.append([
                elevation[i, j],
                slope[i, j],
                dist_river[i, j],
                land_use[i, j],
                drainage[i, j],
                rainfall[i, j],
                label[i, j]
            ])

    return pd.DataFrame(
        rows,
        columns=[
            "elevation_m",
            "slope_deg",
            "distance_to_river_m",
            "land_use_builtup",
            "drainage_density",
            "rainfall_intensity_mmhr",
            "flood_susceptibility"
        ]
    )

ph_data = generate_urban_flood_data(city="Port Harcourt")
warri_data = generate_urban_flood_data(city="Warri")

df = pd.concat([
    prepare_ml_dataset(ph_data),
    prepare_ml_dataset(warri_data)
], ignore_index=True)

X = df.drop("flood_susceptibility", axis=1)
y = df["flood_susceptibility"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Urban Flood Susceptibility Model Performance")
print(classification_report(
    y_test, y_pred,
    target_names=["Low Susceptibility", "High Susceptibility"]
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def visualize_flood_map(data, city_name):
    elevation, slope, dist_river, land_use, drainage, rainfall, _ = data
    stacked = np.stack([elevation, slope, dist_river, land_use, drainage, rainfall], axis=-1)
    reshaped = stacked.reshape(-1, 6)
    reshaped_scaled = scaler.transform(reshaped)
    flood_map = model.predict(reshaped_scaled).reshape(elevation.shape)

    plt.figure(figsize=(6, 5))
    plt.imshow(flood_map, cmap="RdYlBu_r")
    plt.title(f"Flood Susceptibility Map – {city_name}")
    plt.colorbar(label="Flood Risk")
    plt.axis("off")
    plt.show()

visualize_flood_map(ph_data, "Port Harcourt")
visualize_flood_map(warri_data, "Warri")
