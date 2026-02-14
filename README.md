Urban Flood Susceptibility Mapping in Port Harcourt and Warri
Using GIS and Machine Learning (Synthetic Data)
📌 Overview

This project models urban flood susceptibility in Port Harcourt and Warri, Nigeria, using GIS-based spatial factors and machine learning. Realistic synthetic data are used to demonstrate flood risk prediction and mapping workflows.

🎯 Objectives

Simulate urban flood conditioning factors

Train an ML model for flood susceptibility classification

Produce GIS-ready flood risk outputs

Support urban planning and flood risk assessment research

🗂️ Project Structure
Urban-Flood-Susceptibility-Mapping/
│
├── data/
│   └── urban_flood_susceptibility_dataset.xlsx
│
├── scripts/
│   └── urban_flood_susceptibility_portharcourt_warri_ml.py
│
├── outputs/
│   ├── flood_susceptibility_map.tif
│   └── flood_zones.shp
│
├── README.md
└── requirements.txt

📊 Dataset Description

Synthetic dataset includes:

Rainfall (mm)

Elevation (m)

Slope (degrees)

Drainage density

Impervious surface (%)

Distance to river (m)

Flood risk class (Low / High)

🤖 Methodology

Generate realistic synthetic GIS variables

Train a Random Forest classifier

Predict flood susceptibility

Export results as GeoTIFF and Shapefiles

🛠️ Technologies Used

Python

NumPy, Pandas

Scikit-learn

Rasterio, GeoPandas

Matplotlib

🚀 How to Run
pip install -r requirements.txt
python scripts/urban_flood_susceptibility_portharcourt_warri_ml.py

🗺️ Outputs

Flood susceptibility raster map (GeoTIFF)

Flood risk zones (Shapefile)

⚠️ Disclaimer

This project uses synthetic data for academic and demonstration purposes only. Results should not be used for real-world flood management decisions.

📄 License

MIT License

👤 Author

AGBOZU EBINGIYE NELVIN
LinkedIn: *https://www.linkedin.com/in/agbozu-ebi/
