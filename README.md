# Bengaluru Road Accident – Early Warning System

A neural-network-based early warning system that uses seven years (2018–2024) of Bengaluru Traffic Police station-wise accident records to identify which traffic police station jurisdictions are likely to be **high-risk next year**, and visualises the result on an interactive map.

> Mini project for **Neural Networks & Deep Learning (NNDL)**.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Methodology](#methodology)
- [Interactive Map](#interactive-map)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)

---

## Overview

Road accidents in Bengaluru are reported by Traffic Police Stations, but the raw data is scattered across multiple yearly files with inconsistent formats and station spellings. This project:

1. **Consolidates** the 2018–2024 station-wise accident data into one master dataset.
2. **Analyses** accident trends, severity, and hotspots across stations.
3. **Scores** each station with a composite Hotspot Score (frequency, severity, consistency, trend).
4. **Predicts** with a neural network whether a station will fall in the city's **top-10 most accident-prone stations in the following year**.
5. **Maps** the predicted risk (HIGH / MEDIUM / LOW) with per-station explanations.

---

## Key Results

Evaluated on a held-out test set (features from 2023 → predicting 2024 high-risk status, 34 stations):

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 85.29% |
| Precision | 58.33% |
| Recall    | **100%** |
| F1 Score  | 73.68% |
| ROC-AUC   | 0.9841 |

**Confusion matrix**

|                   | Predicted Low | Predicted High |
|-------------------|:-------------:|:--------------:|
| **Actual Low**    | 22            | 5              |
| **Actual High**   | 0             | 7              |

The model caught every actual high-risk station (recall = 100%) at the cost of 5 false alarms, a reasonable trade-off for an early-warning use case where missing a dangerous location is worse than a false warning.

**Predicted risk distribution (34 stations):** 8 HIGH · 4 MEDIUM · 22 LOW

**Top predicted high-risk stations:** Yelahanka, Kamakshipalya, Kengeri, Whitefield, Peenya, Chikkajala, Electronic City, Hebbal.

---

## Repository Structure

```
.
├── README.md
└── NNDL MINI PROJECT/
    ├── Mini_Project_NNDL_Code.ipynb                 # Full pipeline: cleaning → EDA → scoring → NN → map
    ├── master_bengaluru_accidents_2018_2024.csv     # Consolidated master dataset
    ├── bengaluru_accident_early_warning_map.html    # Final interactive risk map (open in browser)
    └── Datasets/
        ├── Bengaluru Urban Traffic Police Stations Map.kml   # Station geo-locations
        └── Accident/
            ├── BTP - Station-wise accidents & fatalities 2018 to 2020.csv
            ├── BTP - Station-wise accidents & fatalities 2020 to 2022.csv
            ├── BTP - Station-wise accidents & fatalities 2023.csv
            └── Station-wise Total Crashes and Fatal Crashes 2024.csv
```

---

## Data

**Source:** Bengaluru Traffic Police (BTP) station-wise accident and fatality statistics, plus a KML file of traffic police station locations.

**Master dataset columns** (`master_bengaluru_accidents_2018_2024.csv`):

| Column | Description |
|--------|-------------|
| `Zone` | Traffic zone (East, West, North, South) |
| `Sub-division` | Sub-division (available for 2023–2024 records) |
| `Station` | Traffic police station |
| `Year` | 2018–2024 |
| `Fatal` | Number of fatal crashes |
| `Killed` | Number of people killed |
| `Non-Fatal` | Number of non-fatal crashes |
| `Injured` | Number of people injured |
| `Total` | Total crashes |
| `Fatal_Rate` | Fatal crashes ÷ total crashes |
| `NonFatal_Rate` | Non-fatal crashes ÷ total crashes |

**Data cleaning highlights**

- Merged four differently-structured yearly files into a single long-format table.
- Removed aggregate rows (`East Total`, `Grand Total`, etc.) that would otherwise double-count.
- Standardised inconsistent station spellings (e.g. `Yalahanka → Yelahanka`, `K.R.Pura → K.R.Puram`, `White Field → Whitefield`, `H.Grounds → High Grounds`, `Int. Aiport → International Airport`) so stations match across years and against the KML.
- Fuzzy matching (`rapidfuzz`) used to reconcile accident-data station names with KML names.

---

## Methodology

The notebook is organised into phases:

### 1. Exploratory Data Analysis
Yearly accident trends, fatal vs non-fatal comparison, top stations by total and fatal crashes, distribution of crashes across stations, and a station × year heatmap.

### 2. Accident Hotspot Score
A weighted composite score (min-max scaled, 0–100) per station:

```
Hotspot Score = 0.35 × Frequency   (total accidents)
              + 0.25 × Severity    (fatal-crash rate)
              + 0.20 × Consistency (share of years in the yearly top 10)
              + 0.20 × Trend       (recent vs historical average, capped at -100%…+200%)
```

Stations are then bucketed into HIGH / MEDIUM / LOW by score terciles.

### 3. Feature Engineering (for the neural network)
Per-station time-series features, built using only past information:

- Previous-year totals, fatal and non-fatal counts and rates
- Year-over-year change (absolute and %)
- 2-year and 3-year rolling averages (shifted to avoid leakage)
- One-hot encoded station

**Target (`Future_High_Risk`):** 1 if the station ranks in the **top 10 by total accidents in the next year**, else 0.

### 4. Time-based Train / Validation / Test Split
To avoid look-ahead leakage, the split is by year rather than random:

| Split | Rows used |
|-------|-----------|
| Train | Year < 2022 |
| Validation | Year = 2022 |
| Test | Year = 2023 (predicting 2024) |

Preprocessing: median imputation → standard scaling of numeric features → one-hot station encoding (58 input features in total). Class weights are used to handle the imbalance between high-risk and non-high-risk stations.

### 5. Neural Network

A feed-forward network built with TensorFlow/Keras:

```
Input (58)
 → Dense(64, ReLU) → Dropout(0.30)
 → Dense(32, ReLU) → Dropout(0.20)
 → Dense(16, ReLU)
 → Dense(1, Sigmoid)
```

- **Loss:** binary cross-entropy | **Optimizer:** Adam (lr = 0.001)
- **Training:** up to 150 epochs, batch size 8, early stopping on validation loss (patience 15), balanced class weights
- **Seeds fixed** to 42 for reproducibility

### 6. Early-Warning Output
Predicted probability is converted to a risk level:

| Probability | Risk Level |
|-------------|-----------|
| ≥ 70% | 🔴 HIGH |
| 40% – 70% | 🟠 MEDIUM |
| < 40% | 🟢 LOW |

Each station also gets human-readable **risk indicators** (e.g. *"High number of fatal crashes"*, *"Strong increase in accidents"*, *"High fatal-crash rate"*) generated from rule-based thresholds, so the warning is explainable and not just a number.

Permutation importance (ROC-AUC drop over 20 shuffles per feature) is also computed to see which features drive the model.

---

## Interactive Map

`NNDL MINI PROJECT/bengaluru_accident_early_warning_map.html` is a self-contained [Folium](https://python-visualization.github.io/folium/) map. Just open it in any browser.

- Each marker is a traffic police station, coloured by predicted risk (red / orange / green).
- Hover for a quick tooltip; click for a popup with predicted risk %, total / fatal / non-fatal crashes, fatal rate, year-over-year change, 2-year average, and the reasons the station is flagged.
- **28 of 34** stations are shown, those with matching coordinates in the KML file.

---

## Getting Started

### Option A – View the results
Open `NNDL MINI PROJECT/bengaluru_accident_early_warning_map.html` in a browser.

### Option B – Run the notebook

The notebook was developed in **Google Colab**.

1. Upload `Mini_Project_NNDL_Code.ipynb` to Colab (or open it from GitHub).
2. Upload `master_bengaluru_accidents_2018_2024.csv` when prompted (first cell).
3. When the map section prompts for the KML, upload the station KML file. The notebook expects it under the name `bengaluru_traffic_stations.kml`, so **rename** `Bengaluru Urban Traffic Police Stations Map.kml` accordingly.
4. Run all cells in order.

To run locally instead, install the dependencies, remove the `google.colab` upload/download cells, and point the file paths at the repo files:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow geopandas fiona pyogrio folium rapidfuzz
jupyter notebook "NNDL MINI PROJECT/Mini_Project_NNDL_Code.ipynb"
```

---

## Limitations

- **Very small dataset.** There are only 34 stations in the final modelling table and 34 test rows (7 positives), so the metrics (especially the near-perfect ROC-AUC) have high variance and should be read as indicative, not conclusive.
- **Data-source discontinuity.** Reported totals jump sharply between 2022 (7,646) and 2023 (14,922). This likely reflects differences in reporting or station coverage between source files rather than a real doubling of accidents. This can influence the trend and rank-based features.
- **Station-level, not road-level.** Risk is predicted per police-station jurisdiction, not for specific roads or junctions. The marker location is the station, not the actual black spot.
- **Precision of 58%.** The model over-flags some stations (5 false positives), which is intentional for early warning but means HIGH/MEDIUM alerts should be treated as prioritisation hints.
- **Counts only.** The model has no information on traffic volume, road type, weather, time of day, or enforcement, so it cannot distinguish "dangerous" from "busy".
- **Partial map coverage.** 6 of the 34 stations could not be matched to KML coordinates and are not plotted.

---

## Future Work

- Add exposure data (traffic volume, vehicle registrations, population) and road-level / black-spot data.
- Incorporate temporal granularity (monthly / hourly) and weather or event features.
- Compare against baselines (logistic regression, random forest, gradient boosting) and use cross-validation across years.
- Cover the remaining stations on the map and add a dashboard with filters by zone and year.

---

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · TensorFlow / Keras · GeoPandas · Folium · RapidFuzz · Google Colab
