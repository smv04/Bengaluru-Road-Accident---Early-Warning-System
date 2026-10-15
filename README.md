# Bengaluru Road Accident Early Warning System

A neural-network mini project (NNDL) that scores Bengaluru traffic police stations by historical accident hotspot risk and predicts which stations are likely to be among the city's **top-10 accident-prone stations next year**. Results are shown on an interactive map.

> **Author:** Likhith, M.Sc. Statistics, CHRIST (Deemed to be University), Bengaluru
> **Course:** Neural Networks and Deep Learning (NNDL)

---

## Highlights

- Cleans and standardizes 7 years (2018–2024) of station-level accident data for **60 traffic police stations**.
- Builds a transparent **Hotspot Score (0–100)** from frequency, severity, consistency, and recent trend.
- Trains a **feed-forward neural network (Keras)** to predict whether a station will rank in next year's top 10 by accident count.
- Explains each warning with rule-based **risk indicators** and ranks features using permutation importance.
- Publishes an interactive **Folium / Leaflet map** with risk-coloured markers and popups.

## Repository contents

| File | Description |
|---|---|
| `Mini_Project_NNDL_Code.ipynb` | Full pipeline: cleaning, EDA, hotspot score, feature engineering, neural network, evaluation, map |
| `master_bengaluru_accidents_2018_2024.csv` | Station-wise yearly accident data (fatal, non-fatal, killed, injured, total) |
| `Bengaluru_Urban_Traffic_Police_Stations_Map.kml` | Traffic police station locations (coordinates) |
| `bengaluru_accident_early_warning_map.html` | Final interactive risk map (open in any browser) |
| `requirements.txt` | Python dependencies |

## Data

**`master_bengaluru_accidents_2018_2024.csv`**: 333 rows, one per station per year.

| Column | Meaning |
|---|---|
| `Zone`, `Sub-division`, `Station` | Police zone, sub-division, and station name |
| `Year` | 2018–2024 |
| `Fatal`, `Killed` | Fatal crashes and people killed |
| `Non-Fatal`, `Injured` | Non-fatal crashes and people injured |
| `Total` | Total crashes (`Fatal + Non-Fatal`) |
| `Fatal_Rate`, `NonFatal_Rate` | Share of fatal / non-fatal crashes |

The raw file contains aggregate rows (`Grand Total`) and inconsistent station spellings (for example, `White Field` / `Whitefield`, `K.R.Pura` / `K.R.Puram`). The notebook removes the aggregates and standardises names, leaving **60 stations**.

**`Bengaluru_Urban_Traffic_Police_Stations_Map.kml`**: 43 station locations. Only part of the accident stations could be matched to these coordinates, so the final map shows **28 of the 34** stations in the test set.


## Methodology

1. **Cleaning:** remove aggregate rows, fix station name variants, standardise names against the KML.
2. **EDA:** yearly accident trend, fatal vs non-fatal, top stations, station-year heatmap, consistently top-ranked stations.
3. **Hotspot Score (historical):**
   `Score = 0.35 × Frequency + 0.25 × Severity + 0.20 × Consistency + 0.20 × Trend`
   Each component is min-max scaled. Stations are split into HIGH / MEDIUM / LOW by terciles.
4. **Feature engineering:** previous-year accidents and fatalities, year-on-year change (absolute and %), 2- and 3-year rolling averages, previous fatal and non-fatal rates, and station one-hot encoding.
5. **Target:** `Future_High_Risk = 1` if the station ranks in the **top 10 by total accidents in the next year**.
6. **Time-based split (no shuffling):** train = years before 2022, validation = 2022, test = 2023 (predicting 2024).
7. **Model:** Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(16, ReLU) → Dense(1, sigmoid). Adam (lr 0.001), binary cross-entropy, balanced class weights, early stopping on validation loss (patience 15).
8. **Risk levels from predicted probability:** HIGH ≥ 0.70, MEDIUM ≥ 0.40, otherwise LOW.
9. **Explainability:** permutation importance (drop in ROC-AUC) and rule-based risk indicators per station.

## Results (test set, 34 station-year rows, predicting 2024)

| Metric | Value |
|---|---|
| Accuracy | 85.29% |
| Precision | 58.33% |
| Recall | 100.00% |
| F1 score | 73.68% |
| ROC-AUC | 0.9841 |

Confusion matrix:

| | Predicted low risk | Predicted high risk |
|---|---|---|
| **Actual low risk** | 22 | 5 |
| **Actual high risk** | 0 | 7 |

Predicted risk levels: 8 HIGH, 4 MEDIUM, 22 LOW stations.

**How to read this:** the model caught every station that actually became high risk (recall 100%) at the cost of 5 false alarms (precision 58%). For an early-warning use case that trade-off is reasonable.

## Limitations

- **Small dataset:** 156 modelled rows in total and only 34 in the test set. Metrics such as the 100% recall and 0.98 AUC rest on just 7 positive cases and can change a lot with a different split.
- **Annual granularity only:** no time-of-day, road, weather or traffic-volume data.
- **Rank-based target:** "high risk" means top 10 by crash count, so the label depends on relative, not absolute, risk.
- **Partial map coverage:** some stations have no matching KML coordinates and are missing from the map.
- Intended as an academic demonstration, **not** an operational policing tool.

## How to run

```bash
git clone https://github.com/LikhithGMuthyalu/Bengaluru-Road-Accident---Early-Warning-System.git
cd Bengaluru-Road-Accident---Early-Warning-System
pip install -r requirements.txt
jupyter notebook Mini_Project_NNDL_Code.ipynb
```

Before running locally:

1. The notebook was written for **Google Colab**. Remove or skip the cells using `google.colab` (`files.upload()`, `drive.mount()`, `files.download()`).
2. The notebook loads the KML as `bengaluru_traffic_stations.kml`. Either rename the uploaded KML to that name or change the filename in the notebook.
3. Run the cells top to bottom. They generate the summary CSVs and map HTML files in the working directory.

Or simply open `bengaluru_accident_early_warning_map.html` in a browser to explore the final map (needs internet for the map tiles and libraries).

## Tech stack

Python, pandas, NumPy, scikit-learn, TensorFlow / Keras, Matplotlib, Seaborn, GeoPandas, Folium (Leaflet), RapidFuzz.


