# Airbnb Pricing Analysis across US Cities

Interactive dashboard for analyzing Airbnb pricing patterns across five major US cities. Built from raw CSVs to a deployed Streamlit app — shows the full data pipeline, not just pretty charts.

**Live demo:** [https://airbnb-pricing-analysis--NikitaGolyshiev.replit.app](https://airbnb-pricing-analysis--NikitaGolyshiev.replit.app)

---

## What this is

I took Inside Airbnb data for Austin, Dallas, Nashville, NYC, and Seattle. Total of 77K+ listings and 28M calendar records. Went through the whole cycle: raw data → cleaning → EDA → predictive model → seasonality analysis → interactive dashboard.

The idea was to build something that actually answers business questions: How seasonal is each market? What separates hobbyist hosts from professionals?

## What's inside the dashboard

**Overview** — market KPIs, price distributions by city and room type, basic stats.

**Predictive** — linear regression model that estimates nightly price from rating, room type, location, and availability. Includes predicted vs actual scatter and feature importance.

**Hosts** — segmented hosts into four tiers by listing count: Hobbyist (1), Small Host (2-5), Manager (6-15), Professional (16+). Compared their pricing strategy and guest satisfaction.

**Geography** — interactive price heatmaps. You can literally see which neighborhoods command premiums and which are budget zones.

**Seasonality** — monthly price and occupancy trends built from calendar data, not just listing snapshots.

## Key findings

- **Nashville** has the highest median price ($198/night) and the strongest seasonality — price swings of ±$120 depending on month. Music City lives up to the hype.
- **NYC** dominates in volume (42K listings) and occupancy (62%), but ratings are the lowest across all five cities. Oversupplied market, race to the bottom on quality.
- **Austin** shows the widest price dispersion — you get everything from $50 shared rooms to $800+ luxury properties. Most polarized market.
- **Professional hosts** (16+ listings) price 40% higher than hobbyists, but their review scores are lower. There's a clear price-quality tradeoff.
- **Hotel rooms** average $126 more than entire homes. Counterintuitive, but the data is consistent across all cities.

## Data

- Source: Inside Airbnb (via Kaggle)
- Period: March 2023 — May 2024
- Cities: Austin, Dallas, Nashville, New York City, Seattle
- 77,494 listings + 28,429,579 calendar records

## How to reproduce this project

Since raw CSV files exceed GitHub's 100MB limit, they are not included in this repo. Here's how to get the data and run everything:

### 1. Download the data

Get the datasets from **Inside Airbnb** or **Kaggle**:

- [Inside Airbnb - Get the Data](http://insideairbnb.com/get-the-data.html)
- Or search Kaggle for "Inside Airbnb" — there are compiled datasets for all cities

You need two file types per city:
- `listings.csv` — property listings with prices, ratings, locations
- `calendar.csv` — daily availability and pricing records

### 2. Create the folder structure

Place files like this inside the project folder:

```
data/
├── raw/
│   ├── Austin/
│   │   ├── listings_austin.csv
│   │   └── calendar_austin.csv
│   ├── Dallas/
│   │   ├── listings_dallas.csv
│   │   └── calendar_dallas.csv
│   ├── Nashville/
│   │   ├── listings_nashville.csv
│   │   └── calendar_nashville.csv
│   ├── New York City/
│   │   ├── listings_newyork.csv
│   │   └── calendar_newyork.csv
│   └── Seattle/
│       ├── listings_seattle.csv
│       └── calendar_seattle.csv
└── processed/
    └── (will be created by scripts)
```

### 3. Run the pipeline

```bash
# Step 1: Load and combine data
python 01_data_loading.py

# Step 2: Clean and engineer features
python 02_data_cleaning.py

# Step 3: Exploratory analysis + plots
python 03_eda.py

# Step 4: Predictive model, segmentation, maps
python 04_advanced_analysis.py

# Step 5: Seasonality from calendar data
python 05_temporal_analysis.py

# Step 6: Launch the dashboard
streamlit run app.py
```

Scripts 01–05 generate processed datasets and save plots to `screenshots/`. The dashboard reads from `data/processed/`.

## Run it locally (if you already have the data)

```bash
git clone https://github.com/golyshevventure/airbnb-pricing-analysis.git
cd airbnb-pricing-analysis

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Project structure

```
.
├── app.py                      # Streamlit dashboard
├── 01_data_loading.py          # Load 5 city CSVs
├── 02_data_cleaning.py         # Clean, engineer features
├── 03_eda.py                   # Exploratory analysis + plots
├── 04_advanced_analysis.py     # Regression, segmentation, maps
├── 05_temporal_analysis.py     # Calendar seasonality
├── data/
│   ├── raw/                    # Source files (not in git)
│   └── processed/              # Cleaned datasets
├── screenshots/                # Generated charts
├── README.md
└── requirements.txt
```

## Contact
Nikita Golyshev, financial and data analyst
E-mail: golyshevventure@gmail.com
