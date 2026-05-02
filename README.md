# Airbnb Pricing Analysis across US Cities

Interactive dashboard for analyzing Airbnb pricing patterns across five major US cities. Built from raw CSVs to a deployed Streamlit app — shows the full data pipeline.

**Live demo:**(https://airbnb-pricing-analysis--NikitaGolyshiev.replit.app)

---

## What this is

I took Inside Airbnb data for Austin, Dallas, Nashville, NYC, and Seattle. Total of 77K+ listings and 28M calendar records. Went through the whole cycle: raw data → cleaning → EDA → predictive model → seasonality analysis → interactive dashboard.

The idea was to build something that actually answers business questions: where should you list to charge more? What drives pricing? How seasonal is each market? What separates hobbyist hosts from professionals?

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

## Run it locally

```bash
git clone https://github.com/yourusername/airbnb-pricing-analysis.git
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
Nikita Golyshev
Financial & Data analyst
E-mail: golyshevventure@gmail.com
