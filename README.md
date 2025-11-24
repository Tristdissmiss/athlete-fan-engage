# Athlete & Fan Engagement Pipelines  

This repository contains two data pipelines designed to analyze athlete and NIL (Name, Image, Likeness) engagement trends. The project includes a Google Trends analytics pipeline and a synthetic Twitter/X engagement simulation pipeline. These tools model keyword interest, regional popularity, demographic insights, media influence, and overall engagement patterns.
 
---

## Overview  
This project provides:

1. **google_trends.pipeline** → Extracts search interest, regional activity, related queries, and keyword correlations.
Outputs structured CSVs and analytics visualizations.

2. **Twitter/X Engagement Simulation Pipeline** → Generates synthetic engagement data focused on NIL and athlete branding.
Produces demographics, media influence charts, keyword dashboards, and viral content analysis.

These pipelines can be used for market research, NIL value modeling, athlete branding insights, and sports analytics studies


---
## Pipeline Details 

**Google Trends Pipeline**

Scripts: **google_trends.py**, **trends_scraper.py** 

**google_trends.py(CLI Version)**
Command-line scraping tool for pulling Google Trends data based on custom keywords and regions. 

**Exports**
- `trends_timeseries.csv `

- `trends_regions.csv `

- `related_queries_top.csv`

- `related_queries_rising.csv`

- `metrics_summary.csv`

**Generates:**
- Line graphs
- Regional heatmaps
- Bar Charts
- Keyword correlation plots

`trends_scraper.py`*(Config Version)*
Uses predefined keywords and timeframes inside the script.
Generates timestamped CSV files and complete trend visualizations.  

## Twitter/X Synthetic Engagement Pipeline 
Script: `x_scraper.py` 

This pipeline models weekly engagement activity for NIL, athlete branding, sponsorship keywords, and sports media influence. It is fully synthetic and designed for analytics demonstrations.

**Outputs include:**

`twitter_analytics_full.csv`

`analytics_summary.csv`

`Keyword dashboard charts`

`Media influence visualizations`

`Demographic insights`

`Word clouds`

`Viral impact charts`


## Tech Stack  
Language $ Libraries

- **Python 3.10+**  
- `pytrends`,
- `pandas`,
- `numpy`
- `matplotlib`
- `seaborn`
- `wordcloud`
- `argparse`

Data Formats
- `CSV files`
- `PNG visualizations`
---

## How to run locally

### 1. Clone the Repository  
```bash
git clone <your_repo_url>
cd athlete-fan-engage
```  
Outputs will be saved under `trends_output/<timestamp>/` with CSVs + charts.  

### 2. Install Dependecies
```bash
poip install -r requirements.txt
```

### 3. Run Google Trends (CLI Version)  
```bash
python google_trends.py --kw NIL "athlete branding" --geo US --timeframe "today 12-m"
```  
### Outputs saved under:
``` 
trends_output/<timestamp>/
```

### 4. Run Google Trends(Config Version)
```bash
python trends_scrapper.py
```
Outputs saved under: 
```bash
data/analytics/google_trends/
```
### 5. RUN Twitter/X Simulation 

```bash
python x_scraper.py
```

Generates: 
- CSV datasets
- Summary reports
- Keyword dashboards
- Demographic charts 
- Word clouds and viral simulations

---

## 📊 Example Outputs  

**CSV Outputs**
- Keyword interest over time 
- Regional search activity
- Weekly engagement scores
- Media influence statistics
- Demographic breakdowns
  
**Visuals**
- Line charts
- Bar graphs
- Heatmaps
- Keywords dashboards
- Word clouds


---

##  Notes  

- Google Trends may return partial regional or related-query data for certain keywords.  
- The Twitter/X pipeline uses fully synthetic data for modeling purposes. 
- These pipelines are intended for analytics, visualization, and portfolio demonstration.

---

## Future Development  

- Integrate live Twitter/X API
- Add a Postgres database for historical storage
- Automate pipeline execution with cron or Airflow  \
- Build an interactive dashboard with Streamlit or Flask
- Add player comparison and multi-athlete analysis tools 

---
