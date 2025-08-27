# Athlete & Fan Engagement Pipelines  

This repo contains **two pipelines** for analyzing fan engagement data:  
1. **Google Trends (google_trends.py / trends_scraper.py)** → Pulls search interest, regions, related queries, and generates CSVs and charts.  
2. **Twitter Analytics (x_scraper.py)** → Simulates and analyzes NIL/athlete branding engagement on Twitter, producing dashboards, demographics, and media correlation insights.  

---

## 🧭 Overview  

- **google_trends.py** → CLI-based scraper with args (`--kw`, `--geo`, `--timeframe`, etc.). Saves:  
  - `trends_timeseries.csv` (interest over time)  
  - `trends_regions.csv` (regional breakdown)  
  - `related_queries_top.csv`, `related_queries_rising.csv`  
  - `metrics_summary.csv` (average, peak, momentum)  
  - Charts: line, bar, pie, heatmap, dashboards  

- **trends_scraper.py** → Config-based version with static keywords/timeframe. Generates:  
  - Interest over time plots  
  - Regional interest bar charts  
  - Keyword correlation heatmap (from related queries)  
  - CSV exports with timestamped names  

- **x_scraper.py** → End-to-end simulated analytics pipeline for NIL/branding:  
  - Weekly engagement data for 1 year  
  - Keyword tracking (NIL, athlete branding, sponsorship, endorsements)  
  - Media outlet mentions & correlations (ESPN, Fox Sports, Bleacher Report, etc.)  
  - Demographic splits (race, age, gender)  
  - Viral tweet generator + word cloud  
  - Dashboards: keyword performance, media influence, demographics, viral content  

---

## 🛠️ Tech Stack  

- **Python 3.10+**  
- `pytrends`, `pandas`, `numpy`  
- Visualization: `matplotlib`, `seaborn`, `wordcloud`  
- CLI: `argparse` (for `google_trends.py`)  

---

## ▶️ Running the Pipelines  

### Google Trends (CLI)  
```bash
python google_trends.py --kw NIL "athlete branding" --geo US --timeframe "today 12-m"
```  
Outputs will be saved under `trends_output/<timestamp>/` with CSVs + charts.  

### Google Trends (Config-based)  
```bash
python trends_scraper.py
```  
Uses predefined `KEYWORDS` in script and saves results under `data/analytics/google_trends/`.  

### Twitter Analytics  
```bash
python x_scraper.py
```  
Saves:  
- `twitter_analytics_full.csv`  
- `analytics_summary.csv`  
- `keyword_dashboard.png`  
- `media_influence.png`  
- `demographic_insights.png`  
- `viral_content_wordcloud.png`, `viral_impact.png` (if viral tweets detected)  

---

## 📊 Example Outputs  

- **CSV** → `date`, `keyword`, `interest` (Google), or full engagement breakdown (Twitter).  
- **Visuals** → dashboards for trends, keyword volume, correlations, demographics, viral content.  

---

## 📌 Notes  

- Google Trends uses fallbacks (12m, 5y) if timeframe fails.  
- Regional data and related queries may be missing for niche terms.  
- Twitter pipeline is a **synthetic simulation** (no live API scraping), built for analytics/visual storytelling.  

---

## 🚀 Next Steps  

- [ ] Connect Twitter/X API (if credentials available)  
- [ ] Integrate Postgres for storing historical runs  
- [ ] Automate daily pipeline with cron/Task Scheduler  
- [ ] Enhance dashboards (Streamlit/Flask)  

---

## 📝 License  

This project is for **educational and portfolio use**. Ensure compliance with platform ToS if adapting for production.  
