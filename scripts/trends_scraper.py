from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates

# Configuration
KEYWORDS = ["athlete branding", "NIL opportunities", "how to make athlete website"]
TIME_RANGE = "today 12-m"   # Last 12 months
GEO = "US"                  # United States
OUTPUT_DIR = Path("data/analytics/google_trends")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_trends_data():
    """Fetch Google Trends data with enhanced error handling."""
    try:
        pytrends = TrendReq(hl="en-US", tz=360, retries=3, backoff_factor=0.5)
        pytrends.build_payload(KEYWORDS, timeframe=TIME_RANGE, geo=GEO)

        over_time = pytrends.interest_over_time()
        by_region = pytrends.interest_by_region(resolution="COUNTRY", inc_low_vol=True, inc_geo_code=False)
        related = pytrends.related_queries()

        # Normalize/clean
        if over_time is None or over_time.empty:
            raise ValueError("interest_over_time returned empty. Try broader terms, different timeframe, or wait and retry.")

        if "isPartial" in over_time.columns:
            over_time = over_time.drop(columns="isPartial")

        # by_region can be empty for niche terms; keep it optional
        if by_region is None:
            by_region = pd.DataFrame()

        # Ensure related has expected structure
        if not isinstance(related, dict):
            related = {}

        return {"over_time": over_time, "by_region": by_region, "related": related}

    except Exception as e:
        print(f"Error fetching Trends data: {e}")
        return None

def generate_visualizations(data):
    """Create publication-ready visualizations with guards."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1) Interest Over Time (Area + Line)
    df = data["over_time"]
    if df is None or df.empty:
        print("Skipping 'Interest Over Time' plot: no data.")
    else:
        plt.figure(figsize=(12, 6))
        for kw in KEYWORDS:
            if kw in df.columns:
                plt.fill_between(df.index, df[kw], alpha=0.3, label=kw)
                plt.plot(df.index, df[kw], linewidth=2)

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.title("Athlete Branding Trends (12 Months)", fontsize=14, pad=20)
        plt.legend(frameon=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        (OUTPUT_DIR / "interest_over_time.png").write_bytes(plt.savefig(fname=None, dpi=300) or b"")  # ensure file write
        plt.savefig(OUTPUT_DIR / "interest_over_time.png", dpi=300)
        plt.close()

    # 2) Regional Interest (Horizontal Bar)
    br = data["by_region"]
    if isinstance(br, pd.DataFrame) and not br.empty:
        # Keep only the keywords we asked for (pytrends may include extra cols sometimes)
        cols = [c for c in KEYWORDS if c in br.columns]
        if cols:
            top10 = br.sort_values(cols[0], ascending=False).head(10)[cols]
            plt.figure(figsize=(10, 6))
            top10.plot(kind="barh", stacked=True, width=0.8)
            plt.title("Top Regions by Interest", fontsize=14)
            plt.xlabel("Relative Interest")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "regional_interest.png", dpi=300)
            plt.close()
        else:
            print("Skipping 'Regional Interest' plot: expected keyword columns not found.")
    else:
        print("Skipping 'Regional Interest' plot: by_region is empty.")

    # 3) Related Queries Heatmap (guarded)
    related = data["related"]
    frames = []
    for kw in KEYWORDS:
        try:
            top_df = related.get(kw, {}).get("top")
            if isinstance(top_df, pd.DataFrame) and not top_df.empty:
                frames.append(top_df.set_index("query")[["value"]].rename(columns={"value": kw}))
        except Exception:
            pass

    if frames:
        merged = pd.concat(frames, axis=1).fillna(0)
        if merged.shape[1] > 1 and merged.sum().sum() > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(merged.corr(), annot=True, cmap="YlOrRd", vmin=0, vmax=1)
            plt.title("Keyword Correlation Heatmap (Related Queries)", fontsize=14)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "keyword_correlation.png", dpi=300)
            plt.close()
        else:
            print("Skipping heatmap: not enough overlap between related queries.")
    else:
        print("Skipping heatmap: no related 'top' queries available.")

def save_data(data):
    """Save all data with timestamps."""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    data["over_time"].to_csv(OUTPUT_DIR / f"interest_over_time_{timestamp}.csv")

    if isinstance(data["by_region"], pd.DataFrame) and not data["by_region"].empty:
        data["by_region"].to_csv(OUTPUT_DIR / f"regional_interest_{timestamp}.csv")

    related = data["related"]
    for kw in KEYWORDS:
        top_df = related.get(kw, {}).get("top")
        if isinstance(top_df, pd.DataFrame) and not top_df.empty:
            top_df.to_csv(OUTPUT_DIR / f"related_queries_top_{kw[:20]}_{timestamp}.csv", index=False)

    print(f"✅ Data saved to {OUTPUT_DIR}")

def print_key_insights(data):
    """Print small, robust metrics that won't crash."""
    df = data["over_time"]
    br = data["by_region"]

    # Most searched term across the period
    most_searched = df.mean().idxmax()

    # Peak interest month for the most-searched term
    peak_date = df[most_searched].idxmax()
    peak_month = peak_date.strftime("%B %Y")

    # Top region for that most-searched term (if available)
    if isinstance(br, pd.DataFrame) and (most_searched in br.columns) and not br[most_searched].empty:
        top_region = br[most_searched].idxmax()
    else:
        top_region = "N/A"

    print("\n📊 Key Insights:")
    print(f"- Most searched term: {most_searched}")
    print(f"- Peak interest month for '{most_searched}': {peak_month}")
    print(f"- Top region for '{most_searched}': {top_region}")

if __name__ == "__main__":
    print("🚀 Google Trends Analyzer")
    print("========================")

    trends_data = get_trends_data()
    if trends_data:
        generate_visualizations(trends_data)
        save_data(trends_data)
        print_key_insights(trends_data)
    else:
        print("❌ Failed to fetch data")
