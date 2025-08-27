import argparse
import os
import sys
import time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- deps ---------- erroe checkinggg
try:
    from pytrends.request import TrendReq
except Exception:
    sys.stderr.write("pytrends not installed. Do: pip install pytrends\n")
    raise


def parse_args():
    p = argparse.ArgumentParser(description="Google Trends Report — Final")
    p.add_argument(
        "--kw", nargs="+",
        default=["NIL", "athlete branding", "athlete website"],
        help="Keywords/phrases to track (space-separated)"
    )
    p.add_argument("--geo", type=str, default="US", help="Geo (e.g., 'US', 'GB', '' for worldwide)")
    p.add_argument(
        "--timeframe", type=str, default="today 12-m",
        help="e.g., 'today 12-m', 'today 5-y', or 'YYYY-MM-DD YYYY-MM-DD'"
    )
    p.add_argument("--regions", type=str, default="REGION",
                   choices=["REGION", "COUNTRY", "CITY"],
                   help="Resolution for interest_by_region")
    p.add_argument("--sleep", type=float, default=1.5, help="Seconds to sleep between pytrends calls")
    p.add_argument("--hl", type=str, default="en-US", help="Host language")
    p.add_argument("--tz", type=int, default=-240, help="Timezone offset in minutes")
    p.add_argument("--gprop", type=str, default="", choices=["", "images", "news", "youtube", "froogle"],
                   help="Google property filter; empty = web search")
    return p.parse_args()

# --------- utils ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in s)

def line_chart(df: pd.DataFrame, out_path: str, title: str, ylabel: str):
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        if col == "date":
            continue
        plt.plot(df["date"], df[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def bar_chart(series: pd.Series, out_path: str, title: str, xlabel: str):
    plt.figure(figsize=(10, 6))
    series.sort_values(ascending=True).plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def compute_metrics(ts_df: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    value_cols = [c for c in ts_df.columns if c != "date"]
    for kw in value_cols:
        s = ts_df[kw].astype(float)
        avg_interest = s.mean()
        peak_interest = s.max()
        peak_date = ts_df.loc[s.idxmax(), "date"] if not s.dropna().empty else ""

        last4 = s.tail(4).mean() if len(s) >= 4 else np.nan
        prev4 = s.tail(8).head(4).mean() if len(s) >= 8 else np.nan
        abs_mom = (last4 - prev4) if (pd.notna(last4) and pd.notna(prev4)) else np.nan
        pct_mom = (abs_mom / prev4 * 100.0) if (pd.notna(abs_mom) and prev4 not in [0, np.nan]) else np.nan

        metrics.append({
            "keyword": kw,
            "avg_interest": round(float(avg_interest), 2) if pd.notna(avg_interest) else np.nan,
            "peak_interest": float(peak_interest) if pd.notna(peak_interest) else np.nan,
            "peak_date": pd.to_datetime(peak_date).date() if str(peak_date) != "" else "",
            "last_4w_avg": round(float(last4), 2) if pd.notna(last4) else np.nan,
            "prev_4w_avg": round(float(prev4), 2) if pd.notna(prev4) else np.nan,
            "abs_momentum": round(float(abs_mom), 2) if pd.notna(abs_mom) else np.nan,
            "pct_momentum": round(float(pct_mom), 2) if pd.notna(pct_mom) else np.nan
        })
    return pd.DataFrame(metrics)

# --------- extra visuals ----------
def make_dashboard(ts_df: pd.DataFrame, rq_top_df: pd.DataFrame, out_png: str):
    """Spacious 2×2 dashboard image."""
    if ts_df is None or ts_df.empty:
        return

    kws = [c for c in ts_df.columns if c != "date"]
    avg_interest = {kw: float(ts_df[kw].mean()) for kw in kws}
    heat = ts_df[kws].fillna(0).astype(float).values

    # top related queries (combined across keywords)
    top_bar_index, top_bar_vals = [], []
    if rq_top_df is not None and not rq_top_df.empty:
        agg = rq_top_df.groupby("query")["value"].sum().sort_values(ascending=False).head(10)
        top_bar_index = list(agg.index)
        top_bar_vals = list(agg.values)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.7, wspace=0.55)  # <-- more spacing

    # (1) Time series
    ax1 = fig.add_subplot(gs[0, 0])
    for kw in kws:
        ax1.plot(ts_df["date"], ts_df[kw], label=kw)
    ax1.set_title("Google Search Interest Over Time")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Relative interest (0–100)")
    ax1.legend(loc="upper left", fontsize=8)

    # (2) Share of Search pie
    ax2 = fig.add_subplot(gs[0, 1])
    vals = [max(v, 0.0) for v in avg_interest.values()]
    labels = list(avg_interest.keys())
    if sum(vals) == 0:
        vals = [1 for _ in vals]
    ax2.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Share of Search (avg interest)")

    # (3) Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(heat, aspect="auto", interpolation="nearest")
    ax3.set_title("Weekly Interest Heatmap")
    ax3.set_xlabel("Keywords"); ax3.set_ylabel("Weeks")
    ax3.set_xticks(range(len(kws))); ax3.set_xticklabels(kws, rotation=30, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # (4) Top related queries bar
    ax4 = fig.add_subplot(gs[1, 1])
    if top_bar_index:
        ax4.barh(range(len(top_bar_index))[::-1], top_bar_vals[::-1])
        ax4.set_yticks(range(len(top_bar_index))[::-1])
        ax4.set_yticklabels(top_bar_index[::-1], fontsize=8)
        ax4.set_title("Top Related Queries (combined)")
        ax4.set_xlabel("Relevance (Google Trends 'value')")
    else:
        ax4.text(0.5, 0.5, "No related queries available", ha="center", va="center")
        ax4.set_axis_off()

    fig.suptitle("Media Influence — Google Trends Dashboard", y=0.98, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def save_separate_charts(ts_df: pd.DataFrame, rq_top_df: pd.DataFrame, charts_dir: str):
    """Also save each panel as its own image (clean for slides)."""
    kws = [c for c in ts_df.columns if c != "date"]
    # 1) Time series
    line_chart(ts_df, os.path.join(charts_dir, "search_interest.png"),
               "Google Search Interest Over Time", "Relative interest (0–100)")

    # 2) Share of Search pie
    plt.figure(figsize=(7, 7))
    avg_interest = [float(ts_df[k].mean()) for k in kws]
    labels = kws
    vals = [max(v, 0.0) for v in avg_interest]
    if sum(vals) == 0:
        vals = [1 for _ in vals]
    plt.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Share of Search (avg interest)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "share_of_search.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Heatmap
    plt.figure(figsize=(9, 7))
    heat = ts_df[kws].fillna(0).astype(float).values
    im = plt.imshow(heat, aspect="auto", interpolation="nearest")
    plt.title("Weekly Interest Heatmap")
    plt.xlabel("Keywords"); plt.ylabel("Weeks")
    plt.xticks(range(len(kws)), kws, rotation=30, ha="right", fontsize=8)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Top related queries bar
    plt.figure(figsize=(9, 7))
    if rq_top_df is not None and not rq_top_df.empty:
        agg = rq_top_df.groupby("query")["value"].sum().sort_values(ascending=False).head(10)
        agg.plot(kind="barh")
        plt.title("Top Related Queries (combined)")
        plt.xlabel("Relevance (Google Trends 'value')")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No related queries available", ha="center", va="center")
        plt.axis("off")
    plt.savefig(os.path.join(charts_dir, "related_queries.png"), dpi=200, bbox_inches="tight")
    plt.close()

def plot_engagement(ts_df: pd.DataFrame, out_path: str):
    """
    Fan engagement proxy derived from Google Trends:
      - Total engagement = sum of keyword interest
      - Stacked area by keyword
    """
    if ts_df is None or ts_df.empty:
        return
    kws = [c for c in ts_df.columns if c != "date"]
    df = ts_df.copy()
    df["total_engagement"] = df[kws].sum(axis=1)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(df["date"], df["total_engagement"], linewidth=2)
    axs[0].set_title("Total Fan Engagement (Google Trends proxy)")
    axs[0].set_ylabel("Relative engagement")

    axs[1].stackplot(df["date"], [df[k] for k in kws], labels=kws, alpha=0.9)
    axs[1].set_title("Fan Engagement by Keyword")
    axs[1].set_ylabel("Relative engagement")
    axs[1].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# --------- core ----------
def main():
    args = parse_args()

    ts_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("trends_output", ts_stamp)
    charts_dir = os.path.join(base_dir, "charts")
    ensure_dir(base_dir); ensure_dir(charts_dir)

    print(f"🔎 Fetching Google Trends for: {args.kw} | geo={args.geo} | timeframe={args.timeframe} | gprop={args.gprop}")

    # Avoid requests_args timeout collision; pass timeout directly.
    pytrends = TrendReq(hl=args.hl, tz=args.tz, timeout=(10, 30))

    # ---- Interest Over Time with fallback ----
    series_list: List[pd.DataFrame] = []
    missing = []
    for kw in args.kw:
        # Try provided timeframe, then 5y, then 12m (as a last try)
        for tf in [args.timeframe, "today 5-y", "today 12-m"]:
            try:
                pytrends.build_payload([kw], timeframe=tf, geo=args.geo, gprop=args.gprop)
                iot = pytrends.interest_over_time()
                time.sleep(args.sleep)
                if iot is not None and not iot.empty:
                    iot = iot.reset_index().drop(columns=[c for c in ["isPartial"] if c in iot.columns])
                    iot.rename(columns={kw: kw, "date": "date"}, inplace=True)
                    series_list.append(iot[["date", kw]])
                    if tf != args.timeframe:
                        print(f"ℹ️ Using fallback timeframe '{tf}' for '{kw}'.")
                    break
            except Exception as e:
                print(f"❌ Error for '{kw}' @ {tf}: {e}")
        else:
            print(f"⚠️ No time series for '{kw}'. Skipping.")
            missing.append(kw)

    if not series_list:
        print("No time series retrieved. Exiting.")
        sys.exit(1)

    # Merge wide
    ts_df = series_list[0]
    for df2 in series_list[1:]:
        ts_df = ts_df.merge(df2, on="date", how="outer")
    ts_df = ts_df.sort_values("date").reset_index(drop=True)
    ts_df.to_csv(os.path.join(base_dir, "trends_timeseries.csv"), index=False)

    # Canonical line chart
    line_chart(ts_df, os.path.join(charts_dir, "interest_over_time.png"),
               "Interest Over Time (Google Trends)", "Relative interest (0–100)")

    # ---- Regions ----
    regions_rows = []
    for kw in [c for c in ts_df.columns if c != "date"]:
        try:
            pytrends.build_payload([kw], timeframe=args.timeframe, geo=args.geo, gprop=args.gprop)
            reg = pytrends.interest_by_region(resolution=args.regions, inc_low_vol=True, inc_geo_code=True)
            time.sleep(args.sleep)
            if reg is not None and not reg.empty:
                reg = reg.reset_index().rename(columns={kw: "interest", "geoName": "region"})
                reg["keyword"] = kw
                regions_rows.append(reg[["region", "geoCode", "keyword", "interest"]])
                # per-keyword chart
                topk = reg.sort_values("interest", ascending=False).head(15)
                ser = pd.Series(topk["interest"].values, index=topk["region"].values)
                bar_chart(ser,
                          os.path.join(charts_dir, f"top_regions_{safe_filename(kw)}.png"),
                          f"Top Regions — {kw}", "Relative interest")
            else:
                print(f"⚠️ No regional data for '{kw}'.")
        except Exception as e:
            print(f"❌ Regions error for '{kw}': {e}")

    if regions_rows:
        pd.concat(regions_rows, ignore_index=True).to_csv(os.path.join(base_dir, "trends_regions.csv"), index=False)

    # ---- Related Queries ----
    rq_top_rows, rq_rising_rows = [], []
    for kw in [c for c in ts_df.columns if c != "date"]:
        try:
            pytrends.build_payload([kw], timeframe=args.timeframe, geo=args.geo, gprop=args.gprop)
            rqs = pytrends.related_queries()
            time.sleep(args.sleep)
            if rqs and rqs.get(kw):
                top_df = rqs[kw].get("top")
                rising_df = rqs[kw].get("rising")
                if isinstance(top_df, pd.DataFrame) and not top_df.empty:
                    t = top_df.copy(); t["keyword"] = kw
                    rq_top_rows.append(t[["keyword", "query", "value"]])
                if isinstance(rising_df, pd.DataFrame) and not rising_df.empty:
                    r = rising_df.copy(); r["keyword"] = kw
                    rq_rising_rows.append(r[["keyword", "query", "value"]])
        except Exception as e:
            print(f"❌ Related queries error for '{kw}': {e}")

    rq_top_df = pd.concat(rq_top_rows, ignore_index=True) if rq_top_rows else pd.DataFrame(columns=["keyword", "query", "value"])
    rq_rising_df = pd.concat(rq_rising_rows, ignore_index=True) if rq_rising_rows else pd.DataFrame(columns=["keyword", "query", "value"])
    if not rq_top_df.empty:
        rq_top_df.to_csv(os.path.join(base_dir, "related_queries_top.csv"), index=False)
    if not rq_rising_df.empty:
        rq_rising_df.to_csv(os.path.join(base_dir, "related_queries_rising.csv"), index=False)

    # ---- Metrics ----
    metrics_df = compute_metrics(ts_df)
    metrics_df.to_csv(os.path.join(base_dir, "metrics_summary.csv"), index=False)

    # ---- Dashboard & separate panels ----
    make_dashboard(ts_df, rq_top_df, os.path.join(charts_dir, "dashboard.png"))
    save_separate_charts(ts_df, rq_top_df, charts_dir)

    # ---- Fan Engagement (proxy from trends) ----
    plot_engagement(ts_df, os.path.join(charts_dir, "fan_engagement.png"))

    print(f"\n✅ DONE! Files saved under: {base_dir}")
    print("   - trends_timeseries.csv, trends_regions.csv, metrics_summary.csv")
    print("   - related_queries_top.csv, related_queries_rising.csv (if available)")
    print("   - charts/interest_over_time.png, charts/top_regions_<keyword>.png")
    print("   - charts/dashboard.png and individual charts: search_interest.png, share_of_search.png, heatmap.png, related_queries.png")
    print("   - charts/fan_engagement.png")
    if missing:
        print(f"   - Insufficient data for: {missing}")

if __name__ == "__main__":
    main()
