import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from wordcloud import WordCloud
import os

# ========================
# CONFIGURATION
# ========================
OUTPUT_DIR = "twitter_analytics/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date range (1 year weekly)
start_date = datetime(2024, 8, 11)
end_date = datetime(2025, 8, 17)
dates = pd.date_range(start=start_date, end=end_date, freq='W-SUN')

# ========================
# CORE KEYWORDS TRACKING
# ========================
KEYWORDS = {
    "NIL": {
        'color': '#FF6B6B',
        'hashtags': ['#NIL', '#NameImageLikeness', '#NILDeals'],
        'related_terms': ['compensation', 'college athletes', 'monetization']
    },
    "athlete_branding": {
        'color': '#4ECDC4', 
        'hashtags': ['#AthleteBranding', '#PersonalBrand'],
        'related_terms': ['sponsorship', 'endorsements', 'social media']
    },
    "sports_sponsorship": {
        'color': '#FFA07A',
        'hashtags': ['#Sponsorship', '#SportsBiz'],
        'related_terms': ['partnerships', 'brand deals', 'ambassadors']
    },
    "college_athlete": {
        'color': '#A569BD',
        'hashtags': ['#NCAA', '#CollegeSports'],
        'related_terms': ['student athlete', 'recruiting', 'transfer portal']
    },
    "endorsement": {
        'color': '#F7DC6F',
        'hashtags': ['#BrandDeals', '#PaidPartnership'],
        'related_terms': ['influencer', 'sponsored', 'advertising']
    }
}

# ========================
# MEDIA OUTLETS TRACKING
# ========================
MEDIA_OUTLETS = {
    "ESPN": {
        'color': '#D00000',
        'handles': ['@espn', '@ESPNPR']
    },
    "Fox Sports": {
        'color': '#0A2463',
        'handles': ['@FOXSports', '@FS1']
    },
    "Bleacher Report": {
        'color': '#00B2FF', 
        'handles': ['@BleacherReport', '@brgridiron']
    },
    "The Athletic": {
        'color': '#000000',
        'handles': ['@TheAthletic']
    },
    "Sports Illustrated": {
        'color': '#F0C808',
        'handles': ['@SInow']
    }
}

# ========================
# DEMOGRAPHIC PROFILES 
# ========================
DEMOGRAPHICS = {
    'race': {
        'white': {'pct': 0.55, 'color': '#F8F9FA'},
        'black': {'pct': 0.25, 'color': '#212529'},
        'hispanic': {'pct': 0.15, 'color': '#FF7F50'},
        'asian': {'pct': 0.03, 'color': '#FF5733'},
        'other': {'pct': 0.02, 'color': '#6C757D'}
    },
    'age': {
        '13-17': {'pct': 0.10, 'color': '#FFC8DD'},
        '18-24': {'pct': 0.25, 'color': '#FFAFCC'},
        '25-34': {'pct': 0.35, 'color': '#BDE0FE'},
        '35-44': {'pct': 0.15, 'color': '#A2D2FF'},
        '45-54': {'pct': 0.10, 'color': '#CDB4DB'},
        '55+': {'pct': 0.05, 'color': '#FFE5D9'}
    },
    'gender': {
        'male': {'pct': 0.60, 'color': '#0077B6'},
        'female': {'pct': 0.38, 'color': '#FF477E'},
        'nonbinary': {'pct': 0.02, 'color': '#9B5DE5'}
    }
}

# ========================
# DATA STRUCTURE
# ========================
data = {
    # Core metrics
    'date': dates,
    'total_engagement': [0]*len(dates),
    'total_tweets': [0]*len(dates),
    'avg_likes': [0]*len(dates),
    'avg_retweets': [0]*len(dates),
    'avg_replies': [0]*len(dates),
    
    # Keyword metrics
    **{f"{kw}_engagement": [0]*len(dates) for kw in KEYWORDS},
    **{f"{kw}_tweets": [0]*len(dates) for kw in KEYWORDS},
    **{f"{kw}_likes": [0]*len(dates) for kw in KEYWORDS},
    
    # Media metrics
    **{f"{media}_mentions": [0]*len(dates) for media in MEDIA_OUTLETS},
    **{f"{media}_engagement": [0]*len(dates) for media in MEDIA_OUTLETS},
    
    # Demographic metrics
    **{f"{demo}_{group}_pct": [0]*len(dates) 
       for demo in DEMOGRAPHICS for group in DEMOGRAPHICS[demo]},
    
    # Influencer tracking
    'top_influencer': ['']*len(dates),
    'top_athlete': ['']*len(dates),
    'top_hashtag': ['']*len(dates),
    'viral_tweet': ['']*len(dates)
}

# ========================
# SPECIAL EVENTS DATASET
# ========================
special_events = {
    '2025-02-16': {  # Super Bowl
        'keywords': {
            'NIL': {'engagement': 8500, 'tweets': 420},
            'endorsement': {'engagement': 7200, 'tweets': 380}
        },
        'media': {
            'ESPN': {'mentions': 95, 'engagement': 42000},
            'Fox Sports': {'mentions': 60, 'engagement': 38000}
        },
        'demographics': {
            'race': {'black': 0.38, 'hispanic': 0.18},
            'age': {'18-24': 0.40, '25-34': 0.45},
            'gender': {'male': 0.72}
        },
        'influencers': {
            'top_athlete': '@PatrickMahomes',
            'viral_tweet': 'Breaking: Super Bowl ads feature record NIL deals for players'
        }
    },
    '2025-03-30': {  # March Madness Final
        'keywords': {
            'college_athlete': {'engagement': 6800, 'tweets': 350},
            'NIL': {'engagement': 7200, 'tweets': 380}
        },
        'demographics': {
            'age': {'13-17': 0.15, '18-24': 0.40}
        }
    }
}

# ========================
# DATA GENERATION ENGINE
# ========================
def generate_organic_pattern(base, volatility, seasonal_boost=0):
    """Generates realistic social media patterns"""
    noise = random.uniform(-volatility, volatility)
    seasonality = np.sin(len(data['date'])/52 * 2*np.pi) * seasonal_boost
    return max(0, int(base * (1 + noise + seasonality)))

for i, date in enumerate(dates):
    date_str = date.strftime('%Y-%m-%d')
    week_num = date.isocalendar()[1]
    month = date.month
    
    # 1. Handle special events
    if date_str in special_events:
        event = special_events[date_str]
        
        # Apply keyword boosts
        if 'keywords' in event:
            for kw, metrics in event['keywords'].items():
                for metric, value in metrics.items():
                    data[f"{kw}_{metric}"][i] = value
        
        # Apply media boosts
        if 'media' in event:
            for media, metrics in event['media'].items():
                for metric, value in metrics.items():
                    data[f"{media}_{metric}"][i] = value
        
        # Apply demographic shifts
        if 'demographics' in event:
            for demo, groups in event['demographics'].items():
                for group, value in groups.items():
                    data[f"{demo}_{group}_pct"][i] = value
        
        # Apply influencer data
        if 'influencers' in event:
            for inf_type, value in event['influencers'].items():
                data[inf_type][i] = value
    
    # 2. Generate organic baseline data
    # Core metrics
    data['total_engagement'][i] = generate_organic_pattern(2000, 0.3, 0.2)
    data['total_tweets'][i] = generate_organic_pattern(500, 0.2)
    data['avg_likes'][i] = generate_organic_pattern(150, 0.25)
    data['avg_retweets'][i] = generate_organic_pattern(30, 0.3)
    data['avg_replies'][i] = generate_organic_pattern(15, 0.4)
    
    # Keyword data
    for kw in KEYWORDS:
        kw_boost = 1.0
        # Apply seasonal boosts
        if kw == 'NIL' and month in [3,4]:  # March Madness
            kw_boost = 1.8
        elif kw == 'athlete_branding' and month in [6,7]:  # Offseason
            kw_boost = 1.5
            
        data[f"{kw}_engagement"][i] = int(data['total_engagement'][i] * random.uniform(0.1, 0.3) * kw_boost)
        data[f"{kw}_tweets"][i] = int(data['total_tweets'][i] * random.uniform(0.05, 0.2) * kw_boost)
        data[f"{kw}_likes"][i] = int(data['avg_likes'][i] * random.uniform(0.8, 1.2) * kw_boost)
    
    # Media data
    for media in MEDIA_OUTLETS:
        data[f"{media}_mentions"][i] = generate_organic_pattern(15, 0.4)
        data[f"{media}_engagement"][i] = data[f"{media}_mentions"][i] * random.randint(50, 200)
    
    # Demographic data (if not set by special event)
    for demo in DEMOGRAPHICS:
        for group in DEMOGRAPHICS[demo]:
            if data[f"{demo}_{group}_pct"][i] == 0:
                base_pct = DEMOGRAPHICS[demo][group]['pct']
                data[f"{demo}_{group}_pct"][i] = base_pct * random.uniform(0.9, 1.1)
    
    # Viral content generation
    if random.random() < 0.1:  # 10% chance of viral tweet
        random_kw = random.choice(list(KEYWORDS.keys()))
        data['viral_tweet'][i] = (
            f"Viral {random_kw.replace('_', ' ')} tweet: "
            f"{random.choice(['Breaking:', 'Hot take:', 'Exclusive:'])} "
            f"{random.choice(KEYWORDS[random_kw]['related_terms'])} "
            f"trending with {random.randint(5000, 20000)} likes"
        )

# Normalize demographics
for i in range(len(dates)):
    for demo in DEMOGRAPHICS:
        total = sum(data[f"{demo}_{group}_pct"][i] for group in DEMOGRAPHICS[demo])
        for group in DEMOGRAPHICS[demo]:
            data[f"{demo}_{group}_pct"][i] /= total

# Create DataFrame
df = pd.DataFrame(data)

# ========================
# VISUALIZATION ENGINE
# ========================
def save_viz(fig, name):
    fig.savefig(f"{OUTPUT_DIR}{name}.png", bbox_inches='tight', dpi=300)
    plt.close()

# 1. Keyword Performance Dashboard
plt.figure(figsize=(16, 12))
plt.suptitle("Keyword Performance Dashboard", y=1.02)

# Engagement Timeline
plt.subplot(2, 2, 1)
for kw in KEYWORDS:
    plt.plot(df['date'], df[f"{kw}_engagement"], 
             color=KEYWORDS[kw]['color'], 
             label=kw.replace('_', ' ').title())
plt.title("Engagement by Keyword")
plt.legend()

# Tweet Volume
plt.subplot(2, 2, 2)
for kw in KEYWORDS:
    plt.plot(df['date'], df[f"{kw}_tweets"], 
             color=KEYWORDS[kw]['color'],
             label=kw.replace('_', ' ').title())
plt.title("Tweet Volume by Keyword")

# Media Correlation
plt.subplot(2, 2, 3)
media_kw_corr = df[[f"{kw}_engagement" for kw in KEYWORDS] + 
                   [f"{media}_mentions" for media in MEDIA_OUTLETS]].corr()
sns.heatmap(media_kw_corr.iloc[:len(KEYWORDS), len(KEYWORDS):], 
            annot=True, cmap='coolwarm', 
            xticklabels=list(MEDIA_OUTLETS.keys()),
            yticklabels=[kw.replace('_', ' ').title() for kw in KEYWORDS])
plt.title("Media Mentions vs Keyword Engagement")

# Demographic Heatmap
plt.subplot(2, 2, 4)
demo_kw_matrix = []
for kw in KEYWORDS:
    row = []
    for demo in ['race', 'age', 'gender']:
        for group in DEMOGRAPHICS[demo]:
            corr = df[f"{kw}_engagement"].corr(df[f"{demo}_{group}_pct"])
            row.append(corr)
    demo_kw_matrix.append(row)

sns.heatmap(demo_kw_matrix, 
            xticklabels=[f"{demo[:1]}_{group}" for demo in DEMOGRAPHICS 
                        for group in DEMOGRAPHICS[demo]],
            yticklabels=list(KEYWORDS.keys()),
            cmap='coolwarm')
plt.title("Demographic Correlations by Keyword")

plt.tight_layout()
save_viz(plt, 'keyword_dashboard')

# 2. Media Influence Report
plt.figure(figsize=(14, 10))
plt.suptitle("Media Influence Analysis", y=1.02)

# Mentions Timeline
plt.subplot(2, 2, 1)
for media in MEDIA_OUTLETS:
    plt.plot(df['date'], df[f"{media}_mentions"], 
             color=MEDIA_OUTLETS[media]['color'],
             label=media)
plt.title("Media Mentions Over Time")
plt.legend()

# Engagement vs Mentions
plt.subplot(2, 2, 2)
for media in MEDIA_OUTLETS:
    plt.scatter(df[f"{media}_mentions"], df[f"{media}_engagement"],
               color=MEDIA_OUTLETS[media]['color'], label=media)
plt.title("Mentions vs Engagement")
plt.legend()

# Media Share of Voice
plt.subplot(2, 2, 3)
mentions_sum = sum(df[f"{media}_mentions"].sum() for media in MEDIA_OUTLETS)
media_share = [(media, df[f"{media}_mentions"].sum()/mentions_sum) 
              for media in MEDIA_OUTLETS]
media_share.sort(key=lambda x: x[1], reverse=True)
plt.pie([x[1] for x in media_share],
        labels=[x[0] for x in media_share],
        colors=[MEDIA_OUTLETS[x[0]]['color'] for x in media_share],
        autopct='%1.1f%%')
plt.title("Media Share of Voice")

# Top Media Days
plt.subplot(2, 2, 4)
top_media_days = pd.concat([
    df[['date'] + [f"{media}_mentions" for media in MEDIA_OUTLETS]]
    .sort_values(f"{media}_mentions", ascending=False).head(3)
    for media in MEDIA_OUTLETS
])
sns.heatmap(top_media_days.set_index('date'), annot=True, cmap='YlOrRd')
plt.title("Peak Media Mention Days")

plt.tight_layout()
save_viz(plt, 'media_influence')

# 3. Demographic Insights
fig, axs = plt.subplots(3, 1, figsize=(14, 15))

# Race Engagement
for race in DEMOGRAPHICS['race']:
    axs[0].plot(df['date'], df[f"race_{race}_pct"] * df['total_engagement'],
               color=DEMOGRAPHICS['race'][race]['color'],
               label=race.title())
axs[0].set_title("Engagement by Race")
axs[0].legend()

# Age Engagement
for age in DEMOGRAPHICS['age']:
    axs[1].plot(df['date'], df[f"age_{age}_pct"] * df['total_engagement'],
               color=DEMOGRAPHICS['age'][age]['color'],
               label=age)
axs[1].set_title("Engagement by Age Group")
axs[1].legend()

# Gender Engagement
for gender in DEMOGRAPHICS['gender']:
    axs[2].plot(df['date'], df[f"gender_{gender}_pct"] * df['total_engagement'],
               color=DEMOGRAPHICS['gender'][gender]['color'],
               label=gender.title())
axs[2].set_title("Engagement by Gender")
axs[2].legend()

plt.tight_layout()
save_viz(fig, 'demographic_insights')

# 4. Viral Content Analysis
if df['viral_tweet'].notna().sum() > 0:
    # Word Cloud of Viral Tweets
    text = ' '.join(df['viral_tweet'].dropna())
    wordcloud = WordCloud(width=1200, height=600).generate(text)
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Viral Tweet Content Analysis")
    save_viz(plt, 'viral_content_wordcloud')

    # Impact Analysis
    viral_days = df[df['viral_tweet'].notna()]
    plt.figure(figsize=(14, 6))
    plt.plot(df['date'], df['total_engagement'], label='Baseline')
    plt.scatter(viral_days['date'], viral_days['total_engagement'],
               color='red', label='Viral Days')
    plt.title("Viral Tweet Impact on Engagement")
    plt.legend()
    save_viz(plt, 'viral_impact')

# ========================
# DATA EXPORT
# ========================
# Save full dataset
df.to_csv(f"{OUTPUT_DIR}twitter_analytics_full.csv", index=False)

# Save summary report
summary = df.describe().T
summary.to_csv(f"{OUTPUT_DIR}analytics_summary.csv")

print(f"""
✅ ANALYSIS COMPLETE
📁 Files saved to: {OUTPUT_DIR}
   - twitter_analytics_full.csv
   - analytics_summary.csv
   - keyword_dashboard.png
   - media_influence.png  
   - demographic_insights.png
   - viral_content_wordcloud.png (if viral tweets detected)
""")

print("\n💡 Pro Tip: Use pandas to explore correlations:")
print("df.corr()[['total_engagement', 'NIL_engagement']].sort_values('total_engagement', ascending=False)")