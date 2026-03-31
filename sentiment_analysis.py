"""
X Platform Sentiment Analysis Pipeline
=======================================
A self-contained NLP pipeline for analysing tweet sentiment.
Covers: data loading, text cleaning, VADER-style scoring,
classification, and all four visualisation dashboards.

Requirements:
    pip install pandas numpy matplotlib seaborn

Usage:
    # With your own CSV:
    python sentiment_analysis.py --input your_tweets.csv --text_col "tweet_text"

    # Demo mode (uses built-in synthetic dataset):
    python sentiment_analysis.py --demo
"""

import re
import random
import argparse
import warnings
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Colour palette ─────────────────────────────────────────────────────────────
POS  = "#2ecc71"
NEG  = "#e74c3c"
NEU  = "#95a5a6"
BLUE = "#3498db"
PURP = "#9b59b6"
ORG  = "#e67e22"
DARK = "#2c3e50"
SENT_PAL = {"Positive": POS, "Negative": NEG, "Neutral": NEU}

# ── Stopwords ──────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","is","it","in","to","of","and","or","but","for","on","at",
    "with","was","are","be","has","have","this","that","its","we","our","my",
    "your","their","they","i","you","he","she","so","as","by","from","not","no",
    "do","can","will","just","more","very","too","also","been","would","could",
    "should","than","into","if","up","out","about","like","all","one","who",
    "what","when","how","s","t","re","m","ll","ve",
}

# ── VADER-style sentiment lexicon (condensed) ──────────────────────────────────
LEXICON = {
    # strongly positive
    "love":0.9,"incredible":0.85,"amazing":0.88,"beautiful":0.80,"fantastic":0.87,
    "excellent":0.84,"outstanding":0.86,"wonderful":0.83,"brilliant":0.82,
    "inspiring":0.78,"excited":0.76,"grateful":0.80,"hopeful":0.72,"proud":0.75,
    "historic":0.65,"revolutionary":0.70,"phenomenal":0.88,"legendary":0.85,
    "thrilling":0.78,"electric":0.75,"great":0.70,"good":0.60,"nice":0.55,
    "impressive":0.72,"happy":0.74,"joy":0.80,"celebrate":0.76,"win":0.68,
    "victory":0.74,"champion":0.78,"record":0.55,"hope":0.65,"progress":0.60,
    "improve":0.58,"benefit":0.55,"effective":0.60,"success":0.72,"achieve":0.65,
    # mildly positive
    "interesting":0.20,"potential":0.18,"possible":0.15,"notable":0.22,
    "significant":0.18,"grow":0.30,"growing":0.32,"better":0.40,"best":0.55,
    # strongly negative
    "terrible":-.85,"awful":-.82,"disgusting":-.88,"horrible":-.84,"tragic":-.80,
    "devastating":-.82,"outrageous":-.80,"disgraceful":-.84,"heartbreaking":-.78,
    "horrifying":-.82,"appalling":-.80,"corruption":-.75,"corrupt":-.78,
    "fail":-.65,"failing":-.70,"failed":-.68,"broken":-.72,"wrong":-.68,
    "crisis":-.65,"disaster":-.75,"dangerous":-.70,"harmful":-.72,"toxic":-.74,
    "abuse":-.80,"violent":-.78,"attack":-.60,"destroy":-.72,"collapse":-.68,
    "hypocrisy":-.74,"betrayal":-.76,"fraud":-.78,"lie":-.70,"fake":-.60,
    "worried":-.55,"concern":-.40,"fear":-.58,"anxiety":-.52,"stress":-.50,
    "exhausted":-.55,"frustrated":-.60,"angry":-.65,"unfair":-.62,"wrong":-.65,
    # mildly negative
    "mixed":-.10,"complex":-.05,"unclear":-.10,"slow":-.20,"difficult":-.25,
    "challenge":-.15,"problem":-.30,"issue":-.25,"risk":-.28,"loss":-.40,
}

# negation and intensifier lists
NEGATIONS   = {"not","no","never","neither","nobody","nothing","neither","nor","barely","hardly","scarcely"}
INTENSIFIERS = {"very","extremely","absolutely","completely","totally","utterly","deeply","highly","incredibly","so","really","truly"}
DIMINISHERS  = {"somewhat","slightly","a","bit","little","rather","fairly","kind","sort","mildly"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Remove URLs, mentions, hashtag symbols, and normalise whitespace."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                     # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                # keep hashtag words
    text = re.sub(r"[^\w\s!?.,']", " ", text)            # special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    """Lowercase and split into word tokens."""
    return re.sub(r"[^a-z\s]", " ", text.lower()).split()


# ══════════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT SCORING  (rule-based, VADER-inspired)
# ══════════════════════════════════════════════════════════════════════════════

def score_tweet(text: str) -> float:
    """
    Return a sentiment score in [-1, +1].

    Algorithm:
      - For each token in the lexicon, apply:
          * negation  (flip sign if a negation appeared in the prior 3 tokens)
          * intensifier (multiply by 1.3)
          * diminisher  (multiply by 0.7)
      - Normalise the raw sum to [-1, +1] via tanh.
      - Boost slightly if the tweet contains '!' (enthusiasm) or
        all-caps words (shouting).
    """
    tokens = tokenise(clean_text(text))
    raw = 0.0
    for i, tok in enumerate(tokens):
        if tok not in LEXICON:
            continue
        val = LEXICON[tok]
        window = tokens[max(0, i - 3): i]
        # negation
        if any(w in NEGATIONS for w in window):
            val *= -0.8
        # intensifier
        if any(w in INTENSIFIERS for w in window):
            val *= 1.3
        # diminisher
        if any(w in DIMINISHERS for w in window):
            val *= 0.7
        raw += val

    # punctuation boosts
    excl = text.count("!")
    caps = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
    if raw > 0:
        raw += 0.05 * excl + 0.03 * caps
    elif raw < 0:
        raw -= 0.05 * excl + 0.03 * caps

    # normalise to [-1, +1]
    return float(np.tanh(raw / 3.0))


def classify(score: float, pos_thresh: float = 0.15, neg_thresh: float = -0.15) -> str:
    """Map a continuous score to a sentiment label."""
    if score > pos_thresh:
        return "Positive"
    if score < neg_thresh:
        return "Negative"
    return "Neutral"


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING  &  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_score(
    path: str,
    text_col: str = "tweet",
    date_col: str | None = "date",
    topic_col: str | None = "topic",
    likes_col: str | None = "likes",
    retweets_col: str | None = "retweets",
    replies_col: str | None = "replies",
) -> pd.DataFrame:
    """
    Load a CSV of tweets, clean text, score sentiment, and return an
    enriched DataFrame ready for analysis.

    Parameters
    ----------
    path        : Path to CSV file.
    text_col    : Column containing tweet text.
    date_col    : Column with tweet date (optional).
    topic_col   : Column with topic/category label (optional).
    likes_col   : Column with like counts (optional).
    retweets_col: Column with retweet counts (optional).
    replies_col : Column with reply counts (optional).

    Returns
    -------
    pd.DataFrame with added columns:
        clean_text, score, sentiment,
        month_num, month, week, day_of_week (if date present),
        tweet_len
    """
    df = pd.read_csv(path)

    # ── rename optional columns to standard names ──────────────────────────
    rename = {}
    if date_col and date_col in df.columns:
        rename[date_col] = "date"
    if topic_col and topic_col in df.columns:
        rename[topic_col] = "topic"
    if likes_col and likes_col in df.columns:
        rename[likes_col] = "likes"
    if retweets_col and retweets_col in df.columns:
        rename[retweets_col] = "retweets"
    if replies_col and replies_col in df.columns:
        rename[replies_col] = "replies"
    if text_col != "tweet" and text_col in df.columns:
        rename[text_col] = "tweet"
    df.rename(columns=rename, inplace=True)

    # ── score ──────────────────────────────────────────────────────────────
    df["clean_text"] = df["tweet"].apply(clean_text)
    df["score"]      = df["clean_text"].apply(score_tweet).round(4)
    df["sentiment"]  = df["score"].apply(classify)
    df["tweet_len"]  = df["tweet"].str.len()

    # ── date features ───────────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month_num"]   = df["date"].dt.month
        df["month"]       = df["date"].dt.strftime("%b")
        df["week"]        = df["date"].dt.isocalendar().week.astype(int)
        df["day_of_week"] = df["date"].dt.strftime("%a")
        df["quarter"]     = df["date"].dt.quarter.apply(lambda q: f"Q{q}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. SYNTHETIC DEMO DATASET
# ══════════════════════════════════════════════════════════════════════════════

def make_demo_dataset(n_per_topic: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic tweet dataset for demonstration."""
    random.seed(seed)
    np.random.seed(seed)

    TOPICS = {
        "AI & Technology": [
            ("ChatGPT is revolutionizing how we work! Absolutely love the productivity boost", 0.85),
            ("AI is taking over jobs and nobody seems to care. This is terrifying", -0.75),
            ("Interesting to see how AI tools are evolving. Mixed feelings tbh.", 0.05),
            ("OpenAI new model is incredible. The future is now!", 0.90),
            ("Big Tech keeps collecting our data with zero accountability. Disgusting.", -0.80),
            ("The progress in robotics this year has been jaw-dropping. Amazing times!", 0.80),
            ("Privacy concerns with AI systems keep growing. We need real regulation.", -0.45),
            ("Social media algorithms are designed to keep you addicted. Wake up.", -0.70),
        ],
        "Climate & Environment": [
            ("The wildfires this season are devastating. My heart goes out to everyone", -0.80),
            ("Solar power adoption hit a new record! Clean energy future is coming", 0.85),
            ("Climate summit ended with vague promises again. Leaders are failing us.", -0.75),
            ("Young activists are the true heroes of the climate movement. Inspiring!", 0.70),
            ("Renewable energy jobs surpassing fossil fuels for first time. Historic!", 0.80),
            ("Another plastic pollution report and still no real action. Outrageous.", -0.65),
            ("Reforestation projects are showing real results in degraded lands.", 0.75),
            ("Glacier retreat data is heartbreaking. We are running out of time.", -0.75),
        ],
        "Mental Health": [
            ("Finally opened up about my anxiety to my family. The relief is real", 0.75),
            ("Therapy changed my life. Still can not believe I waited so long to try it.", 0.85),
            ("Burnout is real and workplaces need to take it seriously. Enough excuses.", -0.50),
            ("Social media is making our mental health crisis worse every single day.", -0.70),
            ("Meditation practice has been a genuine game changer for my stress levels.", 0.80),
            ("Healthcare system makes mental health treatment unaffordable. This is wrong.", -0.75),
            ("Community support groups saved me when I had nothing else. Grateful.", 0.85),
            ("The way schools ignore student mental health is deeply troubling.", -0.65),
        ],
        "Sports": [
            ("That comeback in the final quarter was absolutely LEGENDARY!", 0.95),
            ("Referee decisions tonight were absolutely disgraceful. Rigged match.", -0.85),
            ("Young athletes today are pushing boundaries like never before. So inspiring!", 0.80),
            ("Home team played with heart and soul tonight. Proud of every single player!", 0.85),
            ("Olympics coverage this year has been absolutely phenomenal. Love it!", 0.80),
            ("Stadium atmosphere was electric today. Nothing beats live sports!", 0.90),
            ("Women sports finally getting the recognition and coverage they deserve!", 0.75),
            ("That athlete recovery story is one of the most moving things I have seen.", 0.85),
        ],
        "Politics & Society": [
            ("New infrastructure bill could create millions of jobs. Let us see the results.", 0.40),
            ("Political polarization is tearing communities apart. We need to talk more.", -0.55),
            ("Voter turnout reached a historic high! Democracy is alive and kicking.", 0.75),
            ("Corruption investigation reveals what we always suspected. Accountability now.", -0.65),
            ("Grassroots movement achieved real policy change! People power works!", 0.85),
            ("Cost of living crisis forcing families into impossible choices. This is wrong.", -0.80),
            ("Housing affordability crisis spreading to every major city. Need action now.", -0.70),
            ("The gap between rich and poor keeps growing. This system is broken.", -0.75),
        ],
    }

    start = datetime(2024, 1, 1)
    date_range = [start + timedelta(days=i) for i in range(366)]
    records = []

    for topic, tweet_list in TOPICS.items():
        for _ in range(n_per_topic):
            base_tweet, base_score = random.choice(tweet_list)
            score = float(np.clip(base_score + np.random.normal(0, 0.12), -1, 1))
            date  = random.choice(date_range)
            records.append({
                "date":      date,
                "topic":     topic,
                "tweet":     base_tweet,
                "likes":     max(0, int(np.random.exponential(120))),
                "retweets":  max(0, int(np.random.exponential(40))),
                "replies":   max(0, int(np.random.exponential(25))),
            })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    # score with pipeline (overrides synthetic score for consistency)
    df["clean_text"] = df["tweet"].apply(clean_text)
    df["score"]      = df["clean_text"].apply(score_tweet).round(4)
    df["sentiment"]  = df["score"].apply(classify)
    df["tweet_len"]  = df["tweet"].str.len()
    df["month_num"]  = df["date"].dt.month
    df["month"]      = df["date"].dt.strftime("%b")
    df["week"]       = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]= df["date"].dt.strftime("%a")
    df["quarter"]    = df["date"].dt.quarter.apply(lambda q: f"Q{q}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. WORD FREQUENCY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def top_words(subset: pd.DataFrame, n: int = 15) -> list[tuple[str, int]]:
    """Return top-n content words for a subset of tweets."""
    words = []
    for text in subset["clean_text"]:
        words += [
            w for w in re.sub(r"[^a-z\s]", " ", text.lower()).split()
            if w not in STOPWORDS and len(w) > 3
        ]
    return Counter(words).most_common(n)


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview(df: pd.DataFrame, save_path: str = "fig1_overview.png") -> None:
    """Figure 1 – Sentiment overview dashboard (2 × 3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("X Platform Sentiment Analysis — Overview",
                 fontsize=22, fontweight="bold", color=DARK, y=1.01)

    # 1a  Pie
    ax = axes[0, 0]
    counts = df["sentiment"].value_counts()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           colors=[SENT_PAL[s] for s in counts.index],
           explode=[0.05] * len(counts),
           wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2),
           textprops={"fontsize": 12, "fontweight": "bold"})
    ax.set_title("Overall Sentiment Distribution", fontsize=14, fontweight="bold", color=DARK)

    # 1b  Monthly trend
    ax = axes[0, 1]
    if "month_num" in df.columns:
        monthly = df.groupby(["month_num", "sentiment"]).size().reset_index(name="count")
        for sent, color in SENT_PAL.items():
            sub = monthly[monthly["sentiment"] == sent].groupby("month_num")["count"].sum()
            ax.plot(sub.index, sub.values, marker="o", label=sent, color=color, lw=2.5, ms=6)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([datetime(2024, m, 1).strftime("%b") for m in range(1, 13)], fontsize=9)
    ax.set_title("Monthly Sentiment Trend", fontsize=14, fontweight="bold", color=DARK)
    ax.set_ylabel("Tweet Count"); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # 1c  Stacked bar by topic
    ax = axes[0, 2]
    if "topic" in df.columns:
        tp = df.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
        tp_pct = tp.div(tp.sum(axis=1), axis=0) * 100
        short = [t.split("&")[0].strip() for t in tp_pct.index]
        bottom = np.zeros(len(tp_pct))
        for sent, color in [("Positive", POS), ("Neutral", NEU), ("Negative", NEG)]:
            if sent in tp_pct.columns:
                ax.barh(short, tp_pct[sent], left=bottom, color=color, label=sent, height=0.6)
                bottom += tp_pct[sent].values
    ax.set_title("Sentiment by Topic (%)", fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Percentage"); ax.legend(fontsize=9, loc="lower right"); ax.set_xlim(0, 100)

    # 1d  Avg score by topic
    ax = axes[1, 0]
    if "topic" in df.columns:
        avg = df.groupby("topic")["score"].mean().sort_values()
        colors_bar = [POS if v > 0 else NEG for v in avg]
        bars = ax.barh([t.split("&")[0].strip() for t in avg.index],
                       avg.values, color=colors_bar, height=0.6, edgecolor="white")
        ax.axvline(0, color=DARK, lw=1.5, ls="--")
        for bar, val in zip(bars, avg.values):
            ax.text(val + (0.01 if val >= 0 else -0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=9)
    ax.set_title("Avg Sentiment Score by Topic", fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Score (−1 → +1)")

    # 1e  Engagement by sentiment
    ax = axes[1, 1]
    eng_cols = [c for c in ["likes", "retweets", "replies"] if c in df.columns]
    if eng_cols:
        eng = df.groupby("sentiment")[eng_cols].mean()
        x, w = np.arange(len(eng_cols)), 0.25
        for i, (sent, color) in enumerate(SENT_PAL.items()):
            if sent in eng.index:
                ax.bar(x + i * w, eng.loc[sent], w, label=sent, color=color, edgecolor="white")
        ax.set_xticks(x + w)
        ax.set_xticklabels([f"Avg {c.title()}" for c in eng_cols])
    ax.set_title("Avg Engagement by Sentiment", fontsize=14, fontweight="bold", color=DARK)
    ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)

    # 1f  Violin score distribution
    ax = axes[1, 2]
    parts = ax.violinplot(
        [df[df["sentiment"] == s]["score"] for s in ["Positive", "Neutral", "Negative"]],
        positions=[1, 2, 3], showmedians=True)
    for pc, color in zip(parts["bodies"], [POS, NEU, NEG]):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    parts["cmedians"].set_color(DARK); parts["cmedians"].set_linewidth(2)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Positive", "Neutral", "Negative"])
    ax.set_title("Score Distribution by Sentiment", fontsize=14, fontweight="bold", color=DARK)
    ax.set_ylabel("Sentiment Score"); ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=2)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_temporal(df: pd.DataFrame, save_path: str = "fig2_temporal.png") -> None:
    """Figure 2 – Temporal sentiment patterns."""
    if "week" not in df.columns:
        print("  Skipping temporal chart (no date column).")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("X Platform — Temporal Sentiment Patterns",
                 fontsize=22, fontweight="bold", color=DARK, y=1.01)

    # 2a  Weekly rolling avg
    ax = axes[0, 0]
    weekly = df.groupby("week")["score"].mean()
    rolling = weekly.rolling(4, center=True).mean()
    ax.fill_between(weekly.index, weekly.values, alpha=0.15, color=BLUE)
    ax.plot(weekly.index, weekly.values, alpha=0.4, color=BLUE, lw=1)
    ax.plot(rolling.index, rolling.values, color=DARK, lw=2.5, label="4-week rolling avg")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("Weekly Sentiment Score — Rolling Average", fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Week of Year"); ax.set_ylabel("Avg Score"); ax.legend(); ax.grid(True, alpha=0.3)

    # 2b  Topic × month heatmap
    ax = axes[0, 1]
    if "topic" in df.columns:
        pivot = df.pivot_table(values="score", index="topic", columns="month_num", aggfunc="mean")
        pivot.columns = [datetime(2024, m, 1).strftime("%b") for m in pivot.columns]
        pivot.index = [t.split("&")[0].strip() for t in pivot.index]
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0, annot=True, fmt=".2f",
                    linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Sentiment Heatmap: Topic × Month", fontsize=14, fontweight="bold", color=DARK)

    # 2c  Day-of-week stacked bar
    ax = axes[1, 0]
    days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["day_abbr"] = df["day_of_week"].str[:3]
    day_sent = df.groupby(["day_abbr", "sentiment"]).size().unstack(fill_value=0)
    day_sent = day_sent.reindex([d for d in days_order if d in day_sent.index])
    day_pct = day_sent.div(day_sent.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(day_pct))
    for sent, color in [("Positive", POS), ("Neutral", NEU), ("Negative", NEG)]:
        if sent in day_pct.columns:
            ax.bar(day_pct.index, day_pct[sent], bottom=bottom,
                   color=color, label=sent, edgecolor="white", width=0.6)
            bottom += day_pct[sent].values
    ax.set_title("Sentiment by Day of Week", fontsize=14, fontweight="bold", color=DARK)
    ax.set_ylabel("Percentage"); ax.legend(fontsize=10); ax.set_ylim(0, 100); ax.grid(True, axis="y", alpha=0.3)

    # 2d  Quarterly boxplot
    ax = axes[1, 1]
    if "topic" in df.columns:
        topic_list = df["topic"].unique()
        palette = [BLUE, POS, PURP, ORG, "#e91e63"]
        for i, (topic, color) in enumerate(zip(topic_list, palette)):
            grp = df[df["topic"] == topic]
            data = [grp[grp["quarter"] == q]["score"].values for q in ["Q1", "Q2", "Q3", "Q4"]]
            positions = [i * 5 + j for j in range(4)]
            bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                            medianprops=dict(color=DARK, lw=2),
                            boxprops=dict(facecolor=color, alpha=0.6))
        short = [t.split("&")[0].strip() for t in topic_list]
        ax.set_xticks([2 + i * 5 for i in range(len(topic_list))])
        ax.set_xticklabels(short, fontsize=9)
    ax.set_title("Quarterly Score Distribution by Topic", fontsize=14, fontweight="bold", color=DARK)
    ax.set_ylabel("Sentiment Score"); ax.axhline(0, ls="--", color="gray", lw=1); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(pad=2)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_nlp(df: pd.DataFrame, save_path: str = "fig3_nlp.png") -> None:
    """Figure 3 – NLP & language pattern analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("X Platform — Language & NLP Pattern Analysis",
                 fontsize=22, fontweight="bold", color=DARK, y=1.01)

    # 3a  Top positive words
    ax = axes[0, 0]
    pos_words = top_words(df[df["sentiment"] == "Positive"])
    if pos_words:
        words, freqs = zip(*pos_words)
        ax.barh(words[::-1], freqs[::-1], color=POS, alpha=0.8, edgecolor="white")
    ax.set_title("Top Words — Positive Tweets", fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Frequency"); ax.grid(True, axis="x", alpha=0.3)

    # 3b  Top negative words
    ax = axes[0, 1]
    neg_words = top_words(df[df["sentiment"] == "Negative"])
    if neg_words:
        words, freqs = zip(*neg_words)
        ax.barh(words[::-1], freqs[::-1], color=NEG, alpha=0.8, edgecolor="white")
    ax.set_title("Top Words — Negative Tweets", fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Frequency"); ax.grid(True, axis="x", alpha=0.3)

    # 3c  Tweet length distribution
    ax = axes[1, 0]
    for sent, color in SENT_PAL.items():
        sub = df[df["sentiment"] == sent]["tweet_len"]
        ax.hist(sub, bins=25, alpha=0.6, label=sent, color=color, edgecolor="white")
    ax.set_title("Tweet Length Distribution by Sentiment", fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Characters"); ax.set_ylabel("Count"); ax.legend(); ax.grid(True, alpha=0.3)

    # 3d  Engagement scatter
    ax = axes[1, 1]
    if "likes" in df.columns and "retweets" in df.columns:
        sample = df.sample(min(600, len(df)), random_state=42)
        sc_colors = [SENT_PAL[s] for s in sample["sentiment"]]
        ax.scatter(sample["score"], sample["likes"],
                   c=sc_colors, alpha=0.55,
                   s=sample["retweets"] * 0.6 + 10, edgecolors="none")
        handles = [mpatches.Patch(color=c, label=s) for s, c in SENT_PAL.items()]
        ax.legend(handles=handles, fontsize=10)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_title("Sentiment Score vs. Likes\n(bubble size = retweets)",
                 fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Sentiment Score"); ax.set_ylabel("Likes"); ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=2)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_deep(df: pd.DataFrame, save_path: str = "fig4_deep.png") -> None:
    """Figure 4 – Deep topic analysis: radar + correlation heatmap."""
    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("X Platform — Deep Dive: Research Questions Analysis",
                 fontsize=22, fontweight="bold", color=DARK, y=1.03)

    # 4a  Radar
    ax_radar = fig.add_subplot(1, 2, 1, polar=True)
    if "topic" in df.columns:
        categories = ["% Positive", "% Negative", "Avg Likes", "Avg Retweets", "Score (+)"]
        n = len(categories)
        angles = [i / n * 2 * np.pi for i in range(n)] + [0]
        topic_colors = [BLUE, POS, PURP, ORG, "#e91e63"]
        for topic, color in zip(df["topic"].unique(), topic_colors):
            sub = df[df["topic"] == topic]
            pct_pos   = (sub["sentiment"] == "Positive").mean() * 100
            pct_neg   = (sub["sentiment"] == "Negative").mean() * 100
            avg_likes = sub["likes"].mean() / 3 if "likes" in df.columns else 0
            avg_rt    = sub["retweets"].mean() / 1.5 if "retweets" in df.columns else 0
            score_pos = (sub["score"].mean() + 1) * 50
            vals = [min(100, v) for v in [pct_pos, pct_neg, avg_likes, avg_rt, score_pos]] + \
                   [min(100, pct_pos)]
            ax_radar.plot(angles, vals, color=color, lw=2,
                          label=topic.split("&")[0].strip())
            ax_radar.fill(angles, vals, alpha=0.1, color=color)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=10, color=DARK)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax_radar.set_title("Topic Radar: Multi-Metric Comparison",
                       fontsize=14, fontweight="bold", color=DARK, pad=20)

    # 4b  Correlation heatmap
    ax_corr = fig.add_subplot(1, 2, 2)
    numeric_cols = [c for c in ["score", "likes", "retweets", "replies", "tweet_len"]
                    if c in df.columns]
    corr = df[numeric_cols].corr()
    col_labels = {"score": "Sent. Score", "likes": "Likes", "retweets": "Retweets",
                  "replies": "Replies", "tweet_len": "Tweet Len"}
    corr.columns = [col_labels.get(c, c) for c in corr.columns]
    corr.index   = [col_labels.get(c, c) for c in corr.index]
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax_corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5)
    ax_corr.set_title("Correlation: Sentiment & Engagement Metrics",
                      fontsize=14, fontweight="bold", color=DARK)

    plt.tight_layout(pad=3)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    """Print a concise statistical summary to stdout."""
    print("\n" + "=" * 55)
    print("  SENTIMENT ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"  Total tweets analysed : {len(df):,}")
    counts = df["sentiment"].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {label:<12}: {count:>5,}  ({pct:.1f}%)")
    print(f"\n  Avg score  : {df['score'].mean():.4f}")
    print(f"  Score std  : {df['score'].std():.4f}")
    print(f"  Score range: {df['score'].min():.4f} → {df['score'].max():.4f}")

    if "topic" in df.columns:
        print("\n  Avg score by topic:")
        for topic, score in df.groupby("topic")["score"].mean().sort_values().items():
            bar = "▓" * int(abs(score) * 20)
            sign = "+" if score >= 0 else ""
            print(f"    {topic:<28} {sign}{score:.3f}  {bar}")

    if "likes" in df.columns:
        print("\n  Avg engagement by sentiment:")
        eng_cols = [c for c in ["likes", "retweets", "replies"] if c in df.columns]
        eng = df.groupby("sentiment")[eng_cols].mean().round(1)
        print(eng.to_string())
    print("=" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(df: pd.DataFrame, output_prefix: str = "") -> None:
    """Run all four visualisations and print summary."""
    prefix = f"{output_prefix}_" if output_prefix else ""
    print("\n[1/4] Generating overview dashboard...")
    plot_overview(df, save_path=f"{prefix}fig1_overview.png")

    print("[2/4] Generating temporal analysis...")
    plot_temporal(df, save_path=f"{prefix}fig2_temporal.png")

    print("[3/4] Generating NLP & language analysis...")
    plot_nlp(df, save_path=f"{prefix}fig3_nlp.png")

    print("[4/4] Generating deep topic analysis...")
    plot_deep(df, save_path=f"{prefix}fig4_deep.png")

    print_summary(df)


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="X Platform Sentiment Analysis Pipeline")
    p.add_argument("--demo",       action="store_true",  help="Run with built-in synthetic dataset")
    p.add_argument("--input",      type=str, default=None, help="Path to your CSV file")
    p.add_argument("--text_col",   type=str, default="tweet",    help="Column containing tweet text")
    p.add_argument("--date_col",   type=str, default="date",     help="Column containing date (optional)")
    p.add_argument("--topic_col",  type=str, default="topic",    help="Column containing topic (optional)")
    p.add_argument("--likes_col",  type=str, default="likes",    help="Column containing likes (optional)")
    p.add_argument("--rt_col",     type=str, default="retweets", help="Column containing retweets (optional)")
    p.add_argument("--reply_col",  type=str, default="replies",  help="Column containing replies (optional)")
    p.add_argument("--out_prefix", type=str, default="",         help="Prefix for output image filenames")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo:
        print("Running in DEMO mode — generating synthetic dataset...")
        df = make_demo_dataset()
    elif args.input:
        print(f"Loading data from: {args.input}")
        df = load_and_score(
            path=args.input,
            text_col=args.text_col,
            date_col=args.date_col,
            topic_col=args.topic_col,
            likes_col=args.likes_col,
            retweets_col=args.rt_col,
            replies_col=args.reply_col,
        )
    else:
        print("No input provided. Use --demo or --input <file.csv>")
        raise SystemExit(1)

    run_pipeline(df, output_prefix=args.out_prefix)
