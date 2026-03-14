"""
Analytics Engine
Computes all behavioral, temporal, and NLP insights from the processed DataFrame.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re


# ── Stop Words ────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "this", "that", "was", "are", "be", "as",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
    "our", "its", "me", "him", "us", "them", "do", "did", "does", "have",
    "has", "had", "not", "no", "yes", "ok", "okay", "will", "would",
    "can", "could", "should", "may", "might", "shall", "so", "if", "then",
    "than", "there", "their", "from", "by", "what", "when", "where", "how",
    "who", "which", "all", "also", "just", "up", "out", "about", "more",
    "been", "very", "get", "got", "go", "going", "come", "said", "one",
    "ka", "ki", "ke", "hai", "hain", "bhi", "kya", "nhi", "nahi", "aur",
    "mujhe", "mere", "mera", "tumhara", "apna", "woh", "vo", "yeh", "ye",
    "se", "ko", "ne", "par", "pe", "ab", "toh", "jo", "phir", "koi",
    "media", "omitted", "null", "none", "nan"
}


# ── Overview Metrics ──────────────────────────────────────────────────────────
def get_overview_metrics(df: pd.DataFrame) -> dict:
    """Return top-level KPIs for the Overview tab."""
    total_msgs   = len(df)
    total_users  = df["user"].nunique()
    avg_words    = df["word_count"].mean() if "word_count" in df.columns else 0
    total_emojis = df["emoji_count"].sum() if "emoji_count" in df.columns else 0

    date_range_days = 1
    if "date_only" in df.columns:
        dates = pd.to_datetime(df["date_only"])
        date_range_days = max((dates.max() - dates.min()).days, 1)

    msgs_per_day = round(total_msgs / date_range_days, 1)

    most_active = df["user"].value_counts().idxmax()
    most_active_count = df["user"].value_counts().max()

    avg_response = None
    if "response_time_min" in df.columns:
        valid = df["response_time_min"].dropna()
        valid = valid[(valid > 0) & (valid < 1440)]
        avg_response = round(valid.mean(), 1) if len(valid) > 0 else None

    return {
        "total_messages":   total_msgs,
        "total_users":      total_users,
        "avg_words":        round(avg_words, 1),
        "total_emojis":     int(total_emojis),
        "msgs_per_day":     msgs_per_day,
        "date_range_days":  date_range_days,
        "most_active_user": most_active,
        "most_active_count":most_active_count,
        "avg_response_min": avg_response,
    }


# ── User Analytics ────────────────────────────────────────────────────────────
def get_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user aggregate stats."""
    agg = df.groupby("user").agg(
        total_messages=("message", "count"),
        avg_msg_length=("msg_length",  "mean") if "msg_length"  in df.columns else ("message", lambda x: x.str.len().mean()),
        avg_word_count=("word_count",  "mean") if "word_count"  in df.columns else ("message", lambda x: x.str.split().str.len().mean()),
        total_emojis  =("emoji_count", "sum")  if "emoji_count" in df.columns else ("message", lambda x: 0),
    ).reset_index()

    # Response time if available
    if "response_time_min" in df.columns:
        rt = (
            df[df["response_time_min"].notna() & (df["response_time_min"] > 0) & (df["response_time_min"] < 1440)]
            .groupby("user")["response_time_min"]
            .mean()
            .round(1)
            .reset_index()
            .rename(columns={"response_time_min": "avg_response_min"})
        )
        agg = agg.merge(rt, on="user", how="left")

    agg["avg_msg_length"] = agg["avg_msg_length"].round(1)
    agg["avg_word_count"] = agg["avg_word_count"].round(1)
    return agg.sort_values("total_messages", ascending=False)


def get_behavioral_fingerprint(df: pd.DataFrame, user: str) -> dict:
    """Return a behavioral profile dict for a single user."""
    u = df[df["user"] == user]
    if u.empty:
        return {}

    all_stats  = get_user_stats(df)
    user_stats = all_stats[all_stats["user"] == user].iloc[0]

    # Peak hour
    peak_hour = u["hour"].value_counts().idxmax() if "hour" in u.columns else None

    # Peak day
    peak_day = u["day"].value_counts().idxmax() if "day" in u.columns else None

    # Vocabulary richness (unique words / total words)
    all_words  = " ".join(u["message"].astype(str)).lower().split()
    clean_words = [w for w in all_words if w.isalpha() and w not in STOP_WORDS]
    vocab_richness = round(len(set(clean_words)) / max(len(clean_words), 1), 3)

    # Questions ratio
    q_ratio = round(u["has_question"].mean(), 3) if "has_question" in u.columns else 0

    # Emoji lover score (emojis per message)
    emoji_rate = round(u["emoji_count"].mean(), 2) if "emoji_count" in u.columns else 0

    # Conversation starter rate
    starter_rate = round(u["conv_starter"].mean(), 3) if "conv_starter" in u.columns else 0

    return {
        "user":           user,
        "total_messages": int(user_stats["total_messages"]),
        "avg_msg_length": float(user_stats["avg_msg_length"]),
        "avg_word_count": float(user_stats["avg_word_count"]),
        "total_emojis":   int(user_stats["total_emojis"]),
        "peak_hour":      peak_hour,
        "peak_day":       peak_day,
        "vocab_richness": vocab_richness,
        "question_ratio": q_ratio,
        "emoji_rate":     emoji_rate,
        "conv_starter_rate": starter_rate,
    }


# ── Time Analytics ────────────────────────────────────────────────────────────
def get_hourly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Messages per hour (0–23)."""
    return (
        df.groupby("hour")["message"]
        .count()
        .reset_index(name="messages")
        .sort_values("hour")
    )


def get_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Messages per date."""
    if "date_only" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("date_only")["message"]
        .count()
        .reset_index(name="messages_per_day")
    )


def get_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Hour × Day matrix for heatmap."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    heat = (
        df.groupby(["day", "hour"])["message"]
        .count()
        .reset_index(name="count")
    )
    heat["day"] = pd.Categorical(heat["day"], categories=day_order, ordered=True)
    return heat.sort_values(["day", "hour"])


def get_monthly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Messages per month."""
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    m = (
        df.groupby("month")["message"]
        .count()
        .reset_index(name="messages")
    )
    m["month"] = pd.Categorical(m["month"], categories=month_order, ordered=True)
    return m.sort_values("month")


# ── NLP Insights ─────────────────────────────────────────────────────────────
def get_top_words(df: pd.DataFrame, n: int = 30, user: str = None) -> list[tuple]:
    """Return (word, count) for the top-n most frequent words."""
    subset = df[df["user"] == user] if user else df
    text = " ".join(subset["message"].astype(str)).lower()
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    words = [w for w in words if w not in STOP_WORDS]
    return Counter(words).most_common(n)


def get_wordcloud_text(df: pd.DataFrame, user: str = None) -> str:
    """Return cleaned text for WordCloud generation."""
    subset = df[df["user"] == user] if user else df
    text = " ".join(subset["message"].astype(str)).lower()
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    return " ".join(w for w in words if w not in STOP_WORDS)


def get_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment polarity to each message. Returns df with sentiment cols."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = df["message"].astype(str).apply(
            lambda x: sia.polarity_scores(x)["compound"]
        )
        df = df.copy()
        df["sentiment"] = scores
        df["sentiment_label"] = df["sentiment"].apply(
            lambda s: "Positive" if s > 0.05 else ("Negative" if s < -0.05 else "Neutral")
        )
    except ImportError:
        df = df.copy()
        df["sentiment"] = 0.0
        df["sentiment_label"] = "Neutral"
    return df


def get_sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Daily average sentiment score."""
    if "sentiment" not in df.columns:
        df = get_sentiment_scores(df)
    return (
        df.groupby("date_only")["sentiment"]
        .mean()
        .reset_index(name="avg_sentiment")
    )


def get_user_sentiment_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment per user."""
    if "sentiment" not in df.columns:
        df = get_sentiment_scores(df)
    s = (
        df.groupby("user")["sentiment"]
        .mean()
        .reset_index(name="avg_sentiment")
        .sort_values("avg_sentiment", ascending=False)
    )
    s["avg_sentiment"] = s["avg_sentiment"].round(3)
    return s


# ── Response Chain ────────────────────────────────────────────────────────────
def get_response_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Who responds to whom — returns edge list with counts."""
    df2 = df.copy().sort_values("datetime")
    df2["next_user"] = df2["user"].shift(-1)
    pairs = (
        df2[df2["user"] != df2["next_user"]]
        .groupby(["user", "next_user"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return pairs


# ── Conversation Momentum ─────────────────────────────────────────────────────
def get_conversation_momentum(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Rolling 7-day message volume to show conversation momentum."""
    daily = get_daily_trend(df)
    daily["date_only"] = pd.to_datetime(daily["date_only"])
    daily = daily.sort_values("date_only")
    daily["momentum"] = daily["messages_per_day"].rolling(window, min_periods=1).mean()
    return daily
