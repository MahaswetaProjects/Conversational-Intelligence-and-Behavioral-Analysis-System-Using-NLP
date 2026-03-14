"""
WhatsApp Chat Preprocessor
Handles parsing, cleaning, and feature engineering from raw WhatsApp exports.
"""

import re
import io
import pandas as pd
import numpy as np
from datetime import datetime


# ── WhatsApp Message Regex ────────────────────────────────────────────────────
PATTERN = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?[apAPmM]*)\s-\s(.*?):\s(.*)'

SYSTEM_KEYWORDS = [
    "<Media omitted>", "deleted", "Messages and calls are end-to-end encrypted",
    "added", "removed", "left", "joined", "changed", "created",
    "pinned", "missed voice call", "missed video call"
]


def parse_whatsapp_export(text: str) -> pd.DataFrame:
    """Parse raw WhatsApp chat export text into a DataFrame."""
    matches = re.findall(PATTERN, text)
    if not matches:
        return pd.DataFrame()

    df = pd.DataFrame(matches, columns=["date", "time", "user", "message"])
    return df


def clean_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Remove system messages, media, and empty rows."""
    for kw in SYSTEM_KEYWORDS:
        df = df[~df["message"].str.contains(kw, case=False, na=False)]

    df = df[df["message"].str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def anonymize_users(df: pd.DataFrame) -> pd.DataFrame:
    """Replace real names/numbers with User1, User2, ..."""
    unique_users = df["user"].unique()
    user_map = {u: f"User{i+1}" for i, u in enumerate(unique_users)}
    df["user"] = df["user"].map(user_map)
    return df, user_map


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date+time and extract temporal features."""
    try:
        df["datetime"] = pd.to_datetime(
            df["date"] + " " + df["time"],
            format="%m/%d/%y %I:%M %p",
            errors="coerce"
        )
    except Exception:
        df["datetime"] = pd.to_datetime(
            df["date"] + " " + df["time"],
            infer_datetime_format=True,
            errors="coerce"
        )

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"]      = df["datetime"].dt.hour
    df["day"]       = df["datetime"].dt.day_name()
    df["month"]     = df["datetime"].dt.month_name()
    df["date_only"] = df["datetime"].dt.date
    df["week"]      = df["datetime"].dt.isocalendar().week.astype(int)

    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute text-level features per message."""
    import re as _re
    emoji_pattern = _re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=_re.UNICODE
    )

    df["msg_length"]  = df["message"].str.len()
    df["word_count"]  = df["message"].str.split().str.len()
    df["emoji_count"] = df["message"].apply(
        lambda x: len(emoji_pattern.findall(str(x)))
    )
    df["has_question"] = df["message"].str.contains(r'\?', na=False).astype(int)
    df["has_link"]     = df["message"].str.contains(
        r'http[s]?://', na=False
    ).astype(int)

    return df


def add_response_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute response time and conversation-start flags."""
    df = df.sort_values("datetime").reset_index(drop=True)
    df["prev_user"] = df["user"].shift(1)
    df["prev_time"] = df["datetime"].shift(1)

    df["response_time_min"] = np.where(
        df["user"] != df["prev_user"],
        (df["datetime"] - df["prev_time"]).dt.total_seconds() / 60,
        np.nan
    )

    # Flag conversation starters (gap > 60 min from previous message)
    df["gap_min"] = (df["datetime"] - df["prev_time"]).dt.total_seconds() / 60
    df["conv_starter"] = (df["gap_min"] > 60).astype(int)

    df.drop(columns=["prev_user", "prev_time", "gap_min"], inplace=True)
    return df


def full_pipeline(text: str, anonymize: bool = True) -> pd.DataFrame:
    """Run the complete preprocessing pipeline on raw WhatsApp text."""
    df = parse_whatsapp_export(text)
    if df.empty:
        return df

    df = clean_messages(df)

    if anonymize:
        df, _ = anonymize_users(df)

    df = add_datetime_features(df)
    df = add_text_features(df)
    df = add_response_features(df)

    return df


def load_from_upload(uploaded_file) -> pd.DataFrame:
    """Load a Streamlit UploadedFile and run full pipeline."""
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    return full_pipeline(content)
