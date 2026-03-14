"""
ConvoIQ – Conversational Intelligence & Behavioral Analytics System
===================================================================
Placement-ready NLP project: WhatsApp chat analysis with authorship prediction.

Run:  streamlit run app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Resolve import path ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src import preprocessor, analytics

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ConvoIQ · Conversational Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0f1117; color: #e2e8f0; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2d3748; }
[data-testid="stSidebar"] *        { color: #e2e8f0 !important; }

/* ── Metric cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #243044 100%);
    border: 1px solid #2d4a6e;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-value { font-size: 2rem; font-weight: 700; color: #60a5fa; line-height: 1.1; }
.kpi-label { font-size: 0.78rem; color: #94a3b8; margin-top: .35rem; letter-spacing: .04em; text-transform: uppercase; }

/* ── Section headers ── */
.section-hdr {
    font-size: 1.15rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #3b82f6; padding-left: .7rem;
    margin: 1.8rem 0 .9rem;
}

/* ── Insight box ── */
.insight-box {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: .9rem 1.1rem;
    margin: .4rem 0; font-size: .9rem; color: #cbd5e1;
}
.insight-box b { color: #93c5fd; }

/* ── Fingerprint badge ── */
.badge {
    display: inline-block;
    background: #1e3a5f; border: 1px solid #2563eb;
    border-radius: 20px; padding: .25rem .75rem;
    font-size: .78rem; color: #93c5fd; margin: .2rem;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #1e2a3a; border-radius: 8px 8px 0 0;
    color: #94a3b8; padding: .5rem 1.2rem;
    border: 1px solid #2d3748; border-bottom: none;
}
.stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: white !important; }

/* ── Prediction result ── */
.pred-box {
    background: linear-gradient(135deg, #0c2340, #0f3a5e);
    border: 2px solid #3b82f6; border-radius: 14px;
    padding: 1.5rem 2rem; text-align: center;
}
.pred-user  { font-size: 2.2rem; font-weight: 800; color: #60a5fa; }
.pred-conf  { font-size: 1rem; color: #94a3b8; margin-top: .4rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants / Paths ─────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")

DEMO_CSV     = os.path.join(DATA_PATH,   "powerbi_chat_dataset.csv")
USER_SUM_CSV = os.path.join(DATA_PATH,   "user_summary.csv")
DAILY_CSV    = os.path.join(DATA_PATH,   "daily_trend.csv")
VEC_PATH     = os.path.join(MODELS_PATH, "vectorizer.pkl")
LE_PATH      = os.path.join(MODELS_PATH, "label_encoder.pkl")
MODEL_PATH   = os.path.join(MODELS_PATH, "model.pkl")

ACCENT  = "#3b82f6"
PALETTE = px.colors.qualitative.Plotly

# ── Plotly dark template ──────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0f1117",
    plot_bgcolor="#161b27",
    font=dict(color="#e2e8f0", size=12),
    margin=dict(l=40, r=20, t=45, b=40),
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    df = pd.read_csv(DEMO_CSV)
    df["date_only"] = pd.to_datetime(df["date_only"])
    df["datetime"]  = pd.to_datetime(df["datetime"])
    # Ensure required text-feature cols exist
    if "msg_length"  not in df.columns: df["msg_length"]  = df["message"].str.len()
    if "word_count"  not in df.columns: df["word_count"]  = df["message"].str.split().str.len()
    if "emoji_count" not in df.columns: df["emoji_count"] = 0
    if "has_question" not in df.columns:
        df["has_question"] = df["message"].str.contains(r'\?', na=False).astype(int)
    if "has_link" not in df.columns:
        df["has_link"] = df["message"].str.contains(r'http', na=False).astype(int)
    if "conv_starter" not in df.columns:
        df["conv_starter"] = 0
    return df


@st.cache_data(show_spinner=False)
def load_models():
    """Load ML model artifacts. Returns (model, vectorizer, label_encoder) or Nones."""
    try:
        import joblib
        model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        vec   = joblib.load(VEC_PATH)   if os.path.exists(VEC_PATH)   else None
        le    = joblib.load(LE_PATH)    if os.path.exists(LE_PATH)    else None
        return model, vec, le
    except Exception:
        return None, None, None


def kpi(label: str, value, icon: str = ""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{icon} {value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def section(title: str):
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


def dark_fig(fig):
    """Apply dark theme layout to a plotly figure."""
    fig.update_layout(**PLOT_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 ConvoIQ")
    st.markdown("*Conversational Intelligence System*")
    st.divider()

    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose input:",
        ["Use Demo Dataset", "Upload WhatsApp Export"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "Upload WhatsApp Export":
        uploaded_file = st.file_uploader(
            "Upload `.txt` WhatsApp export",
            type=["txt"],
            help="Export any WhatsApp chat: Open Chat → ⋮ → More → Export Chat (without media)"
        )

    st.divider()
    st.markdown("### 📊 Navigation")
    nav = st.radio(
        "Go to:",
        ["🏠 Overview", "👥 User Analytics", "⏰ Time Analysis",
         "💬 NLP Insights", "🤖 Author Prediction"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown(
        "<small style='color:#64748b'>Built with Python · Scikit-learn · Streamlit<br>"
        "NLP · Random Forest · TF-IDF · VADER</small>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# Load Data
# ══════════════════════════════════════════════════════════════════════════════

df = None

if data_source == "Use Demo Dataset":
    with st.spinner("Loading demo dataset…"):
        df = load_demo_data()
    st.sidebar.success(f"✅ Demo loaded · {len(df):,} messages")

elif uploaded_file is not None:
    with st.spinner("Parsing WhatsApp export…"):
        df = preprocessor.load_from_upload(uploaded_file)
    if df.empty:
        st.sidebar.error("❌ Could not parse this file. Check format.")
        df = None
    else:
        st.sidebar.success(f"✅ Parsed · {len(df):,} messages")

if df is None:
    # Landing state
    st.markdown('<p class="kpi-value" style="font-size:3rem;text-align:center;margin-top:3rem">🧠</p>', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;color:#60a5fa'>ConvoIQ</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:1.1rem'>Conversational Intelligence & Behavioral Analytics</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='max-width:600px;margin:2rem auto;background:#1e293b;border-radius:14px;padding:2rem;border:1px solid #334155'>
    <b style='color:#60a5fa'>What this system does:</b>
    <ul style='color:#cbd5e1;margin-top:.8rem;line-height:2'>
    <li>📊 Behavioral analytics from WhatsApp group chats</li>
    <li>🧬 Communication fingerprinting per user</li>
    <li>❤️ Sentiment analysis & emotional timeline</li>
    <li>🕐 Activity heatmaps & response-time analysis</li>
    <li>🤖 ML-powered author prediction (Random Forest + TF-IDF)</li>
    <li>☁️ Word clouds & NLP topic insights</li>
    </ul>
    <p style='color:#64748b;font-size:.85rem;margin-top:1rem'>← Select <b>Use Demo Dataset</b> from the sidebar to begin instantly, or upload your own chat export.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Pre-compute sentiment (cached)
@st.cache_data(show_spinner=False)
def enrich_sentiment(df_hash):
    return analytics.get_sentiment_scores(df)

df = enrich_sentiment(hash(str(df.shape) + str(df.iloc[0]["message"])))

# Load models
model, vectorizer, label_encoder = load_models()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

if nav == "🏠 Overview":
    st.markdown("## 🏠 Overview")
    st.markdown("<small style='color:#64748b'>High-level snapshot of the conversation dataset.</small>", unsafe_allow_html=True)

    metrics = analytics.get_overview_metrics(df)

    # ── KPI Row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Total Messages",    f"{metrics['total_messages']:,}",        "💬"), unsafe_allow_html=True)
    c2.markdown(kpi("Unique Users",      metrics["total_users"],                  "👥"), unsafe_allow_html=True)
    c3.markdown(kpi("Days Tracked",      metrics["date_range_days"],              "📅"), unsafe_allow_html=True)
    c4.markdown(kpi("Msgs / Day (avg)",  metrics["msgs_per_day"],                 "📈"), unsafe_allow_html=True)
    c5.markdown(kpi("Total Emojis",      f"{metrics['total_emojis']:,}",          "😊"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Message volume trend ──
    section("📈 Conversation Volume Over Time")
    momentum = analytics.get_conversation_momentum(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=momentum["date_only"], y=momentum["messages_per_day"],
        mode="lines", name="Daily Messages",
        line=dict(color="#3b82f6", width=1.5), fill="tozeroy",
        fillcolor="rgba(59,130,246,0.12)"
    ))
    fig.add_trace(go.Scatter(
        x=momentum["date_only"], y=momentum["momentum"],
        mode="lines", name="7-day Momentum",
        line=dict(color="#f59e0b", width=2.5, dash="dash")
    ))
    fig.update_layout(xaxis_title="Date", yaxis_title="Messages", **PLOT_LAYOUT,
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # ── Top contributors + Sentiment distribution ──
    col1, col2 = st.columns(2)

    with col1:
        section("🏆 Top 10 Most Active Users")
        top10 = df["user"].value_counts().head(10).reset_index()
        top10.columns = ["user", "messages"]
        fig2 = px.bar(
            top10, y="user", x="messages", orientation="h",
            color="messages", color_continuous_scale="Blues",
            labels={"messages": "Messages", "user": ""}
        )
        fig2.update_layout(**PLOT_LAYOUT, showlegend=False,
                           coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section("❤️ Overall Sentiment Distribution")
        sent_counts = df["sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        color_map = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}
        fig3 = px.pie(
            sent_counts, names="Sentiment", values="Count",
            color="Sentiment", color_discrete_map=color_map,
            hole=0.5
        )
        fig3.update_layout(**PLOT_LAYOUT, showlegend=True,
                           legend=dict(orientation="h", y=-0.1))
        fig3.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Key Insights ──
    section("💡 Auto-Generated Insights")
    most_active = metrics["most_active_user"]
    pct_share   = round(metrics["most_active_count"] / metrics["total_messages"] * 100, 1)
    avg_r       = f"{metrics['avg_response_min']} min" if metrics["avg_response_min"] else "N/A"
    pos_pct     = round((df["sentiment_label"] == "Positive").mean() * 100, 1)

    insight(f"<b>{most_active}</b> dominates the conversation with <b>{metrics['most_active_count']} messages</b> ({pct_share}% of all messages).")
    insight(f"Average group response time is <b>{avg_r}</b> — reflecting engagement speed across all participants.")
    insight(f"<b>{pos_pct}%</b> of messages carry a positive sentiment, indicating overall healthy group dynamics.")
    insight(f"The group sent an average of <b>{metrics['msgs_per_day']} messages/day</b> across <b>{metrics['date_range_days']} days</b> of activity.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — User Analytics
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "👥 User Analytics":
    st.markdown("## 👥 User Analytics")
    st.markdown("<small style='color:#64748b'>Per-user behavioral profiles and communication patterns.</small>", unsafe_allow_html=True)

    user_stats = analytics.get_user_stats(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        section("📋 User Leaderboard")
        display = user_stats.head(20).copy()
        display.columns = [c.replace("_", " ").title() for c in display.columns]
        st.dataframe(
            display.style.background_gradient(subset=["Total Messages"], cmap="Blues"),
            use_container_width=True, height=380
        )

    with col2:
        section("📊 Message Share — Top 8")
        top8 = user_stats.head(8)
        others_count = user_stats.iloc[8:]["total_messages"].sum()
        pie_data = pd.concat([
            top8[["user", "total_messages"]],
            pd.DataFrame([{"user": "Others", "total_messages": others_count}])
        ])
        fig = px.pie(pie_data, names="user", values="total_messages", hole=0.4)
        fig.update_layout(**PLOT_LAYOUT, showlegend=False)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Behavioral Fingerprint ──
    section("🧬 Behavioral Fingerprint")
    st.markdown("<small style='color:#94a3b8'>Select a user to reveal their unique communication DNA.</small>", unsafe_allow_html=True)

    # Only users with ≥3 messages
    active_users = user_stats[user_stats["total_messages"] >= 3]["user"].tolist()
    selected_user = st.selectbox("Select User", active_users, key="fingerprint_user")

    fp = analytics.get_behavioral_fingerprint(df, selected_user)

    if fp:
        b1, b2, b3, b4 = st.columns(4)
        b1.markdown(kpi("Total Messages",    fp["total_messages"], "💬"), unsafe_allow_html=True)
        b2.markdown(kpi("Avg Msg Length",    f"{fp['avg_msg_length']:.0f} chars", "📝"), unsafe_allow_html=True)
        b3.markdown(kpi("Vocab Richness",    fp["vocab_richness"], "📚"), unsafe_allow_html=True)
        b4.markdown(kpi("Emoji Rate",        fp["emoji_rate"], "😊"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        badges = []
        if fp["peak_hour"] is not None:
            time_label = "🌅 Morning" if fp["peak_hour"] < 12 else ("☀️ Afternoon" if fp["peak_hour"] < 17 else "🌙 Night Owl")
            badges.append(time_label)
        if fp["peak_day"]:
            badges.append(f"📅 {fp['peak_day']} Person")
        if fp["question_ratio"] > 0.15:
            badges.append("❓ Question Asker")
        if fp["emoji_rate"] > 0.5:
            badges.append("😊 Emoji Lover")
        if fp["conv_starter_rate"] > 0.1:
            badges.append("🔥 Conversation Starter")
        if fp["vocab_richness"] > 0.7:
            badges.append("📖 Rich Vocabulary")

        badge_html = " ".join(f'<span class="badge">{b}</span>' for b in badges)
        st.markdown(badge_html, unsafe_allow_html=True)

        # Radar chart
        st.markdown("<br>", unsafe_allow_html=True)
        col_r, col_m = st.columns(2)
        with col_r:
            # Normalise metrics for radar
            max_msgs = user_stats["total_messages"].max()
            max_len  = user_stats["avg_msg_length"].max()
            radar_vals = [
                fp["total_messages"]    / max(max_msgs, 1),
                fp["vocab_richness"],
                fp["emoji_rate"]        / max(user_stats["total_emojis"].max() / max(user_stats["total_messages"].max(), 1), 1),
                fp["question_ratio"],
                fp["conv_starter_rate"] * 3,
                fp["avg_msg_length"]    / max(max_len, 1),
            ]
            radar_labels = ["Volume", "Vocab\nRichness", "Emoji\nUsage",
                            "Questions", "Conv\nStarter", "Msg\nLength"]
            fig_r = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor="rgba(59,130,246,0.2)",
                line=dict(color="#3b82f6", width=2),
                name=selected_user
            ))
            fig_r.update_layout(
                **PLOT_LAYOUT,
                polar=dict(
                    bgcolor="#161b27",
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9), showticklabels=False),
                    angularaxis=dict(tickfont=dict(size=10, color="#94a3b8"))
                ),
                title=f"Communication DNA — {selected_user}",
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with col_m:
            section(f"Hourly Activity — {selected_user}")
            u_df  = df[df["user"] == selected_user]
            h_cnt = u_df["hour"].value_counts().reset_index()
            h_cnt.columns = ["hour", "count"]
            h_cnt = h_cnt.sort_values("hour")
            fig_h = px.bar(h_cnt, x="hour", y="count",
                           color="count", color_continuous_scale="Blues",
                           labels={"hour": "Hour of Day", "count": "Messages"})
            fig_h.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_h, use_container_width=True)

    st.divider()

    # ── Response Time Analysis ──
    section("⚡ Fastest Responders")
    if "avg_response_min" in user_stats.columns:
        resp_df = user_stats[user_stats["avg_response_min"].notna()].nsmallest(10, "avg_response_min")
        fig_rt = px.bar(
            resp_df, x="avg_response_min", y="user", orientation="h",
            color="avg_response_min", color_continuous_scale="Greens_r",
            labels={"avg_response_min": "Avg Response (min)", "user": ""}
        )
        fig_rt.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False,
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_rt, use_container_width=True)
    else:
        st.info("Response time data not available in this dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Time Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "⏰ Time Analysis":
    st.markdown("## ⏰ Time Analysis")
    st.markdown("<small style='color:#64748b'>When does the conversation happen? Heatmaps, peaks, and trends.</small>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section("🕐 Hourly Message Distribution")
        hourly = analytics.get_hourly_distribution(df)
        peak_h = hourly.loc[hourly["messages"].idxmax(), "hour"]
        fig = px.bar(
            hourly, x="hour", y="messages",
            color="messages", color_continuous_scale="Blues",
            labels={"hour": "Hour of Day", "messages": "Messages"}
        )
        fig.add_vline(x=peak_h, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Peak: {peak_h}:00", annotation_position="top")
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("📅 Day-of-Week Pattern")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_cnt = df["day"].value_counts().reindex(day_order, fill_value=0).reset_index()
        day_cnt.columns = ["day", "messages"]
        fig2 = px.bar(
            day_cnt, x="day", y="messages",
            color="messages", color_continuous_scale="Purples",
            labels={"day": "Day", "messages": "Messages"}
        )
        fig2.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Heatmap ──
    section("🔥 Activity Heatmap — Hour × Day")
    heat = analytics.get_heatmap_data(df)
    if not heat.empty:
        pivot = heat.pivot_table(index="day", columns="hour", values="count", fill_value=0)
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])

        fig_h = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="Blues",
            showscale=True,
            hoverongaps=False,
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Messages: %{z}<extra></extra>"
        ))
        fig_h.update_layout(**PLOT_LAYOUT, height=320,
                            xaxis_title="Hour of Day", yaxis_title="",
                            title="Message Activity — Heatmap")
        st.plotly_chart(fig_h, use_container_width=True)

    # ── Monthly trend ──
    section("📆 Monthly Volume")
    monthly = analytics.get_monthly_breakdown(df)
    if not monthly.empty:
        fig_m = px.bar(
            monthly, x="month", y="messages",
            color="messages", color_continuous_scale="Teal",
            labels={"month": "Month", "messages": "Messages"}
        )
        fig_m.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_m, use_container_width=True)

    # ── Insights ──
    section("💡 Time-Based Insights")
    peak_hour_label = f"{peak_h}:00 – {peak_h+1}:00"
    day_cnt_sorted  = day_cnt.sort_values("messages", ascending=False)
    peak_day        = day_cnt_sorted.iloc[0]["day"]
    quiet_day       = day_cnt_sorted.iloc[-1]["day"]
    insight(f"The group is most active at <b>{peak_hour_label}</b> — likely during lunch or evening downtime.")
    insight(f"<b>{peak_day}</b> sees the highest message volume, while <b>{quiet_day}</b> is the quietest day.")
    if not monthly.empty:
        peak_month = monthly.sort_values("messages", ascending=False).iloc[0]["month"]
        insight(f"<b>{peak_month}</b> recorded the highest monthly activity in this dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — NLP Insights
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "💬 NLP Insights":
    st.markdown("## 💬 NLP Insights")
    st.markdown("<small style='color:#64748b'>Language patterns, sentiment analysis, and word-level intelligence.</small>", unsafe_allow_html=True)

    # ── Sentiment Timeline ──
    section("❤️ Sentiment Timeline")
    sent_trend = analytics.get_sentiment_trend(df)
    if not sent_trend.empty:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=sent_trend["date_only"], y=sent_trend["avg_sentiment"],
            mode="lines", fill="tozeroy",
            line=dict(color="#22c55e", width=2),
            fillcolor="rgba(34,197,94,0.15)",
            name="Avg Sentiment"
        ))
        fig_s.add_hline(y=0, line_dash="dash", line_color="#64748b",
                        annotation_text="Neutral baseline")
        fig_s.update_layout(**PLOT_LAYOUT, xaxis_title="Date",
                            yaxis_title="Sentiment Score (-1 to +1)",
                            title="Daily Sentiment Trend")
        st.plotly_chart(fig_s, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section("😊 Most Positive Users")
        user_sent = analytics.get_user_sentiment_profile(df)
        top_pos   = user_sent.head(8)
        fig_sp = px.bar(
            top_pos, x="avg_sentiment", y="user", orientation="h",
            color="avg_sentiment", color_continuous_scale="Greens",
            range_color=[-1, 1],
            labels={"avg_sentiment": "Sentiment Score", "user": ""}
        )
        fig_sp.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False,
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_sp, use_container_width=True)

    with col2:
        section("😠 Most Negative Users")
        top_neg = user_sent.tail(8).sort_values("avg_sentiment")
        fig_sn = px.bar(
            top_neg, x="avg_sentiment", y="user", orientation="h",
            color="avg_sentiment", color_continuous_scale="Reds_r",
            range_color=[-1, 1],
            labels={"avg_sentiment": "Sentiment Score", "user": ""}
        )
        fig_sn.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False,
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_sn, use_container_width=True)

    st.divider()

    # ── Word Cloud ──
    section("☁️ Word Cloud")
    col_wc1, col_wc2 = st.columns([1, 2])
    with col_wc1:
        wc_user_options = ["All Users"] + df["user"].value_counts().head(30).index.tolist()
        wc_user = st.selectbox("Filter by user:", wc_user_options, key="wc_user")
        wc_colormap = st.selectbox("Color scheme:", ["Blues", "Purples", "YlOrRd", "Greens"], key="wc_cmap")

    with col_wc2:
        wc_text = analytics.get_wordcloud_text(df, None if wc_user == "All Users" else wc_user)
        if wc_text.strip():
            try:
                from wordcloud import WordCloud as WC
                wc = WC(
                    width=800, height=320, background_color="#0f1117",
                    colormap=wc_colormap, max_words=100,
                    prefer_horizontal=0.8, min_font_size=10
                ).generate(wc_text)
                fig_wc, ax = plt.subplots(figsize=(10, 4))
                fig_wc.patch.set_facecolor("#0f1117")
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc, use_container_width=True)
            except ImportError:
                st.info("Install `wordcloud` package to enable this feature.")
        else:
            st.info("Not enough text data for this selection.")

    # ── Top Words Bar ──
    section("🔤 Top Keywords Frequency")
    top_words = analytics.get_top_words(df, n=20,
                                        user=None if wc_user == "All Users" else wc_user)
    if top_words:
        words_df = pd.DataFrame(top_words, columns=["word", "count"])
        fig_w = px.bar(
            words_df, x="count", y="word", orientation="h",
            color="count", color_continuous_scale="Blues",
            labels={"count": "Frequency", "word": ""}
        )
        fig_w.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False,
                            yaxis=dict(autorange="reversed"), height=480)
        st.plotly_chart(fig_w, use_container_width=True)

    # ── Message Length Distribution ──
    section("📏 Message Length Distribution")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        fig_ml = px.histogram(
            df, x="msg_length", nbins=40, color_discrete_sequence=["#3b82f6"],
            labels={"msg_length": "Characters", "count": "Messages"}
        )
        fig_ml.update_layout(**PLOT_LAYOUT, title="Character Length Distribution")
        st.plotly_chart(fig_ml, use_container_width=True)
    with col_l2:
        fig_wl = px.histogram(
            df, x="word_count", nbins=30, color_discrete_sequence=["#8b5cf6"],
            labels={"word_count": "Words", "count": "Messages"}
        )
        fig_wl.update_layout(**PLOT_LAYOUT, title="Word Count Distribution")
        st.plotly_chart(fig_wl, use_container_width=True)

    # ── Response Chain ──
    section("🔗 Response Chain — Who Replies to Whom")
    if "datetime" in df.columns:
        pairs = analytics.get_response_pairs(df).head(15)
        if not pairs.empty:
            fig_p = px.bar(
                pairs, x="count",
                y=pairs["user"] + " → " + pairs["next_user"],
                orientation="h",
                color="count", color_continuous_scale="Oranges",
                labels={"count": "Exchanges", "y": "Pair"}
            )
            fig_p.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False,
                                yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_p, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Author Prediction
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🤖 Author Prediction":
    st.markdown("## 🤖 Author Prediction")
    st.markdown(
        "<small style='color:#64748b'>Type any message to predict the most likely author — powered by TF-IDF + Random Forest.</small>",
        unsafe_allow_html=True
    )

    col_inp, col_res = st.columns([1.2, 1])

    with col_inp:
        section("✍️ Enter a Message")
        user_msg = st.text_area(
            "Message", height=140,
            placeholder="e.g. 'Aaj class cancel hai kya?'",
            label_visibility="collapsed"
        )

        # Feature hints
        hour_input = st.slider("Hour sent (0–23):", 0, 23, 12)
        day_input  = st.selectbox("Day of week:", ["Monday","Tuesday","Wednesday",
                                                   "Thursday","Friday","Saturday","Sunday"])

        predict_btn = st.button("🔍 Predict Author", use_container_width=True,
                                type="primary")

    with col_res:
        section("🎯 Prediction Result")
        if predict_btn:
            if not user_msg.strip():
                st.warning("Please enter a message first.")
            elif vectorizer is None or label_encoder is None:
                st.error(
                    "⚠️ Model artifacts not found (`models/model.pkl`, `models/vectorizer.pkl`, "
                    "`models/label_encoder.pkl`). Train the model using `notebooks/03_modelling.ipynb` first."
                )
            elif model is None:
                # Fallback: TF-IDF cosine similarity to find closest training message
                st.info("🔄 Full model not loaded — using TF-IDF similarity fallback.")
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    vec_query = vectorizer.transform([user_msg])
                    vec_all   = vectorizer.transform(df["message"].astype(str).tolist())
                    sims = cosine_similarity(vec_query, vec_all).flatten()
                    top_idx = np.argsort(sims)[::-1][:5]
                    top_users = df.iloc[top_idx]["user"].value_counts()
                    predicted_user = top_users.idxmax()
                    confidence     = round(sims[top_idx[0]], 3)

                    st.markdown(f"""
                    <div class="pred-box">
                        <div style="color:#94a3b8;font-size:.85rem;margin-bottom:.5rem">PREDICTED AUTHOR</div>
                        <div class="pred-user">{predicted_user}</div>
                        <div class="pred-conf">Similarity Score: {confidence:.3f}</div>
                        <div style="color:#64748b;font-size:.75rem;margin-top:.8rem">(Cosine similarity fallback)</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                try:
                    from scipy.sparse import hstack, csr_matrix
                    day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                               "Friday":4,"Saturday":5,"Sunday":6}

                    text_vec  = vectorizer.transform([user_msg])
                    num_feats = csr_matrix([[
                        hour_input,
                        day_map.get(day_input, 0),
                        len(user_msg),
                        len(user_msg.split())
                    ]])
                    X_pred = hstack([text_vec, num_feats])

                    pred       = model.predict(X_pred)[0]
                    probs      = model.predict_proba(X_pred)[0]
                    pred_user  = label_encoder.inverse_transform([pred])[0]
                    confidence = float(np.max(probs))

                    # Confidence bar colour
                    conf_color = "#22c55e" if confidence > 0.6 else ("#f59e0b" if confidence > 0.35 else "#ef4444")

                    st.markdown(f"""
                    <div class="pred-box">
                        <div style="color:#94a3b8;font-size:.85rem;margin-bottom:.5rem">PREDICTED AUTHOR</div>
                        <div class="pred-user">{pred_user}</div>
                        <div class="pred-conf">Confidence: <span style="color:{conf_color};font-weight:700">{confidence*100:.1f}%</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Top-5 probabilities
                    st.markdown("<br>", unsafe_allow_html=True)
                    section("Top Candidates")
                    top5_idx   = np.argsort(probs)[::-1][:5]
                    top5_users = label_encoder.inverse_transform(top5_idx)
                    top5_probs = probs[top5_idx]
                    prob_df = pd.DataFrame({"User": top5_users, "Probability": top5_probs})
                    fig_p = px.bar(
                        prob_df, x="Probability", y="User", orientation="h",
                        color="Probability", color_continuous_scale="Blues",
                        range_x=[0, 1]
                    )
                    fig_p.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False,
                                        height=220, margin=dict(l=10, r=10, t=10, b=10),
                                        yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_p, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.markdown("""
            <div style="background:#1e293b;border:1px dashed #334155;border-radius:12px;
                        padding:2rem;text-align:center;color:#64748b;margin-top:1rem">
                <div style="font-size:2rem">🤖</div>
                <div style="margin-top:.5rem">Enter a message and click <b>Predict Author</b></div>
                <div style="font-size:.8rem;margin-top:.5rem">Random Forest + TF-IDF with behavioral features</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Model Info ──
    section("📐 Model Architecture")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.markdown(kpi("Algorithm",    "Random Forest",   "🌲"), unsafe_allow_html=True)
    col_m2.markdown(kpi("Features",     "TF-IDF + Behavioral", "🔢"), unsafe_allow_html=True)
    col_m3.markdown(kpi("Output",       "User Classification", "🎯"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    <b>Feature Engineering Pipeline</b><br>
    <span style="color:#94a3b8">
    ① Text → TF-IDF vectorization (top N-grams) &nbsp;|&nbsp;
    ② Behavioral: hour, day-of-week, message length, word count &nbsp;|&nbsp;
    ③ Combined sparse matrix → Random Forest classifier &nbsp;|&nbsp;
    ④ Probability output with label decoding via LabelEncoder
    </span>
    </div>
    """, unsafe_allow_html=True)

    insight("Each user leaves a distinct <b>linguistic fingerprint</b> — vocabulary, message style, and timing patterns — that the Random Forest exploits for attribution.")
    insight("Low confidence predictions (&lt;35%) indicate either very short messages or users with insufficient training samples (&lt;5 messages).")
