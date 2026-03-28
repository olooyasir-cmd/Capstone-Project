"""
app.py — Mobile App Success Predictor & Recommender
Run: streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from utils import predict_new_app, recommend_apps

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="App Success Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DARK-THEME CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1c2333;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 18px;
}
/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 2.2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
/* ── Section cards ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
.rating-badge {
    display: inline-block;
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    font-size: 2.4rem;
    font-weight: 700;
    padding: 14px 30px;
    border-radius: 16px;
    margin: 10px 0;
}
.warn-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    font-size: 2.4rem;
    font-weight: 700;
    padding: 14px 30px;
    border-radius: 16px;
    margin: 10px 0;
}
h1 { color: #a78bfa !important; }
h2, h3 { color: #c4b5fd !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL BUNDLE
# ─────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    bundle_path = os.path.join(os.path.dirname(__file__), "model_bundle.pkl")
    with open(bundle_path, "rb") as f:
        return pickle.load(f)

bundle  = load_bundle()
rf      = bundle["rf"]
encoders = bundle["encoders"]
knn     = bundle["knn"]
scaler  = bundle["scaler"]
rec_features = bundle["rec_features"]
df      = bundle["df"]
mae     = bundle["mae"]
rmse    = bundle["rmse"]

categories     = sorted(encoders["le_cat"].classes_.tolist())
content_ratings = sorted(encoders["le_cr"].classes_.tolist())
genres         = sorted(encoders["le_genre"].classes_.tolist())

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:2.6rem; margin-bottom:0'>
    📱 Mobile App Success Predictor & Recommender
</h1>
<p style='text-align:center; color:#8b949e; margin-top:6px; font-size:1.05rem'>
    Powered by Random Forest · KNN Cosine Similarity · Google Play Store Dataset
</p>
<hr style='border-color:#30363d; margin:18px 0'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — MODEL METRICS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Performance")
    st.metric("Mean Absolute Error", f"{mae:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Apps", f"{len(df):,}")
    st.metric("Categories", f"{df['Category'].nunique()}")
    st.metric("Avg Rating", f"{df['Rating'].mean():.2f} / 5.0")
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.info(
        "**Prediction** uses a Random Forest Regressor "
        "trained on 8,000+ Play Store apps.\n\n"
        "**Recommendations** use KNN with cosine similarity "
        "across category, genre, price & size."
    )

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Predict Success", "🔍 Find Similar Apps", "📈 Data Insights"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab1:
    st.markdown("#### 🛠 Enter Your App Specifications")
    st.markdown(
        "<p style='color:#8b949e'>Fill in the details below to receive a predicted "
        "rating for your app concept before you launch.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**App Category**")
        category = st.selectbox("Category", categories, label_visibility="collapsed",
                                key="cat_sel")
        st.markdown("**Content Rating**")
        content_rating = st.selectbox("Content Rating", content_ratings,
                                      label_visibility="collapsed", key="cr_sel")

    with c2:
        st.markdown("**Genre**")
        genre = st.selectbox("Genre", genres, label_visibility="collapsed", key="gen_sel")
        st.markdown("**App Type**")
        app_type = st.selectbox("App Type", ["Free", "Paid"],
                                label_visibility="collapsed", key="type_sel")

    with c3:
        st.markdown("**Planned Size (MB)**")
        size_mb = st.slider("Size (MB)", 1.0, 100.0, 20.0, 0.5,
                            label_visibility="collapsed")
        if app_type == "Paid":
            st.markdown("**Price (USD $)**")
            price = st.number_input("Price", 0.99, 400.0, 2.99, 0.50,
                                    label_visibility="collapsed")
        else:
            price = 0.0
            st.markdown("**Price (USD $)**")
            st.info("Free app — price set to $0.00")

    st.markdown("")
    predict_btn = st.button("⚡ Predict My App Rating", use_container_width=True)

    if predict_btn:
        predicted = predict_new_app(
            rf, encoders, category, size_mb, price,
            content_rating, app_type, genre,
        )
        predicted = max(1.0, min(5.0, predicted))

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            badge_class = "rating-badge" if predicted >= 4.0 else "warn-badge"
            st.markdown(
                f"<p style='color:#8b949e; margin-bottom:4px'>Expected Rating</p>"
                f"<div class='{badge_class}'>⭐ {predicted:.1f} / 5.0</div>",
                unsafe_allow_html=True,
            )
            if predicted >= 4.3:
                st.success("🚀 Top-tier potential! Excellent prospects.")
            elif predicted >= 4.0:
                st.success("✅ Good rating expected. Strong launch ahead.")
            elif predicted >= 3.5:
                st.warning("⚠️ Average rating. Consider refining UX & features.")
            else:
                st.error("❌ Below average. Revisit category, pricing, or size.")

        with res_col2:
            # gauge chart
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#161b22")
            bar_color = "#10b981" if predicted >= 4.0 else "#f59e0b" if predicted >= 3.5 else "#ef4444"
            ax.barh(["Your App"], [predicted], color=bar_color, height=0.4)
            ax.barh(["Your App"], [5.0], color="#30363d", height=0.4)
            ax.barh(["Your App"], [predicted], color=bar_color, height=0.4)
            ax.set_xlim(0, 5)
            ax.set_xlabel("Rating", color="#8b949e")
            ax.tick_params(colors="#8b949e")
            ax.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
            ax.axvline(4.0, color="#a78bfa", linestyle="--", linewidth=1, label="Good threshold (4.0)")
            ax.legend(facecolor="#1c2333", edgecolor="#30363d", labelcolor="#e0e0e0",
                      fontsize=8, loc="lower right")
            ax.set_title("Rating Gauge", color="#c4b5fd", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # feature importance mini-chart
        st.markdown("#### 🧠 Feature Importance (What drives ratings?)")
        fi_df = pd.DataFrame({
            "Feature": ["Category", "Size (MB)", "Installs", "Price",
                        "Content Rating", "Type", "Genre", "Reviews"],
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#161b22")
        colors = ["#7c3aed" if fi > fi_df["Importance"].median() else "#4f46e5"
                  for fi in fi_df["Importance"]]
        ax2.barh(fi_df["Feature"], fi_df["Importance"], color=colors)
        ax2.tick_params(colors="#8b949e")
        ax2.set_xlabel("Importance Score", color="#8b949e")
        ax2.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
        ax2.set_title("Random Forest Feature Importances", color="#c4b5fd")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

# ══════════════════════════════════════════════
# TAB 2 — RECOMMEND
# ══════════════════════════════════════════════
with tab2:
    st.markdown("#### 🔍 Find Competing & Similar Apps")
    st.markdown(
        "<p style='color:#8b949e'>Enter an existing Play Store app name to discover "
        "the 5 most similar apps — your potential competition.</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        app_query = st.text_input(
            "App name", placeholder="e.g. Clash of Clans, Instagram, Spotify …",
            label_visibility="collapsed",
        )
    with col_b:
        rec_btn = st.button("🔎 Find Similar", use_container_width=True)

    # show sample app names
    with st.expander("📋 Browse available apps (sample)"):
        sample = df.sample(50, random_state=7)[["App", "Category", "Rating"]].reset_index(drop=True)
        st.dataframe(sample, use_container_width=True)

    if rec_btn and app_query.strip():
        with st.spinner("Running KNN cosine similarity …"):
            recs = recommend_apps(app_query.strip(), df, knn, scaler, rec_features)

        if "Error" in recs.columns:
            st.error(recs.iloc[0, 0])
        else:
            st.markdown(f"#### 🏆 Top 5 Apps Similar to **{app_query}**")
            # Style the dataframe
            def style_rating(v):
                if v >= 4.3: return "background-color:#064e3b; color:#6ee7b7"
                elif v >= 4.0: return "background-color:#1e3a5f; color:#93c5fd"
                elif v >= 3.5: return "background-color:#451a03; color:#fcd34d"
                return "background-color:#4c0519; color:#fca5a5"

            styled = (
                recs.style
                .format({"Price": "${:.2f}", "Installs": "{:,.0f}",
                         "Rating": "{:.1f}", "Similarity": "{:.3f}"})
                .applymap(style_rating, subset=["Rating"])
                .set_properties(**{"background-color": "#161b22", "color": "#e0e0e0",
                                   "border-color": "#30363d"})
            )
            st.dataframe(styled, use_container_width=True)

            # similarity bar chart
            fig3, ax3 = plt.subplots(figsize=(7, 3))
            fig3.patch.set_facecolor("#0e1117")
            ax3.set_facecolor("#161b22")
            bars = ax3.barh(recs["App"].str[:28], recs["Similarity"],
                            color=["#7c3aed", "#4f46e5", "#3b82f6", "#06b6d4", "#10b981"])
            ax3.set_xlim(0, 1)
            ax3.set_xlabel("Cosine Similarity Score", color="#8b949e")
            ax3.tick_params(colors="#8b949e")
            ax3.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
            ax3.set_title("Similarity Scores (KNN Cosine)", color="#c4b5fd")
            for bar, val in zip(bars, recs["Similarity"]):
                ax3.text(bar.get_width() - 0.04, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", ha="right", color="white", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

    elif rec_btn:
        st.warning("Please enter an app name.")

# ══════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("#### 📈 Exploratory Data Analysis")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Apps", f"{len(df):,}")
    metric_cols[1].metric("Free Apps", f"{(df['Type']=='Free').sum():,}")
    metric_cols[2].metric("Paid Apps", f"{(df['Type']=='Paid').sum():,}")
    metric_cols[3].metric("Avg Rating", f"{df['Rating'].mean():.2f}")

    st.markdown("---")

    # ── Plot 1: Reviews vs Ratings scatter ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Reviews vs Rating**")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        fig4.patch.set_facecolor("#0e1117")
        ax4.set_facecolor("#161b22")
        sample_df = df[df["Reviews"] < df["Reviews"].quantile(0.98)].sample(1500, random_state=1)
        ax4.scatter(np.log1p(sample_df["Reviews"]), sample_df["Rating"],
                    alpha=0.35, s=12, c="#7c3aed")
        ax4.set_xlabel("log(Reviews + 1)", color="#8b949e")
        ax4.set_ylabel("Rating", color="#8b949e")
        ax4.tick_params(colors="#8b949e")
        ax4.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
        ax4.set_title("Reviews vs Ratings (log scale)", color="#c4b5fd")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    with col2:
        st.markdown("**Rating Distribution**")
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        fig5.patch.set_facecolor("#0e1117")
        ax5.set_facecolor("#161b22")
        ax5.hist(df["Rating"].dropna(), bins=30, color="#4f46e5", edgecolor="#0e1117",
                 alpha=0.85)
        ax5.axvline(df["Rating"].mean(), color="#10b981", linestyle="--",
                    linewidth=1.5, label=f"Mean: {df['Rating'].mean():.2f}")
        ax5.set_xlabel("Rating", color="#8b949e")
        ax5.set_ylabel("Count", color="#8b949e")
        ax5.tick_params(colors="#8b949e")
        ax5.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
        ax5.legend(facecolor="#1c2333", edgecolor="#30363d", labelcolor="#e0e0e0")
        ax5.set_title("Rating Distribution", color="#c4b5fd")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    # ── Plot 2: Top categories by avg rating ──────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Avg Rating by Category**")
        cat_rating = (
            df.groupby("Category")["Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        fig6, ax6 = plt.subplots(figsize=(5, 5))
        fig6.patch.set_facecolor("#0e1117")
        ax6.set_facecolor("#161b22")
        cat_colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(cat_rating)))
        ax6.barh(cat_rating.index[::-1], cat_rating.values[::-1], color=cat_colors[::-1])
        ax6.set_xlabel("Avg Rating", color="#8b949e")
        ax6.tick_params(colors="#8b949e", labelsize=8)
        ax6.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
        ax6.set_title("Top 15 Categories by Rating", color="#c4b5fd")
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    with col4:
        st.markdown("**Free vs Paid — Rating Comparison**")
        fig7, ax7 = plt.subplots(figsize=(5, 5))
        fig7.patch.set_facecolor("#0e1117")
        ax7.set_facecolor("#161b22")
        free_ratings = df[df["Type"] == "Free"]["Rating"].dropna()
        paid_ratings = df[df["Type"] == "Paid"]["Rating"].dropna()
        ax7.hist(free_ratings, bins=25, alpha=0.65, label="Free", color="#4f46e5",
                 edgecolor="#0e1117")
        ax7.hist(paid_ratings, bins=25, alpha=0.65, label="Paid", color="#10b981",
                 edgecolor="#0e1117")
        ax7.set_xlabel("Rating", color="#8b949e")
        ax7.set_ylabel("Count", color="#8b949e")
        ax7.tick_params(colors="#8b949e")
        ax7.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
        ax7.legend(facecolor="#1c2333", edgecolor="#30363d", labelcolor="#e0e0e0")
        ax7.set_title("Free vs Paid Rating Distribution", color="#c4b5fd")
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()

    # ── Plot 3: Top genres by count ───────────────────────────────────
    st.markdown("**Top 12 Genres by App Count**")
    genre_counts = df["Genre_Clean"].value_counts().head(12)
    fig8, ax8 = plt.subplots(figsize=(10, 3.5))
    fig8.patch.set_facecolor("#0e1117")
    ax8.set_facecolor("#161b22")
    bar_colors = plt.cm.cool(np.linspace(0.2, 0.9, len(genre_counts)))
    ax8.bar(genre_counts.index, genre_counts.values, color=bar_colors)
    ax8.set_ylabel("Count", color="#8b949e")
    ax8.tick_params(colors="#8b949e", axis="x", rotation=30, labelsize=9)
    ax8.tick_params(colors="#8b949e", axis="y")
    ax8.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
    ax8.set_title("Most Common Genres on Google Play", color="#c4b5fd")
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()
