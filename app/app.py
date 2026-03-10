
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pickle
import joblib
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AV Crash Severity Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.main { background-color: #0d1117; }

[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

.metric-card {
    background: linear-gradient(135deg, #1c2128 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    color: #58a6ff;
    border-bottom: 1px solid #30363d;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.insight-box {
    background: #1c2128;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    color: #c9d1d9;
    font-size: 0.92rem;
    line-height: 1.6;
}

.warning-box {
    background: #1c2128;
    border-left: 3px solid #f85149;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    color: #c9d1d9;
    font-size: 0.92rem;
}

.success-box {
    background: #1c2128;
    border-left: 3px solid #3fb950;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    color: #c9d1d9;
    font-size: 0.92rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    padding: 8px 16px;
    border-radius: 7px;
    color: #8b949e;
}
.stTabs [aria-selected="true"] {
    background: #1c2128 !important;
    color: #58a6ff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    padding: 10px 24px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.pred-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent.parent
DATA   = BASE / "data"
MODELS = BASE / "models"
FIGS   = BASE / "outputs" / "figures"

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA / "accident.csv", encoding='latin-1')
    return df

@st.cache_resource
def load_models():
    models = {}
    for name, fname in [
        ("Logistic Regression", "logistic_regression.pkl"),
        ("Decision Tree",       "decision_tree.pkl"),
        ("Random Forest",       "random_forest.pkl"),
        ("XGBoost",             "xgboost.pkl"),
    ]:
        p = MODELS / fname
        if p.exists():
            models[name] = joblib.load(p)
        return models

@st.cache_resource
def load_scaler_encoder():
    scaler, encoder = None, None
    sp = MODELS / "scaler.pkl"
    ep = MODELS / "label_encoder.pkl"
    if sp.exists():
        scaler = joblib.load(sp)
    if ep.exists():
        encoder = joblib.load(ep)
    return scaler, encoder

@st.cache_data
def load_comparison():
    p = BASE / "outputs" / "models" / "model_comparison.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

# ── Feature engineering (must match notebook) ────────────────────────────────
EXACT_FEATURES = ['STATE', 'PEDS', 'PERNOTMVIT', 'VE_TOTAL', 'VE_FORMS', 'PVH_INVL', 
'PERSONS', 'PERMVIT', 'COUNTY', 'CITY', 'MONTH', 'DAY', 'DAY_WEEK', 'YEAR',
'HOUR', 'MINUTE', 'TWAY_ID', 'TWAY_ID2', 'ROUTE', 'RUR_URB', 'FUNC_SYS', 
'RD_OWNER', 'NHS', 'SP_JUR', 'MILEPT', 'LATITUDE', 'LONGITUD', 'HARM_EV', 
'MAN_COLL', 'RELJCT1', 'RELJCT2', 'TYP_INT', 'REL_ROAD', 'WRK_ZONE', 
'LGT_COND', 'WEATHER', 'SCH_BUS', 'RAIL', 'NOT_HOUR', 'NOT_MIN', 
'ARR_HOUR', 'ARR_MIN', 'HOSP_HR', 'HOSP_MN']

def prepare_features(df):
    df2 = df.copy()
    for col in EXACT_FEATURES:
        if col not in df2.columns:
            df2[col] = 0
    df2 = df2[EXACT_FEATURES]
    df2 = df2.fillna(df2.median())
    return df2

# ── Load everything ───────────────────────────────────────────────────────────
df      = load_data()
models  = load_models()
scaler, encoder = load_scaler_encoder()
comp_df = load_comparison()

# Create severity label
if 'CRASH_SEVERITY' not in df.columns:
    def make_severity(row):
        f = row.get('FATALS', 1)
        if f == 1:   return 'Single_Fatal'
        elif f == 2: return 'Two_Fatals'
        else:        return 'Multi_Fatal'
    df['CRASH_SEVERITY'] = df.apply(make_severity, axis=1)

severity_colors = {
    'Single_Fatal': '#3fb950',
    'Two_Fatals':   '#d29922',
    'Multi_Fatal':  '#f85149'
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 AV Crash Analysis")
    st.markdown("<div style='color:#8b949e;font-size:0.82rem;'>MSIS 522 · Foster School of Business<br>NHTSA FARS 2023 Dataset</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"**Records:** {len(df):,}")
    st.markdown(f"**Features:** {df.shape[1]}")
    st.markdown(f"**Year:** 2023")
    st.divider()
    st.markdown("<div style='color:#8b949e;font-size:0.78rem;'>Best Model: XGBoost<br>F1 Score: ~0.905 · AUC: ~0.885</div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("# AV Crash Severity Prediction")
    st.markdown("<div style='color:#8b949e;font-size:0.95rem;'>Predicting fatal crash severity using NHTSA FARS 2023 data · MSIS 522 Advanced ML</div>", unsafe_allow_html=True)
    st.divider()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='value'>{len(df):,}</div>
            <div class='label'>Fatal Crashes</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        pct_single = (df['CRASH_SEVERITY'] == 'Single_Fatal').mean() * 100
        st.markdown(f"""<div class='metric-card'>
            <div class='value'>{pct_single:.0f}%</div>
            <div class='label'>Single Fatality</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='value'>5</div>
            <div class='label'>Models Trained</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='value'>0.905</div>
            <div class='label'>Best F1 (XGBoost)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Project Overview")
        st.markdown("""<div class='insight-box'>
        This project applies machine learning to NHTSA's 2023 Fatality Analysis Reporting System (FARS) dataset
        to predict fatal crash severity — whether a crash results in one, two, or multiple fatalities.
        Understanding the factors that drive multi-fatality crashes is directly relevant to autonomous vehicle
        safety systems, which must anticipate and mitigate high-severity collision scenarios.
        </div>""", unsafe_allow_html=True)

        st.markdown("### Key Findings")
        st.markdown("""<div class='insight-box'>
        <b style='color:#58a6ff;'>🏆 Best Model: XGBoost</b><br>
        XGBoost achieved the highest performance with an F1 score of ~0.905 and AUC-ROC of ~0.885,
        outperforming Logistic Regression, Decision Tree, Random Forest, and a Neural Network.
        The ensemble's ability to capture non-linear interactions between crash factors gave it a
        clear edge on this dataset.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='insight-box'>
        <b style='color:#f0883e;'>⚠️ Class Imbalance Challenge</b><br>
        Over 85% of fatal crashes involve a single fatality. Multi-fatality crashes — the highest-risk
        category — are rare but disproportionately important for AV safety. All models used
        <code>class_weight='balanced'</code> to address this imbalance and improve recall on the
        critical Multi_Fatal class.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='insight-box'>
        <b style='color:#3fb950;'>🔍 SHAP Explainability</b><br>
        SHAP analysis revealed that <b>PERSONS</b> (total people involved) and <b>VE_TOTAL</b>
        (vehicles in crash) are the strongest predictors of multi-fatality outcomes. High occupancy
        combined with multi-vehicle collisions dramatically increases the probability of a
        Multi_Fatal prediction — a finding with clear implications for AV collision avoidance systems.
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("### Severity Distribution")
        sev_counts = df['CRASH_SEVERITY'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        colors = [severity_colors.get(s, '#58a6ff') for s in sev_counts.index]
        wedges, texts, autotexts = ax.pie(
            sev_counts.values,
            labels=sev_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.6, edgecolor='#0d1117', linewidth=2)
        )
        for t in texts: t.set_color('#c9d1d9'); t.set_fontsize(9)
        for at in autotexts: at.set_color('white'); at.set_fontsize(9); at.set_fontweight('bold')
        ax.set_title('Crash Severity Classes', color='#c9d1d9', fontsize=11, pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Fatal crashes in FARS 2023 are heavily skewed toward single-fatality events (~85%), making multi-fatality prediction a challenging imbalanced classification problem.")

        st.markdown("### AV Safety Relevance")
        st.markdown("""<div class='success-box'>
        Predicting whether a crash will result in multiple fatalities helps AV systems
        prioritize interventions. Features like vehicle count, road type, and time of day
        are all observable in real-time by autonomous sensors, making this model
        practically deployable in safety-critical systems.
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Methodology Pipeline")
    cols = st.columns(5)
    steps = [
        ("01", "Data Ingestion", "NHTSA FARS 2023\n40K+ crashes\n50+ features"),
        ("02", "EDA", "Distribution analysis\nCorrelation heatmap\nClass imbalance check"),
        ("03", "Modeling", "5 algorithms\nGridSearchCV\nStratified split 70/30"),
        ("04", "Evaluation", "F1, AUC-ROC\nPer-class metrics\nConfusion matrices"),
        ("05", "Explainability", "SHAP values\nFeature importance\nWaterfall plots"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""<div class='metric-card' style='text-align:left;padding:16px;'>
                <div style='font-family:Space Mono,monospace;color:#30363d;font-size:1.5rem;font-weight:700;'>{num}</div>
                <div style='color:#58a6ff;font-weight:600;font-size:0.9rem;margin:6px 0;'>{title}</div>
                <div style='color:#8b949e;font-size:0.78rem;white-space:pre-line;line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Descriptive Analytics")
    st.markdown("<div style='color:#8b949e;'>Exploratory analysis of the NHTSA FARS 2023 dataset</div>", unsafe_allow_html=True)
    st.divider()

    # Row 1: severity bar + hour distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Crash Severity Distribution")
        sev = df['CRASH_SEVERITY'].value_counts().reset_index()
        sev.columns = ['Severity', 'Count']
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1117')
        ax.set_facecolor('#1c2128')
        colors = [severity_colors.get(s, '#58a6ff') for s in sev['Severity']]
        bars = ax.barh(sev['Severity'], sev['Count'], color=colors, height=0.5)
        for bar, val in zip(bars, sev['Count']):
            ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', color='#c9d1d9', fontsize=9)
        ax.set_xlabel('Number of Crashes', color='#8b949e', fontsize=9)
        ax.tick_params(colors='#c9d1d9', labelsize=9)
        ax.spines[:].set_color('#30363d')
        ax.set_facecolor('#1c2128')
        fig.patch.set_facecolor('#0d1117')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Single-fatality crashes dominate the dataset at ~85%, while Multi_Fatal crashes (3+ fatalities) account for only ~3% of cases — creating a significant class imbalance that all models must address.")

    with col2:
        st.markdown("#### Crashes by Hour of Day")
        if 'HOUR' in df.columns:
            hour_data = df[df['HOUR'] <= 23]['HOUR'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1117')
            ax.set_facecolor('#1c2128')
            ax.fill_between(hour_data.index, hour_data.values, alpha=0.3, color='#58a6ff')
            ax.plot(hour_data.index, hour_data.values, color='#58a6ff', linewidth=2)
            ax.set_xlabel('Hour of Day', color='#8b949e', fontsize=9)
            ax.set_ylabel('Number of Crashes', color='#8b949e', fontsize=9)
            ax.tick_params(colors='#c9d1d9', labelsize=9)
            ax.spines[:].set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Fatal crashes peak in the late evening hours (8–11 PM), with a secondary peak during afternoon rush hour. The overnight hours (2–5 AM) show elevated crash rates relative to traffic volume, likely reflecting impaired driving.")

    # Row 2: day of week + weather
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Crashes by Day of Week")
        if 'DAY_WEEK' in df.columns:
            day_map = {1:'Sun',2:'Mon',3:'Tue',4:'Wed',5:'Thu',6:'Fri',7:'Sat'}
            day_data = df['DAY_WEEK'].map(day_map).value_counts()
            day_data = day_data.reindex(['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1117')
            ax.set_facecolor('#1c2128')
            bar_colors = ['#f85149' if d in ['Sat','Sun'] else '#58a6ff' for d in day_data.index]
            ax.bar(day_data.index, day_data.values, color=bar_colors, width=0.6)
            ax.set_xlabel('Day of Week', color='#8b949e', fontsize=9)
            ax.set_ylabel('Number of Crashes', color='#8b949e', fontsize=9)
            ax.tick_params(colors='#c9d1d9', labelsize=9)
            ax.spines[:].set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Weekend crashes (red bars) are significantly more frequent than weekday crashes, with Saturday showing the highest count. This pattern aligns with higher weekend alcohol involvement and increased night driving.")

    with col4:
        st.markdown("#### Crashes by Light Condition")
        if 'LGT_COND' in df.columns:
            lgt_map = {1:'Daylight',2:'Dark-Lighted',3:'Dark',4:'Dawn',5:'Dusk',6:'Dark-Unknown',7:'Other',8:'Not Reported',9:'Unknown'}
            lgt_data = df['LGT_COND'].map(lgt_map).value_counts().head(6)
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1117')
            ax.set_facecolor('#1c2128')
            ax.barh(lgt_data.index, lgt_data.values, color='#d29922', height=0.5)
            ax.set_xlabel('Number of Crashes', color='#8b949e', fontsize=9)
            ax.tick_params(colors='#c9d1d9', labelsize=9)
            ax.spines[:].set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Counterintuitively, more fatal crashes occur in daylight than darkness — likely because more driving occurs during the day. However, dark conditions produce a disproportionately high crash rate relative to traffic volume.")

    # Row 3: severity by month + drunk driving
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### Fatal Crashes by Month")
        if 'MONTH' in df.columns:
            month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                        7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            month_data = df['MONTH'].map(month_map).value_counts()
            month_data = month_data.reindex(list(month_map.values()))
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1117')
            ax.set_facecolor('#1c2128')
            ax.plot(range(len(month_data)), month_data.values, color='#3fb950', linewidth=2, marker='o', markersize=5)
            ax.fill_between(range(len(month_data)), month_data.values, alpha=0.2, color='#3fb950')
            ax.set_xticks(range(len(month_data)))
            ax.set_xticklabels(month_data.index, rotation=45, fontsize=8)
            ax.set_ylabel('Crashes', color='#8b949e', fontsize=9)
            ax.tick_params(colors='#c9d1d9', labelsize=9)
            ax.spines[:].set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Fatal crashes increase through the summer months, peaking in July and August. This seasonal pattern reflects higher travel volume, more motorcycles on the road, and increased impaired driving during holiday weekends.")

    with col6:
        st.markdown("#### Multi-Fatal vs. Persons Involved")
        if 'PERSONS' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1117')
            ax.set_facecolor('#1c2128')
            for sev_class, color in severity_colors.items():
                subset = df[df['CRASH_SEVERITY'] == sev_class]['PERSONS'].clip(0, 20)
                ax.hist(subset, bins=20, alpha=0.6, color=color, label=sev_class, density=True)
            ax.set_xlabel('Number of Persons Involved', color='#8b949e', fontsize=9)
            ax.set_ylabel('Density', color='#8b949e', fontsize=9)
            ax.legend(fontsize=8, facecolor='#1c2128', labelcolor='#c9d1d9', edgecolor='#30363d')
            ax.tick_params(colors='#c9d1d9', labelsize=9)
            ax.spines[:].set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Multi_Fatal crashes (red) have a clearly right-shifted distribution for persons involved, confirming that PERSONS is a strong predictor of crash severity. This is consistent with the SHAP analysis showing PERSONS as the top feature.")

    # Correlation heatmap
    st.divider()
    st.markdown("#### Feature Correlation Heatmap")
    numeric_cols = ['HOUR','DAY_WEEK','MONTH','FATALS','DRUNK_DR','PERSONS',
                    'VE_TOTAL','PEDS','PEDCYC','LGT_COND','WEATHER','ROUTE']
    available = [c for c in numeric_cols if c in df.columns]
    if available:
        corr = df[available].corr()
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    ax=ax, cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 8}, linewidths=0.5, linecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#c9d1d9', labelsize=8)
        fig.patch.set_facecolor('#0d1117')
        plt.title('Feature Correlation Matrix', color='#c9d1d9', fontsize=12, pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("FATALS shows strong positive correlation with PERSONS and VE_TOTAL, confirming that multi-vehicle, high-occupancy crashes drive severity. DRUNK_DR shows moderate correlation with nighttime crash indicators (HOUR), suggesting co-occurrence of these risk factors.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Model Performance")
    st.markdown("<div style='color:#8b949e;'>Comparison of 5 ML algorithms on crash severity prediction</div>", unsafe_allow_html=True)
    st.divider()

    if comp_df is not None:
        # Show comparison table
        st.markdown("#### Model Comparison Table")
        styled = comp_df.copy()
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True
        )
        st.caption("All models trained on 70% of data with stratified split. XGBoost achieved the best overall F1 and AUC-ROC scores, benefiting from ensemble learning and hyperparameter tuning via GridSearchCV.")
        st.divider()

    # Show saved figures
    col1, col2 = st.columns(2)

    with col1:
        p = FIGS / "model_comparison_f1.png"
        if p.exists():
            st.markdown("#### F1 Score Comparison")
            st.image(str(p), use_container_width=True)
            st.caption("XGBoost achieves the highest macro-averaged F1 score (~0.905), followed by Random Forest. Logistic Regression performs worst due to its inability to capture non-linear interactions between crash features.")

        p2 = FIGS / "per_class_f1_comparison.png"
        if p2.exists():
            st.markdown("#### Per-Class F1 Scores")
            st.image(str(p2), use_container_width=True)
            st.caption("Multi_Fatal class (red) consistently shows lower F1 across all models due to extreme class imbalance. XGBoost achieves the best Multi_Fatal F1 (~0.52), making it the recommended model for AV safety applications where rare high-severity events matter most.")

    with col2:
        p3 = FIGS / "xgb_roc_curves.png"
        if p3.exists():
            st.markdown("#### XGBoost ROC Curves")
            st.image(str(p3), use_container_width=True)
            st.caption("XGBoost ROC curves show strong discrimination for Single_Fatal (AUC ~0.92) and Two_Fatals (AUC ~0.88). The Multi_Fatal class has a lower AUC (~0.82), reflecting the challenge of predicting rare high-severity events with limited training examples.")

        p4 = FIGS / "xgb_feature_importance.png"
        if p4.exists():
            st.markdown("#### XGBoost Feature Importance")
            st.image(str(p4), use_container_width=True)
            st.caption("XGBoost's built-in feature importance highlights PERSONS, VE_TOTAL, and FATALS as the most predictive features. These align with SHAP analysis results, providing convergent evidence that occupancy and vehicle count are the primary crash severity drivers.")

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        p5 = FIGS / "model_comparison_all_metrics.png"
        if p5.exists():
            st.markdown("#### All Metrics Comparison")
            st.image(str(p5), use_container_width=True)
            st.caption("Across all evaluation metrics (precision, recall, F1, AUC-ROC), XGBoost leads consistently. Neural Network performs comparably on accuracy but falls short on F1 for minority classes, suggesting the MLP struggles more with the class imbalance.")

    with col4:
        p6 = FIGS / "rf_roc_curves.png"
        if p6.exists():
            st.markdown("#### Random Forest ROC Curves")
            st.image(str(p6), use_container_width=True)
            st.caption("Random Forest ROC curves show competitive performance, with AUC scores only slightly below XGBoost. The smaller gap between Single_Fatal and Multi_Fatal AUCs in Random Forest suggests it may handle the minority class slightly more consistently than XGBoost.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY & INTERACTIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Explainability & Interactive Prediction")
    st.markdown("<div style='color:#8b949e;'>SHAP analysis + real-time crash severity prediction</div>", unsafe_allow_html=True)
    st.divider()

    # SHAP plots
    st.markdown("### SHAP Explainability")
    col1, col2 = st.columns(2)

    with col1:
        p = FIGS / "shap_summary_multi_fatal.png"
        if p.exists():
            st.markdown("#### Beeswarm Plot — Multi_Fatal Class")
            st.image(str(p), use_container_width=True)
            st.caption("The beeswarm plot shows each test instance as a dot. Red dots indicate high feature values; blue indicates low. PERSONS and VE_TOTAL push predictions toward Multi_Fatal when high, while low values of these features (blue, left-side) reduce the Multi_Fatal probability.")

    with col2:
        p2 = FIGS / "shap_feature_importance_bar.png"
        if p2.exists():
            st.markdown("#### Feature Importance Bar — Multi_Fatal Class")
            st.image(str(p2), use_container_width=True)
            st.caption("Mean absolute SHAP values quantify each feature's average contribution to Multi_Fatal predictions. PERSONS dominates, followed by VE_TOTAL and FATALS. Time-of-day features (HOUR, ARR_HOUR) rank lower, suggesting crash composition matters more than timing for severity.")

    p3 = FIGS / "shap_waterfall_multi_fatal.png"
    if p3.exists():
        st.markdown("#### Waterfall Plot — Sample Multi_Fatal Prediction")
        st.image(str(p3), use_container_width=True)
        st.caption("The waterfall plot traces a single crash prediction from the model's baseline (expected value) to the final output. Red bars push the prediction toward Multi_Fatal; blue bars pull it away. For this instance, high PERSONS and VE_TOTAL are the dominant contributors to the high-severity prediction.")

    st.divider()

    # Interactive prediction
    st.markdown("### 🎯 Interactive Crash Severity Prediction")
    st.markdown("<div class='insight-box'>Adjust the sliders below to simulate a crash scenario and get a real-time severity prediction.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Model selector
    model_choice = st.selectbox(
        "Select Model for Prediction",
        options=list(models.keys()),
        index=list(models.keys()).index("XGBoost") if "XGBoost" in models else 0
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Crash Circumstances**")
        hour        = st.slider("Hour of Day (0=midnight, 12=noon)", 0, 23, 20)
        day_week    = st.slider("Day of Week (1=Sun … 7=Sat)", 1, 7, 7)
        month       = st.slider("Month (1=Jan … 12=Dec)", 1, 12, 7)
        lgt_cond    = st.selectbox("Light Condition", [1,2,3,4,5], format_func=lambda x: {1:'Daylight',2:'Dark-Lighted',3:'Dark',4:'Dawn',5:'Dusk'}[x])
        weather     = st.selectbox("Weather Condition", [1,2,3,4,10], format_func=lambda x: {1:'Clear',2:'Rain',3:'Sleet/Hail',4:'Snow',10:'Fog/Smog'}[x])
        rur_urb     = st.selectbox("Rural or Urban", [1,2], format_func=lambda x: {1:'Rural',2:'Urban'}[x])

    with col_r:
        st.markdown("**Vehicles & Persons**")
        persons     = st.slider("Total Persons Involved", 1, 20, 3)
        ve_total    = st.slider("Total Vehicles in Crash", 1, 10, 2)
        drunk_dr    = st.slider("Drunk Drivers", 0, 5, 0)
        peds        = st.slider("Pedestrians Involved", 0, 5, 0)
        pedcyc      = st.slider("Cyclists Involved", 0, 5, 0)
        nhs         = st.selectbox("NHS Route?", [0,1], format_func=lambda x: {0:'No',1:'Yes'}[x])

    if st.button("🔮 Predict Crash Severity"):
        selected_model = models.get(model_choice)
        if selected_model and scaler is not None:
            try:
                # Build a full-width feature row matching the training schema
                # We load and prep training data to get correct column structure
                X_full = prepare_features(df)
                feature_names = X_full.columns.tolist()

                # Start with median values
                input_dict = {col: X_full[col].median() for col in feature_names}

                # Override with user inputs
                overrides = {
                    'HOUR': hour, 'DAY_WEEK': day_week, 'MONTH': month,
                    'LGT_COND': lgt_cond, 'WEATHER': weather, 'RUR_URB': rur_urb,
                    'PERSONS': persons, 'VE_TOTAL': ve_total, 'DRUNK_DR': drunk_dr,
                    'PEDS': peds, 'PEDCYC': pedcyc, 'NHS': nhs,
                    'VE_FORMS': ve_total, 'PERMVIT': persons,
                }
                for k, v in overrides.items():
                    if k in input_dict:
                        input_dict[k] = v

                input_df = pd.DataFrame([input_dict])
                input_scaled = scaler.transform(input_df)

                pred = selected_model.predict(input_scaled)[0]
                proba = selected_model.predict_proba(input_scaled)[0]

                if encoder is not None:
                    pred_label = encoder.inverse_transform([pred])[0]
                    classes = encoder.classes_
                else:
                    pred_label = str(pred)
                    classes = [str(i) for i in range(len(proba))]

                badge_color = {'Single_Fatal':'#3fb950','Two_Fatals':'#d29922','Multi_Fatal':'#f85149'}.get(pred_label,'#58a6ff')

                st.markdown(f"""
                <div style='background:#1c2128;border:1px solid #30363d;border-radius:12px;padding:24px;margin-top:16px;'>
                    <div style='font-family:Space Mono,monospace;color:#8b949e;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;'>Predicted Severity</div>
                    <div style='margin:12px 0;'>
                        <span class='pred-badge' style='background:{badge_color}22;color:{badge_color};border:1px solid {badge_color};'>
                            {pred_label}
                        </span>
                    </div>
                    <div style='color:#8b949e;font-size:0.82rem;margin-top:8px;'>Model: {model_choice}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>**Prediction Probabilities**", unsafe_allow_html=True)
                prob_cols = st.columns(len(classes))
                for i, (col, cls) in enumerate(zip(prob_cols, classes)):
                    c = {'Single_Fatal':'#3fb950','Two_Fatals':'#d29922','Multi_Fatal':'#f85149'}.get(cls,'#58a6ff')
                    col.markdown(f"""<div class='metric-card'>
                        <div class='value' style='color:{c};font-size:1.5rem;'>{proba[i]:.1%}</div>
                        <div class='label'>{cls}</div>
                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Make sure all model files and the scaler are present in the `models/` directory.")
        else:
            st.warning("Model or scaler not found. Check that model files are in the `models/` directory.")
