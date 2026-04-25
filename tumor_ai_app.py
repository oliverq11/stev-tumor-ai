import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import qrcode
from io import BytesIO

import os
st.write("Files in directory:", os.listdir("."))


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="STEV: Stochastic Tumor Response AI", layout="wide", page_icon="🧬")

# ============================================================
# CUSTOM CSS WITH FORCED WHITE BUTTON TEXT (ALL MODES)
# ============================================================
st.markdown("""
<style>
    /* Light mode background */
    .stApp { background-color: #f5f7fb; }
    h1, h2, h3, .stMarkdown, p, div, span, label {
        color: #212529;
    }
    h1 { color: #1e466e; font-family: 'Segoe UI', sans-serif; }
    .subtitle { color: #2c6e9e; font-size: 1.2rem; margin-bottom: 1rem; }
    .author { color: #6c757d; font-size: 0.9rem; margin-bottom: 2rem; }
    
    /* ===== BUTTONS: FORCE WHITE TEXT IN ALL SITUATIONS ===== */
    .stButton button,
    .stButton button:link,
    .stButton button:visited,
    .stButton button:active,
    .stButton button:focus,
    .stButton button:hover,
    .stButton > button,
    div[data-testid="stBaseButton-primary"] button,
    div[data-testid="stBaseButton-secondary"] button,
    button[kind="primary"],
    button[kind="secondary"] {
        background-color: #1e466e !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        transition: 0.2s !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #0f2e4a !important;
        color: white !important;
        transform: scale(1.02) !important;
    }
    .stButton button *,
    .stButton button span,
    .stButton button div {
        color: white !important;
    }
    
    .streamlit-expanderHeader { background-color: #e9ecef; border-radius: 8px; color: #212529; }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: bold; color: #1e466e; }
    [data-testid="stMetricLabel"] { color: #212529; }

    @media (prefers-color-scheme: dark) {
        .stApp, .main, .stAppViewContainer, .css-18e3th9, .css-1d391kg {
            background-color: #0a0a0a !important;
        }
        
        body, p, div, span, label, .stMarkdown, .stText, .stSelectbox label, 
        .stSlider label, .stMultiSelect label, .stTextInput label, 
        .stNumberInput label, .stDateInput label, .stTimeInput label,
        .stTextArea label, .stRadio label, .stCheckbox label,
        h1, h2, h3, h4, h5, h6, .subtitle, .author, .caption {
            color: #ffffff !important;
        }
        
        h1, .subtitle, .author { color: #ffffff !important; }
        
        .css-1lcbmhc, .css-1adrfps, .sidebar .stMarkdown, .sidebar p, .sidebar div {
            color: #ffffff !important;
        }
        
        .stButton button,
        .stButton button:link,
        .stButton button:visited,
        .stButton button:active,
        .stButton button:focus,
        .stButton button:hover,
        .stButton > button,
        div[data-testid="stBaseButton-primary"] button,
        div[data-testid="stBaseButton-secondary"] button,
        button[kind="primary"],
        button[kind="secondary"] {
            background-color: #2c6e9e !important;
            color: white !important;
        }
        .stButton button:hover {
            background-color: #1e466e !important;
            color: white !important;
        }
        .stButton button *,
        .stButton button span,
        .stButton button div {
            color: white !important;
        }
        
        .streamlit-expanderHeader {
            background-color: #1e1e1e !important;
            color: white !important;
        }
        .streamlit-expanderContent {
            background-color: #0a0a0a !important;
            color: white !important;
        }
        
        [data-testid="stMetricValue"] { color: #79c2ff !important; }
        [data-testid="stMetricLabel"] { color: #dddddd !important; }
        
        .stDataFrame, .stDataFrame div, .dataframe, .dataframe td, .dataframe th {
            background-color: #1e1e1e !important;
            color: white !important;
        }
        
        .plotly-graph-div .gtitle, .plotly-graph-div .xtitle, .plotly-graph-div .ytitle,
        .plotly-graph-div .legendtext, .plotly-graph-div .hovertext text,
        .plotly-graph-div .annotation-text, .plotly-graph-div .axis text,
        .plotly-graph-div .legend .traces .text {
            fill: #ffffff !important;
            color: #ffffff !important;
        }
        .plotly-graph-div .main-svg { background-color: #0a0a0a !important; }
        .plotly-graph-div .axis line, .plotly-graph-div .axis path { stroke: #888888 !important; }
        .plotly-graph-div .axis tick text { fill: #cccccc !important; }
        
        .stImage caption, .stImage figcaption { color: #cccccc !important; }
        
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
        }
        .stTabs [data-baseweb="tab-list"] button:hover { background-color: #2c6e9e !important; }
        
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1e1e1e !important;
            color: white !important;
        }
        
        .stSlider div[data-baseweb="slider"] div[role="slider"] {
            background-color: #2c6e9e !important;
        }
    }
</style>
""", unsafe_allow_html=True)

if 'disclaimer_shown' not in st.session_state:
    st.session_state.disclaimer_shown = False

st.title("🧬 STEV: Stochastic Tumor Evolution and Immunological Response")
st.markdown('<div class="subtitle">Lynch Syndrome Colorectal Tumors</div>', unsafe_allow_html=True)
st.markdown('<div class="author">Horatio Quinones / Sherry Johnson / et-al</div>', unsafe_allow_html=True)
st.markdown("""
### 🔍 What this app does

This tool uses a **stochastic model** (STEV) built on real clinical data from Lynch syndrome colorectal cancer patients treated with dostarlimab. It helps answer two questions:

1. **Given a tumor size at a specific week, what is the most likely underlying biology?**  
   (e.g., POLE, MLH1, MSH2, MSI-H, MSH6)

2. **Given a known biology, what range of tumor sizes is expected at a given week?**

Predictions are based on published clinical data and include 95% credible intervals to reflect uncertainty.
""")

# ============================================================
# EXPANDER 1: GROWTH CURVES (30mm and 60mm)
# ============================================================
with st.expander("📈 Tumor Growth & Immunotherapy Response (30mm & 60mm starting points)", expanded=False):
    st.markdown("""
    ### 📖 What these curves show
    
    Each plot traces the **complete tumor size trajectory** over time (weeks) for Lynch syndrome patients treated with dostarlimab immunotherapy:
    
    - **Growth phase (weeks 0–6)**: Tumor grows from a very small, barely detectable size until it reaches either **30 mm** or **60 mm**.
    - **Immunotherapy response phase (weeks 6–24)**: Dostarlimab treatment begins. The tumor **shrinks** continuously – the curve shows tumor size decreasing week by week.
    - **Cure plateau (week 12–24)**: Tumor size approaches a minimal residual level (approximately 1.1 mm).
    
    The two plots compare outcomes based on tumor size at treatment initiation (30 mm vs. 60 mm).
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("30mm.png", caption="**Plot 1:** Start size = 30 mm", use_container_width=True)
        st.caption("Tumor grows to 30 mm, then shrinks after immunotherapy. Shaded band = 90% credible interval.")
    with col2:
        st.image("60mm.png", caption="**Plot 2:** Start size = 60 mm", use_container_width=True)
        st.caption("Same treatment, but tumor starts larger. Compare the shrinkage trajectory.")
    
    st.markdown("""
    **Key insight:** Earlier detection (30 mm) leads to faster shrinkage to the cure plateau. The shaded bands represent uncertainty in the stochastic model.
    """)

# ============================================================
# EXPANDER 2: TWO-HIT DYNAMICS (6 plots)
# ============================================================
with st.expander("🕰️ Two‑Hit Dynamics: Incubation, Latency, Age at Detection & Risk", expanded=False):
    st.markdown("""
    ### 📖 What is "First Hit" and "Second Hit"?
    
    - **First hit (inherited mutation):** A person with Lynch syndrome is born with **one faulty copy** of an MMR gene (e.g., MLH1, MSH2) inherited from a parent. This alone does not cause cancer – it only creates a **predisposition**.
      
    - **Second hit (acquired mutation):** At some point later in life, the **second healthy copy** of that MMR gene is damaged or lost (due to random chance, environment, or aging). When this happens, the cell can no longer repair DNA mistakes, leading to microsatellite instability (MSI) and eventually **tumor formation**.
      
    - **Key insight:** The time from **birth to second hit** (Plot b) can be decades. After the second hit, the tumor must grow until it becomes detectable >1 mm (Plot a). This is why Lynch syndrome cancers typically appear in adulthood, not childhood.
    """)
    
    st.markdown("### Complete stochastic model output (6 plots)")
    
    # Row 1: Incubation and Latency (Plots a and b)
    col1, col2 = st.columns(2)
    with col1:
        st.image("incubation.png", caption="**Plot a:** Incubation (birth → second hit)", use_container_width=True)
        st.caption("Age at which the second hit occurs. Most occur between ages 30–55. This is why Lynch cancers appear in adulthood.")
    with col2:
        st.image("latency.png", caption="**Plot b:** Latency (second hit → detectable tumor >1 mm)", use_container_width=True)
        st.caption("Waiting time from the second hit until the tumor becomes detectable. Shorter latency = faster tumor growth.")
    
    # Row 2: Conditional plots (Plots c and d)
    col1, col2 = st.columns(2)
    with col1:
        st.image("detection_age_conditional.png", caption="**Plot c (Conditional):** Detection age distribution", use_container_width=True)
        st.caption("Given that a second hit has occurred, this shows the age at clinical detection.")
    with col2:
        st.image("probability_conditional.png", caption="**Plot d (Conditional):** Probability of detection by age", use_container_width=True)
        st.caption("Given a second hit, the cumulative probability that the tumor has been detected by a given age.")
    
    # Row 3: Unconditional plots (Plots c and d repeated)
    col1, col2 = st.columns(2)
    with col1:
        st.image("detection_age_unconditional.png", caption="**Plot c (Unconditional):** Detection age distribution", use_container_width=True)
        st.caption("For all Lynch patients (including those without a second hit), the age at clinical detection.")
    with col2:
        st.image("probability_unconditional.png", caption="**Plot d (Unconditional):** Probability of detection by age", use_container_width=True)
        st.caption("Overall probability that a Lynch patient will have a detected tumor by a given age (∼70–80% by age 70).")
    
    st.markdown("""
    ---
    ### 🔑 Conditional vs. Unconditional
    
    - **Conditional (Plots c & d, top row):** *"Given that you already had the second hit, what is the probability of detection by age X?"*
      
    - **Unconditional (Plots c & d, bottom row):** *"At birth, what is your overall chance of ever having a detected tumor by age X?"*
    
    The unconditional curves are always lower than the conditional curves because some Lynch patients never experience the second hit and therefore never develop cancer.
    
    *Distributions derived from STEV stochastic model with Lynch syndrome epidemiology.*
    """)

# ============================================================
# PARAMETERS (STEV + subgroup means)
# ============================================================
weeks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24]

mu_STEV = {
    0: 23.00, 1: 23.00, 2: 23.00, 3: 20.79, 4: 16.65, 5: 12.37,
    6: 8.61, 7: 5.77, 8: 3.84, 9: 2.64, 10: 1.92, 11: 1.51,
    12: 1.27, 13: 1.14, 14: 1.10, 15: 1.10, 16: 1.10, 17: 1.10,
    18: 1.10, 19: 1.10, 20: 1.10, 21: 1.10, 24: 1.10
}

sigma_STEV = {
    0: 0.00, 1: 0.00, 2: 0.00, 3: 2.00, 4: 2.50, 5: 2.80,
    6: 2.70, 7: 2.60, 8: 2.62, 9: 2.40, 10: 2.10, 11: 1.80,
    12: 1.50, 13: 1.20, 14: 0.90, 15: 0.70, 16: 0.50, 17: 0.40,
    18: 0.30, 19: 0.20, 20: 0.10, 21: 0.05, 24: 0.05
}

env_factor = 0.30
sigma_env = {w: sigma_STEV[w] * np.sqrt(env_factor) for w in weeks}
names = ['POLE', 'MLH1', 'MSH2', 'MSIH', 'MSH6']

S0 = 23.0
HR = {'POLE': 1.159, 'MLH1': 1.127, 'MSH2': 1.117, 'MSIH': 1.099, 'MSH6': 1.091}
means = {}
for w in weeks:
    means[w] = {}
    for name in names:
        m = S0 + HR[name] * (mu_STEV[w] - S0)
        means[w][name] = max(m, 0.01)

priors = {name: 1.0/5 for name in names}

def normal_pdf(x, mu, sigma):
    if sigma <= 0:
        return 1e-10
    return np.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

def predict_inverse(size, week):
    sigma = sigma_env[week]
    unnorm = {}
    for name in names:
        mu = means[week][name]
        like = normal_pdf(size, mu, sigma)
        unnorm[name] = like * priors[name]
    total = sum(unnorm.values())
    if total == 0:
        return {name: 0.2 for name in names}
    return {name: unnorm[name]/total for name in names}

def predict_forward(biology, week):
    mu = means[week][biology]
    sigma = sigma_env[week]
    ci_95 = (mu - 1.96*sigma, mu + 1.96*sigma)
    return mu, sigma, ci_95

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    app_url = "https://stev-tumor-ai-skrobcqyqyyz4sjpvqdqmh.streamlit.app/"
    qr = qrcode.QRCode(box_size=5, border=2)
    qr.add_data(app_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), width=150, caption="Scan to open on phone")
    
    st.markdown("### ℹ️ How to use")
    st.markdown("""
    - **🔍 Size → Biology:** Enter tumor size → get most likely biology.
    - **🔮 Biology → Size:** Select biology → get predicted size range.
    - **📈 Growth-Immunotherapy & Two-Hit Dynamics:** Click the expanders above to see all 8 plots.
    """)
    st.markdown("---")
    st.markdown("**STEV model** – Lynch Syndrome Colorectal Tumors")
    st.markdown("*Horatio Quinones / Sherry Johnson / et al*")

# ============================================================
# MAIN APP WITH TWO TABS
# ============================================================
tab1, tab2 = st.tabs(["🔍 Size → Biology", "🔮 Biology → Size"])

with tab1:
    col_left, col_right = st.columns(2)
    with col_left:
        week = st.selectbox("📅 Week", weeks, index=8)
    with col_right:
        size = st.slider("📏 Tumor size (mm)", min_value=0.0, max_value=30.0, value=1.4, step=0.1)

    if st.button("Predict Biology", use_container_width=True):
        with st.spinner("Computing probabilities..."):
            probs = predict_inverse(size, week)
        most_likely = max(probs, key=probs.get)

        col_a, col_b = st.columns(2)
        col_a.metric("🧬 Most likely biology", most_likely)
        col_b.metric("📊 Probability", f"{probs[most_likely]:.1%}")

        df = pd.DataFrame(list(probs.items()), columns=['Biology', 'Probability'])
        fig = px.bar(df, x='Biology', y='Probability', color='Biology',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title=f'Week {week}, size = {size} mm')
        fig.update_layout(yaxis_title='Posterior probability', xaxis_title='Biology')
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Detailed probabilities"):
            st.dataframe(df.style.format({'Probability': '{:.3f}'}))

        st.markdown("---")
        st.caption("⚠️ Disclaimer: For research & education only – not medical advice. Always consult your doctor.")

with tab2:
    col_left, col_right = st.columns(2)
    with col_left:
        week = st.selectbox("📅 Week", weeks, index=8, key="forward_week")
    with col_right:
        biology = st.selectbox("🧬 Biology", names, index=1)

    if st.button("Predict Size", use_container_width=True):
        with st.spinner("Calculating predicted size..."):
            mu, sigma, ci = predict_forward(biology, week)

        col_a, col_b = st.columns(2)
        col_a.metric("📏 Predicted mean size", f"{mu:.2f} mm")
        col_b.metric("📊 95% credible interval", f"[{ci[0]:.2f}, {ci[1]:.2f}] mm")

        x_vals = np.linspace(max(0, mu - 4*sigma), mu + 4*sigma, 200)
        y_vals = norm.pdf(x_vals, mu, sigma)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', line_color='#1e466e', name='Density'))
        fig.add_vline(x=mu, line_dash="dash", line_color="red", annotation_text=f"Mean = {mu:.2f} mm")
        fig.update_layout(title=f'{biology} at week {week}',
                          xaxis_title='Tumor size (mm)',
                          yaxis_title='Probability density')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.caption("⚠️ Disclaimer: For research & education only – not medical advice. Always consult your doctor.")
