import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import qrcode
from io import BytesIO

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="STEV: Stochastic Tumor Response AI", layout="wide", page_icon="🧬")

# ============================================================
# CUSTOM CSS WITH DARK MODE SUPPORT
# ============================================================
st.markdown("""
<style>
    /* Light mode (default) */
    .stApp { background-color: #f5f7fb; }
    h1 { color: #1e466e; font-family: 'Segoe UI', sans-serif; }
    .subtitle { color: #2c6e9e; font-size: 1.2rem; margin-bottom: 1rem; }
    .author { color: #6c757d; font-size: 0.9rem; margin-bottom: 2rem; }
    .stButton button {
        background-color: #1e466e; color: white; font-weight: bold;
        border-radius: 8px; padding: 0.5rem 1rem; width: 100%;
        transition: 0.2s;
    }
    .stButton button:hover { background-color: #0f2e4a; transform: scale(1.02); }
    .streamlit-expanderHeader { background-color: #e9ecef; border-radius: 8px; }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: bold; color: #1e466e; }
    /* Ensure all text inherits properly in light mode */
    body, .stMarkdown, .stText, .stSelectbox, .stSlider, .stDataFrame {
        color: #212529;
    }

    /* === DARK MODE OVERRIDE === */
    /* Activates automatically when phone/computer system dark mode is ON */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0a0a0a !important;
        }
        h1, h2, h3, .subtitle, .author, .stMarkdown, .stText, .stSelectbox label, .stSlider label {
            color: #f0f0f0 !important;
        }
        .stButton button {
            background-color: #2c6e9e !important;
            color: white !important;
        }
        .stButton button:hover {
            background-color: #1e466e !important;
        }
        .streamlit-expanderHeader {
            background-color: #1e1e1e !important;
            color: #f0f0f0 !important;
        }
        [data-testid="stMetricValue"] {
            color: #79c2ff !important;
        }
        [data-testid="stMetricLabel"] {
            color: #dddddd !important;
        }
        .stDataFrame, .stDataFrame div, .dataframe {
            background-color: #1e1e1e !important;
            color: #f0f0f0 !important;
        }
        /* Force Plotly chart background and text */
        .plotly-graph-div .main-svg {
            background-color: #0a0a0a !important;
        }
        .plotly-graph-div .gtitle, .plotly-graph-div .xtitle, .plotly-graph-div .ytitle,
        .plotly-graph-div .legendtext, .plotly-graph-div .hovertext text {
            fill: #f0f0f0 !important;
            color: #f0f0f0 !important;
        }
        .plotly-graph-div .axis line, .plotly-graph-div .axis path {
            stroke: #888888 !important;
        }
        .plotly-graph-div .axis tick text {
            fill: #cccccc !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for disclaimer
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

# Subgroup means
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
# SIDEBAR (with real QR code)
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
    """)
    st.markdown("---")
    st.markdown("**STEV model** – Lynch Syndrome Colorectal Tumors")
    st.markdown("*Horatio Quinones / Sherry Johnson / et al*")

# ============================================================
# MAIN APP WITH TABS
# ============================================================
tab1, tab2 = st.tabs(["🔍 Size → Biology", "🔮 Biology → Size"])

# ---------- TAB 1: INVERSE PREDICTION ----------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        week = st.selectbox("📅 Week", weeks, index=8)
    with col2:
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
        # Optional: force dark template if needed – but CSS should handle it
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Detailed probabilities"):
            st.dataframe(df.style.format({'Probability': '{:.3f}'}))

        # DISCLAIMER APPEARS ONLY HERE, AFTER PREDICTION
        st.markdown("---")
        st.caption("⚠️ Disclaimer: For research & education only – not medical advice. Always consult your doctor.")

# ---------- TAB 2: FORWARD PREDICTION ----------
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        week = st.selectbox("📅 Week", weeks, index=8, key="forward_week")
    with col2:
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

        # DISCLAIMER APPEARS ONLY HERE, AFTER PREDICTION
        st.markdown("---")
        st.caption("⚠️ Disclaimer: For research & education only – not medical advice. Always consult your doctor.")
