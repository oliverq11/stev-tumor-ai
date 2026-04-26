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

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="STEV: Stochastic Tumor Response AI", layout="wide", page_icon="🧬")

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
# Widget values
if 'week' not in st.session_state:
    st.session_state.week = 8
if 'size' not in st.session_state:
    st.session_state.size = 1.4
if 'week_forward' not in st.session_state:
    st.session_state.week_forward = 8
if 'biology' not in st.session_state:
    st.session_state.biology = 'MLH1'

# Expander states (all closed by default)
if 'expander_growth' not in st.session_state:
    st.session_state.expander_growth = False
if 'expander_twohit' not in st.session_state:
    st.session_state.expander_twohit = False
if 'expander_math' not in st.session_state:
    st.session_state.expander_math = False
if 'expander_clinical' not in st.session_state:
    st.session_state.expander_clinical = False

# Prediction results flag
if 'show_tab1_prediction' not in st.session_state:
    st.session_state.show_tab1_prediction = False
if 'tab1_probs' not in st.session_state:
    st.session_state.tab1_probs = None
if 'tab1_most_likely' not in st.session_state:
    st.session_state.tab1_most_likely = None
if 'tab1_week' not in st.session_state:
    st.session_state.tab1_week = None
if 'tab1_size' not in st.session_state:
    st.session_state.tab1_size = None

# ============================================================
# CUSTOM CSS
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
    
    /* ===== BUTTONS: FORCE WHITE TEXT IN ALL MODES ===== */
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
    /* Ensure any inner spans or divs inside button are also white */
    .stButton button *,
    .stButton button span,
    .stButton button div {
        color: white !important;
    }
    
    .streamlit-expanderHeader { background-color: #e9ecef; border-radius: 8px; color: #212529; }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: bold; color: #1e466e; }
    [data-testid="stMetricLabel"] { color: #212529; }

    /* === DARK MODE OVERRIDE === */
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
        
        /* DARK MODE BUTTONS */
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

st.title("🧬 STEV: Stochastic Tumor Evolution and Immunological Response")
st.markdown('<div class="subtitle">Lynch Syndrome Colorectal Tumors</div>', unsafe_allow_html=True)
st.markdown('<div class="author">Horatio Quinones / Sherry Johnson / et-al</div>', unsafe_allow_html=True)

# ============================================================
# WHAT THIS APP DOES
# ============================================================
st.markdown("""
### 🔍 What this app does

This tool uses a **stochastic model** (STEV) built on real clinical data from Lynch syndrome colorectal cancer patients treated with dostarlimab. The app is organized into four main sections:

1. **🔍 Size -> Biology** - Enter tumor size -> get most likely biology
2. **🔮 Biology -> Size** - Select biology -> get predicted size range
3. **📈 Growth and Immunotherapy Curves** - 30mm and 60mm trajectories with 90% credible bands
4. **🕰️ Two-Hit Dynamics** - Incubation, latency, and probability curves

**Plus:** Mathematical framework (18 equations) and clinical case validation.
""")

# ============================================================
# EXPANDER 1: GROWTH CURVES
# ============================================================
with st.expander("📈 Tumor Growth and Immunotherapy Response (Treatment initiated at 30mm and 60mm)", expanded=st.session_state.expander_growth):
    st.markdown("""
    **Growth phase:** Tumor grows from tiny (~1.1 mm) to threshold size.  
    **Treatment initiation:** At 30 mm or 60 mm, immunotherapy begins.  
    **Cure phase:** Tumor shrinks back to minimal residual size (~1.1 mm).
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("30mm.png"):
            st.image("30mm.png", caption="Treatment initiated at 30 mm", use_container_width=True)
    with col2:
        if os.path.exists("60mm.png"):
            st.image("60mm.png", caption="Treatment initiated at 60 mm", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Real Patient Validation: MLH1 Tumor (TMB = 55.4)")
    col1, col2 = st.columns([2, 1])
    with col1:
        if os.path.exists("MLH1_55TMB.png"):
            st.image("MLH1_55TMB.png", caption="~20 mm to ~1.5 mm in 9.3 weeks", width=350)
    with col2:
        st.markdown("**MLH1 | TMB = 55.4** | ~20 mm → ~1.5 mm in 9 weeks | Faster than population mean")

# ============================================================
# EXPANDER 2: TWO-HIT DYNAMICS
# ============================================================
with st.expander("🕰️ Two-Hit Dynamics: Incubation, Latency, Age at Detection and Risk", expanded=st.session_state.expander_twohit):
    st.markdown("""
    - **First hit:** Inherited MMR mutation (e.g., MLH1, MSH2) - predisposition only
    - **Second hit:** Acquired mutation - leads to MSI and tumor formation
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("incubation.png"):
            st.image("incubation.png", caption="Incubation (birth to second hit)", use_container_width=True)
    with col2:
        if os.path.exists("latency.png"):
            st.image("latency.png", caption="Latency (second hit to detectable tumor)", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("detection_age_conditional.png"):
            st.image("detection_age_conditional.png", caption="Detection age (conditional)", use_container_width=True)
    with col2:
        if os.path.exists("probability_conditional.png"):
            st.image("probability_conditional.png", caption="Probability of detection (conditional)", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("detection_age_unconditional.png"):
            st.image("detection_age_unconditional.png", caption="Detection age (unconditional)", use_container_width=True)
    with col2:
        if os.path.exists("probability_uconditional.png"):
            st.image("probability_uconditional.png", caption="Probability of detection (unconditional)", use_container_width=True)

# ============================================================
# EXPANDER 3: MATHEMATICAL FRAMEWORK
# ============================================================
with st.expander("📐 Mathematical Framework of the STEV Model", expanded=st.session_state.expander_math):
    st.markdown(r"""
    **Equation 1: Logit Transformation** – $Z = \ln((S-L)/(U-S))$, $L=1.0$ mm, $U=60.0$ mm  
    **Equation 2: Sensitivity Coefficient** – $\kappa(S) = (U-L)/((S-L)(U-S))$  
    **Equation 3: Stochastic Process** – $Z_{t+1} = Z_t + \Delta Z_t$, $\mathbb{E}[\Delta Z_t] = r$, $\text{Var}(\Delta Z_t) = \sigma_{\text{cycle}}^2(S_t)$  
    **Equation 4: Growth Mean Path** – $\mu_Z(t) = \alpha + r \cdot t$, $r=0.0426$/week  
    **Equation 5: Per-Cycle Noise** – $\sigma_S(S) = \max(0.5,\; 0.20 \cdot S)$ mm  
    **Equation 6: Logit Variance** – $\sigma_{\text{cycle}}^2(S) = [\kappa(S) \cdot \sigma_S(S)]^2$  
    **Equation 7: Biological Modulation** – strength $= \ln(1+TMB)/\ln(11) \times f_{MMR}$  
    **Equation 8: Immunotherapy Delay** – $t_{\text{delay}} = \max(1.5,\; 3.0 \times (1 - 0.30 \cdot \text{strength}/(1+\text{strength})))$  
    **Equation 9: Cure Phase** – $S_{\text{cure}}(t) = K_c + (L_c-K_c)/(1+e^{-k_c(t-t_{\text{delay}}-x_{0c})})$  
    **Equation 10: Growth Variance** – $\text{Var}[Z(t)] = t^2 \text{Var}(r) + \sum \sigma_{\text{cycle}}^2(\mu_S(i))$  
    **Equation 11: Cure Variance** – $\text{Var}[Z_{\text{cure}}(t)] = (t-t_{\text{delay}})^2 \text{Var}(r_{\text{decay}}) + \sum \sigma_{\text{cycle}}^2(S_{\text{cure}}(i))$  
    **Equation 12: Biological Variance Fraction** – $\phi_{\text{bio}} = \min(0.70,\; \text{strength}/(1+\text{strength}))$  
    **Equation 13: CLT Confidence Bands** – $Z_{\text{lo,hi}}(t) = \mu_Z(t) \pm 1.645\sqrt{\text{Var}[Z(t)]}$  
    **Equations 14-18: Two-Hit Dynamics** – Gamma distributions, convolution, conditional/unconditional probabilities
    
    *Full details in code repository.*
    """)

# ============================================================
# EXPANDER 4: CLINICAL CASE
# ============================================================
with st.expander("📋 Clinical Case: Benign Polyp Responded to Dostarlimab", expanded=st.session_state.expander_clinical):
    st.markdown("""
    **Clinical history:** Lynch patient on dostarlimab. Flat benign polyp (~5-6 mm) in descending colon. Could not be removed during two colonoscopies (ESD failed).
    
    **Outcome:** Polyp shrank progressively (slower than MLH1 tumor mean). Third colonoscopy removed it successfully. Pathology confirmed benign.
    
    **Validation:** Both fast MLH1 tumor and slower benign polyp fell within model's 90% credible interval.
    """)
    
    if os.path.exists("benign_polyp_STEV.png"):
        st.image("benign_polyp_STEV.png", caption="STEV projection vs. actual measurements", use_container_width=True)

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
# SIDEBAR (WITH WORKING RESET BUTTON)
# ============================================================
with st.sidebar:
    app_url = "https://stev-tumor-ai-skrobcqyqyyz4sjpvqdqmh.streamlit.app/"
    
    qr = qrcode.QRCode(box_size=5, border=2)
    qr.add_data(app_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), width=150, caption="Scan with phone camera")
    
    st.markdown("**Or copy link to share:**")
    st.code(app_url, language="text")
    
    st.markdown("---")
    
    # RESET BUTTON - THIS WORKS
    if st.button("🔄 Reset Everything", use_container_width=True):
        # Reset widget values
        st.session_state.week = 8
        st.session_state.size = 1.4
        st.session_state.week_forward = 8
        st.session_state.biology = 'MLH1'
        
        # Clear predictions
        st.session_state.show_tab1_prediction = False
        st.session_state.tab1_probs = None
        st.session_state.tab1_most_likely = None
        
        # Close all expanders
        st.session_state.expander_growth = False
        st.session_state.expander_twohit = False
        st.session_state.expander_math = False
        st.session_state.expander_clinical = False
        
        st.rerun()
    
    st.markdown("### ℹ️ How to use")
    st.markdown("- **Size -> Biology:** Enter size, get biology")
    st.markdown("- **Biology -> Size:** Select biology, get size range")
    st.markdown("- **Expanders above** show curves and math")
    st.markdown("---")
    st.markdown("**STEV model** - Lynch Syndrome")
    st.markdown("*Horatio Quinones / et al*")

# ============================================================
# MAIN APP WITH TWO TABS
# ============================================================
tab1, tab2 = st.tabs(["🔍 Size -> Biology", "🔮 Biology -> Size"])

# ========== TAB 1: SIZE -> BIOLOGY ==========
with tab1:
    col_left, col_right = st.columns(2)
    with col_left:
        week = st.selectbox("📅 Week", weeks, index=weeks.index(st.session_state.week), key="week")
    with col_right:
        size = st.slider("📏 Tumor size (mm)", 0.0, 30.0, st.session_state.size, 0.1, key="size")

    if st.button("Predict Biology", use_container_width=True):
        probs = predict_inverse(size, week)
        most_likely = max(probs, key=probs.get)
        st.session_state.show_tab1_prediction = True
        st.session_state.tab1_probs = probs
        st.session_state.tab1_most_likely = most_likely
        st.session_state.tab1_week = week
        st.session_state.tab1_size = size

    # Show prediction if exists
    if st.session_state.show_tab1_prediction and st.session_state.tab1_probs:
        probs = st.session_state.tab1_probs
        most_likely = st.session_state.tab1_most_likely
        week = st.session_state.tab1_week
        size = st.session_state.tab1_size
        
        col_a, col_b = st.columns(2)
        col_a.metric("🧬 Most likely biology", most_likely)
        col_b.metric("📊 Probability", f"{probs[most_likely]:.1%}")
        
        df = pd.DataFrame(list(probs.items()), columns=['Biology', 'Probability'])
        fig = px.bar(df, x='Biology', y='Probability', color='Biology',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title=f'Week {week}, size = {size} mm')
        fig.update_layout(yaxis_title='Posterior probability', xaxis_title='Biology')
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("⚠️ Research & education only - not medical advice")

# ========== TAB 2: BIOLOGY -> SIZE ==========
with tab2:
    col_left, col_right = st.columns(2)
    with col_left:
        week = st.selectbox("📅 Week", weeks, index=weeks.index(st.session_state.week_forward), key="week_forward")
    with col_right:
        bio_index = names.index(st.session_state.biology) if st.session_state.biology in names else 1
        biology = st.selectbox("🧬 Biology", names, index=bio_index, key="biology")

    if st.button("Predict Size", use_container_width=True):
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
        
        st.caption("⚠️ Research & education only - not medical advice")
