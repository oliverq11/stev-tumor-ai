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
#  HQS
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
    
    Each plot traces the **complete tumor size trajectory** over time (weeks) for Lynch syndrome patients treated with dostarlimab immunotherapy.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("30mm.png", caption="**Plot 1:** Start size = 30 mm", use_container_width=True)
        st.caption("Tumor grows to 30 mm, then shrinks after immunotherapy. Shaded band = 90% credible interval.")
    with col2:
        st.image("60mm.png", caption="**Plot 2:** Start size = 60 mm", use_container_width=True)
        st.caption("Same treatment, but tumor starts larger. Compare the shrinkage trajectory.")

# ============================================================
# EXPANDER 2: TWO-HIT DYNAMICS (6 plots)
# ============================================================
with st.expander("🕰️ Two‑Hit Dynamics: Incubation, Latency, Age at Detection & Risk", expanded=False):
    st.markdown("""
    ### 📖 What is "First Hit" and "Second Hit"?
    
    - **First hit (inherited mutation):** A person with Lynch syndrome is born with **one faulty copy** of an MMR gene (e.g., MLH1, MSH2) inherited from a parent. This alone does not cause cancer – it only creates a **predisposition**.
      
    - **Second hit (acquired mutation):** At some point later in life, the **second healthy copy** of that MMR gene is damaged or lost. When this happens, the cell can no longer repair DNA mistakes, leading to microsatellite instability (MSI) and eventually **tumor formation**.
    """)
    
    st.markdown("### Complete stochastic model output (6 plots)")
    
    # Row 1: incubation and latency
    col1, col2 = st.columns(2)
    with col1:
        st.image("incubation.png", caption="**Plot a:** Incubation (birth → second hit)", use_container_width=True)
        st.caption("Age at which the second hit occurs. Most occur between ages 30–55.")
    with col2:
        st.image("latency.png", caption="**Plot b:** Latency (second hit → detectable tumor >1 mm)", use_container_width=True)
        st.caption("Waiting time from the second hit until the tumor becomes detectable.")
    
    # Row 2: detection_age_conditional and probability_conditional
    col1, col2 = st.columns(2)
    with col1:
        st.image("detection_age_conditional.png", caption="**Plot c (Conditional):** Detection age distribution", use_container_width=True)
        st.caption("Given that a second hit has occurred, this shows the age at clinical detection.")
    with col2:
        st.image("probability_conditional.png", caption="**Plot d (Conditional):** Probability of detection by age", use_container_width=True)
        st.caption("Given a second hit, the cumulative probability that the tumor has been detected by a given age.")
    
    # Row 3: detection_age_unconditional and probability_uconditional
    col1, col2 = st.columns(2)
    with col1:
        st.image("detection_age_unconditional.png", caption="**Plot c (Unconditional):** Detection age distribution", use_container_width=True)
        st.caption("For all Lynch patients (including those without a second hit), the age at clinical detection.")
    with col2:
        st.image("probability_uconditional.png", caption="**Plot d (Unconditional):** Probability of detection by age", use_container_width=True)
        st.caption("Overall probability that a Lynch patient will have a detected tumor by a given age.")
    
    st.markdown("""
    ---
    ### 🔑 Conditional vs. Unconditional
    
    - **Conditional (top row c & d):** *"Given that you already had the second hit, what is the probability of detection by age X?"*
      
    - **Unconditional (bottom row c & d):** *"At birth, what is your overall chance of ever having a detected tumor by age X?"*
    
    The unconditional curves are always lower because some Lynch patients never experience the second hit.
    """)

# ============================================================
# EXPANDER 3: MATHEMATICAL FRAMEWORK (15 EQUATIONS)
# ============================================================
with st.expander("📐 Mathematical Framework of the STEV Model", expanded=False):
    st.markdown(r"""
    ### A True Stochastic Process
    
    The STEV model is a **purely stochastic simulation**. At each weekly cycle, tumor growth or shrinkage is random. The mean path is only the average of many random realizations – no single patient follows the mean exactly.
    
    ---
    
    ### Equation 1: Logit Transformation
    
    $$
    Z = \ln\left(\frac{S - L}{U - S}\right)
    $$
    
    | Symbol | Meaning | Value |
    |--------|---------|-------|
    | $S$ | Tumor size (mm) | 1.1 – 60 mm |
    | $L$ | Lower physical bound | 1.0 mm |
    | $U$ | Upper physical bound | 60.0 mm |
    | $Z$ | Logit-transformed size | $-\infty$ to $+\infty$ |
    
    **What it does:** Transforms bounded tumor size into an unbounded variable where linear equations work.
    
    **Inverse:**
    $$
    S = L + \frac{U - L}{1 + e^{-Z}}
    $$
    
    ---
    
    ### Equation 2: Sensitivity Coefficient
    
    $$
    \kappa(S) = \frac{dZ}{dS} = \frac{U - L}{(S - L)(U - S)}
    $$
    
    **What it does:** Converts changes in $S$ to changes in $Z$. Essential for variance conversion.
    
    ---
    
    ### Equation 3: Stochastic Process at Each Cycle
    
    At each weekly cycle, the logit variable evolves as:
    
    $$
    Z_{t+1} = Z_t + \Delta Z_t
    $$
    
    where $\Delta Z_t$ is random. The deterministic increment (drift) is constant:
    
    $$
    \mathbb{E}[\Delta Z_t] = r
    $$
    
    The variance of the increment depends on current tumor size:
    
    $$
    \text{Var}(\Delta Z_t) = \sigma_{\text{cycle}}^2(S_t)
    $$
    
    **What it does:** At **every cycle**, growth or shrinkage is random. The mean path is only an average, not a deterministic trajectory.
    
    ---
    
    ### Equation 4: Growth Phase – Mean Path
    
    $$
    \mu_Z(t) = \alpha + r \cdot t
    $$
    
    | Symbol | Meaning | Value |
    |--------|---------|-------|
    | $r$ | Growth rate (logit/week) | $0.149133 / 3.5 \approx 0.0426$ |
    | $\alpha$ | Intercept | $-r \cdot t_0 \approx -6.44$ |
    | $t_0$ | Time offset | $43.19 \times 3.5 \approx 151.17$ weeks |
    
    **Convert to mm:**
    $$
    \mu_S(t) = L + \frac{U - L}{1 + e^{-\mu_Z(t)}}
    $$
    
    ---
    
    ### Equation 5: Per‑Cycle Noise (mm space)
    
    $$
    \sigma_{S}(S) = \max(\sigma_{\text{floor}},\; \sigma_{\text{rel}} \cdot S)
    $$
    
    | Symbol | Meaning | Value |
    |--------|---------|-------|
    | $\sigma_{\text{floor}}$ | Minimum noise | 0.5 mm |
    | $\sigma_{\text{rel}}$ | Relative noise factor | 0.20 |
    
    **What it does:** Larger tumors have larger absolute fluctuations.
    
    ---
    
    ### Equation 6: Convert Noise to Logit Variance
    
    $$
    \sigma_{Z}(S) = \kappa(S) \cdot \sigma_{S}(S)
    $$
    
    $$
    \sigma_{\text{cycle}}^2(S) = [\kappa(S) \cdot \sigma_S(S)]^2
    $$
    
    **What it does:** Translates mm‑space variance to logit space using the sensitivity coefficient.
    
    ---
    
    ### Equation 7: Biological Modulation (TMB & MMR)
    
    $$
    \text{TMB}_{\text{norm}} = \frac{\ln(1 + \text{TMB})}{\ln(11)}
    $$
    
    $$
    \text{strength} = \text{TMB}_{\text{norm}} \times f_{\text{MMR}}
    $$
    
    | MMR Status | $f_{\text{MMR}}$ |
    |------------|------------------|
    | dMMR / MSI-H | 1.3 |
    | MSS | 1.0 |
    | POLE | 1.5 |
    
    ---
    
    ### Equation 8: Immunotherapy Delay
    
    $$
    t_{\text{delay}} = 3.0 \times \left(1 - 0.30 \cdot \frac{\text{strength}}{1 + \text{strength}}\right)
    $$
    
    $$
    t_{\text{delay}} = \max(1.5,\; t_{\text{delay}})
    $$
    
    **What it does:** Higher TMB/dMMR shortens the delay (down to 1.5 weeks). Lower TMB/MSS lengthens it (up to 3.0 weeks).
    
    ---
    
    ### Equation 9: Cure Phase – 4PL Mean Path
    
    $$
    S_{\text{cure}}(\tau) = K_c + \frac{L_c - K_c}{1 + e^{-k_c (\tau - x_{0c})}}
    $$
    
    where $\tau = t - t_{\text{start}}$ (weeks since treatment started).
    
    **With effective delay:**
    $$
    S(t) = \begin{cases}
    S_{\text{start}}, & t < t_{\text{delay}} \\[4pt]
    K_c + \frac{L_c - K_c}{1 + e^{-k_c ((t - t_{\text{delay}}) - x_{0c})}}, & t \ge t_{\text{delay}}
    \end{cases}
    $$
    
    | Symbol | Meaning |
    |--------|---------|
    | $L_c$ | Upper asymptote (starting size) |
    | $K_c$ | Lower asymptote (cure floor ~1.1 mm) |
    | $k_c$ | Slope (negative = decay) |
    | $x_{0c}$ | Inflection point |
    
    ---
    
    ### Equation 10: Logit Transform for Cure Phase (LK Space)
    
    Because the cure phase has different bounds ($L_c$ and $K_c$):
    
    $$
    Z_{\text{cure}} = \ln\left(\frac{S - K_c}{L_c - S}\right)
    $$
    
    **Inverse:**
    $$
    S = K_c + \frac{L_c - K_c}{1 + e^{-Z_{\text{cure}}}}
    $$
    
    ---
    
    ### Equation 11: Total Variance – Growth Phase
    
    The total variance in $Z$ comes from **two independent sources**:
    
    $$
    \text{Var}[Z(t)] = t^2 \cdot \text{Var}(r) + \sum_{i=1}^{t} \sigma_{\text{cycle}}^2(\mu_S(i))
    $$
    
    | Term | Meaning |
    |------|---------|
    | $t^2 \cdot \text{Var}(r)$ | Patient‑to‑patient heterogeneity (slope variance) |
    | $\sum \sigma_{\text{cycle}}^2$ | Within‑patient stochasticity (per‑cycle noise) |
    
    **Why two sources?** This separates between‑patient variation from week‑to‑week randomness.
    
    ---
    
    ### Equation 12: Total Variance – Cure Phase
    
    For $t \ge t_{\text{delay}}$:
    
    $$
    \text{Var}[Z_{\text{cure}}(t)] = (t - t_{\text{delay}})^2 \cdot \text{Var}(r_{\text{decay}}) + \sum_{i=1}^{t - t_{\text{delay}}} \sigma_{\text{cycle}}^2(S_{\text{cure}}(i))
    $$
    
    **Variance freezing:** When the mean tumor size hits the floor $K_c$, variance stops accumulating.
    
    ---
    
    ### Equation 13: Biological Variance Fraction
    
    $$
    \phi_{\text{bio}} = \min\left(0.70,\; \frac{\text{strength}}{1 + \text{strength}}\right)
    $$
    
    The per‑cycle variance is multiplied by $(1 - \phi_{\text{bio}})$:
    
    - High $\phi_{\text{bio}}$ (strong biology) → **less** stochastic variance
    - Low $\phi_{\text{bio}}$ (weak biology) → **more** stochastic variance
    
    ---
    
    ### Equation 14: Biological Time Factor
    
    Let:
    $$
    I(t) = \frac{1}{1 + e^{-1.2 (t - t_{\text{delay}})}}
    $$
    
    Then:
    $$
    \psi(t) = (1 - I(t)) + 0.25 \cdot I(t)
    $$
    
    **What it does:** Reduces variance to 25% after immunotherapy response begins.
    
    ---
    
    ### Equation 15: CLT Confidence Bands (90% Credible Intervals)
    
    In logit space, because increments are independent, the Central Limit Theorem applies:
    
    $$
    z_{0.95} = \Phi^{-1}(0.95) \approx 1.645
    $$
    
    **Lower bound:**
    $$
    Z_{\text{lo}}(t) = \mu_Z(t) - 1.645 \cdot \sqrt{\text{Var}[Z(t)]}
    $$
    
    **Upper bound:**
    $$
    Z_{\text{hi}}(t) = \mu_Z(t) + 1.645 \cdot \sqrt{\text{Var}[Z(t)]}
    $$
    
    **Convert back to mm:**
    $$
    S_{\text{lo}}(t) = L + \frac{U - L}{1 + e^{-Z_{\text{lo}}(t)}}, \quad
    S_{\text{hi}}(t) = L + \frac{U - L}{1 + e^{-Z_{\text{hi}}(t)}}
    $$
    
    **What it does:** Provides the 90% credible intervals shown as shaded bands in the plots.
    """)

    # --- SEPARATE SECTION FOR TWO-HIT DYNAMICS ---
    st.markdown(r"""
    ### Two‑Hit Dynamics: Mathematical Formulation
    
    The following equations describe the **stochastic process of tumor initiation and detection** in Lynch syndrome.
    
    ---
    
    #### Equation 16: Incubation (Birth → Second Hit)
    
    The time from birth until the second hit occurs follows a **Gamma distribution**:
    
    $$
    f_{\text{inc}}(t) = \frac{t^{k-1} e^{-t/\theta}}{\theta^k \, \Gamma(k)}, \quad t \ge 0
    $$
    
    | Symbol | Meaning | Typical Value |
    |--------|---------|----------------|
    | $k$ | Shape parameter | ~4–6 |
    | $\theta$ | Scale parameter | ~5–8 years |
    | $\Gamma(k)$ | Gamma function | |
    
    **What it does:** Models the random waiting time for the second MMR hit. Most second hits occur between ages 30–55 years.
    
    ---
    
    #### Equation 17: Latency (Second Hit → Detectable Tumor)
    
    Once the second hit occurs, the tumor grows until it becomes detectable (>1 mm). This waiting time also follows a **Gamma distribution**:
    
    $$
    f_{\text{lat}}(t) = \frac{t^{k_{\text{lat}}-1} e^{-t/\theta_{\text{lat}}}}{\theta_{\text{lat}}^{k_{\text{lat}}} \, \Gamma(k_{\text{lat}})}, \quad t \ge 0
    $$
    
    **What it does:** Models the time from tumor initiation to clinical detection. Shorter latency means faster-growing tumors.
    
    ---
    
    #### Equation 18: Convolution (Incubation + Latency)
    
    The **age at detection** is the sum of two independent random variables:
    
    $$
    T_{\text{detection}} = T_{\text{inc}} + T_{\text{lat}}
    $$
    
    The probability density of the sum is the **convolution integral**:
    
    $$
    f_{\text{det}}(t) = \int_{0}^{t} f_{\text{inc}}(\tau) \, f_{\text{lat}}(t - \tau) \, d\tau
    $$
    
    **What it does:** Combines the waiting time for the second hit with the subsequent growth time to get the overall age at clinical detection.
    
    ---
    
    #### Equation 19: Conditional Probability of Detection by Age
    
    Given that a second hit has occurred, the **conditional probability** that the tumor has been detected by age $t$ is:
    
    $$
    P_{\text{cond}}(t) = \int_{0}^{t} f_{\text{det}}(\tau) \, d\tau
    $$
    
    **What it does:** Answers: *"If a Lynch patient has already experienced the second hit, what is the probability that their tumor has been detected by age $t$?"*
    
    ---
    
    #### Equation 20: Unconditional Probability of Detection by Age
    
    The **unconditional probability** accounts for the fact that not all Lynch patients experience the second hit. Let $R_{\text{lifetime}}$ be the lifetime risk of a second hit (~80% for MLH1/MSH2):
    
    $$
    P_{\text{uncond}}(t) = R_{\text{lifetime}} \cdot P_{\text{cond}}(t)
    $$
    
    **What it does:** Answers: *"At birth, what is the overall chance that a Lynch patient will have a detected tumor by age $t$?"*
    
    ---
    
    #### Equation 21: Lifetime Risk by Gene
    
    For Lynch syndrome, the lifetime risk of colorectal cancer varies by gene:
    
    | Gene | Lifetime Risk |
    |------|---------------|
    | MLH1 | ~70–80% |
    | MSH2 | ~70–80% |
    | MSH6 | ~50–60% |
    | PMS2 | ~15–20% |
    
    These values calibrate the scale parameter $\theta$ in the incubation Gamma distribution.
    
    ---
    
    #### Summary of Two‑Hit Parameters
    
    | Parameter | Meaning | Typical Range |
    |-----------|---------|----------------|
    | $k_{\text{inc}}$ | Incubation shape | 4–6 |
    | $\theta_{\text{inc}}$ | Incubation scale | 5–8 years |
    | $k_{\text{lat}}$ | Latency shape | 2–4 |
    | $\theta_{\text{lat}}$ | Latency scale | 1–3 years |
    | $R_{\text{lifetime}}$ | Lifetime risk of second hit | 0.50–0.80 |
    
    *Distributions calibrated to published Lynch syndrome data (Hampel et al., 2008; Jenkins et al., 2006).*
    """)

    # --- FINAL SUMMARY TABLE ---
    st.markdown(r"""
    ---
    
    ### Summary: All Parameters at a Glance
    
    | Parameter | Meaning | Value |
    |-----------|---------|-------|
    | $L$ | Lower physical bound | 1.0 mm |
    | $U$ | Upper physical bound | 60.0 mm |
    | $r$ | Growth rate (logit/week) | 0.0426 |
    | $\sigma_{\text{floor}}$ | Minimum noise | 0.5 mm |
    | $\sigma_{\text{rel}}$ | Relative noise factor | 0.20 |
    | $t_{\text{delay}}$ | Immunotherapy delay | 1.5–3.0 weeks |
    | $\text{Var}(r)$ | Slope variance (growth) | Calibrated |
    | $\text{Var}(r_{\text{decay}})$ | Slope variance (cure) | Calibrated |
    | $\phi_{\text{bio}}$ | Biological variance fraction | 0–0.70 |
    | $k_{\text{inc}}$ | Incubation shape | 4–6 |
    | $\theta_{\text{inc}}$ | Incubation scale | 5–8 years |
    | $k_{\text{lat}}$ | Latency shape | 2–4 |
    | $\theta_{\text{lat}}$ | Latency scale | 1–3 years |
    
    ---
    
    *Full simulation code available in the repository. Model parameters calibrated to published trial data (GARNET, KEYNOTE-177) and Lynch syndrome epidemiology.*
    """)
    ---
    
    #### Summary of Two‑Hit Parameters
    
    | Parameter | Meaning | Typical Range |
    |-----------|---------|----------------|
    | $k_{\text{inc}}$ | Incubation shape | 4–6 |
    | $\theta_{\text{inc}}$ | Incubation scale | 5–8 years |
    | $k_{\text{lat}}$ | Latency shape | 2–4 |
    | $\theta_{\text{lat}}$ | Latency scale | 1–3 years |
    | $R_{\text{lifetime}}$ | Lifetime risk of second hit | 0.50–0.80 |
    
    ---
    
    *Distributions calibrated to published Lynch syndrome data (Hampel et al., 2008; Jenkins et al., 2006).*
    """)

    
    ---
    
    ### Summary: All Parameters at a Glance
    
    | Parameter | Meaning | Value |
    |-----------|---------|-------|
    | $L$ | Lower physical bound | 1.0 mm |
    | $U$ | Upper physical bound | 60.0 mm |
    | $r$ | Growth rate (logit/week) | 0.0426 |
    | $\sigma_{\text{floor}}$ | Minimum noise | 0.5 mm |
    | $\sigma_{\text{rel}}$ | Relative noise factor | 0.20 |
    | $t_{\text{delay}}$ | Immunotherapy delay | 1.5–3.0 weeks |
    | $\text{Var}(r)$ | Slope variance (growth) | Calibrated |
    | $\text{Var}(r_{\text{decay}})$ | Slope variance (cure) | Calibrated |
    | $\phi_{\text{bio}}$ | Biological variance fraction | 0–0.70 |
    
    ---
    
    *Full simulation code available in the repository. Model parameters calibrated to published trial data (GARNET, KEYNOTE-177).*
    """)

# ============================================================
# EXPANDER 4: CLINICAL CASE (BENIGN POLYP RESPONSE)
# ============================================================
with st.expander("📋 Clinical Case: Benign Polyp Responded to Dostarlimab", expanded=False):
    st.markdown("""
    ### A Surprising Validation of the STEV Model
    
    **Clinical history:**
    
    - Patient with Lynch syndrome undergoing dostarlimab immunotherapy
    - Prior **MLH1‑deficient malignant tumor** (TMB = 55) showed shrinkage **faster than the STEV model's population mean**
    - A **flat, benign polyp** (~5–6 mm) in the descending colon (≈30 cm from anal orifice)
    - **Could not be removed** during two colonoscopies – the second attempted by a specialist using **Endoscopic Submucosal Dissection (ESD)**, which failed due to the polyp's flat, fibrotic morphology
    
    **Outcome:**
    
    - During dostarlimab treatment, the benign polyp **shrank progressively**
    - Shrinkage was **slower than the STEV model's population mean** (unlike the faster‑than‑mean MLH1 tumor)
    - Both trajectories – the fast MLH1 tumor and the slower benign polyp – fell **within the model's 90% credible interval**
    - Third colonoscopy successfully removed the polyp without complications
    - Post‑removal pathology confirmed **benign** histology
    
    **Why this matters:**
    
    1. **Model validation** – The STEV model's credible interval captured **both** an exceptionally fast malignant tumor and a slower benign polyp
    
    2. **Biological insight** – Response speed varies continuously:
       - MLH1, high TMB → faster than mean
       - Benign polyps → slower than mean
       - Both still within the model's stochastic range
    
    3. **Practical guidance** – For flat, unresectable polyps where ESD fails, a trial of immunotherapy may enable subsequent removal – but response may be **slower than the model's mean**
    
    4. **Future directions** – This suggests possible **chemoprevention** applications of immunotherapy in Lynch syndrome
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
    
    - **📈 Growth & Immunotherapy:** Click expander to see 30mm and 60mm tumor growth/shrinkage curves.
    
    - **🕰️ Two‑Hit Dynamics:** Click expander to see incubation, latency, conditional & unconditional probability plots.
    
    - **📐 Mathematical Framework:** Click expander to see the full 15‑equation formulation.
    
    - **📋 Clinical Case:** Click expander to see real‑world validation with a benign polyp.
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
