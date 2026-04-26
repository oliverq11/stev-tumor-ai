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
# HQS
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="STEV: Stochastic Tumor Response AI", layout="wide", page_icon="🧬")

# ============================================================
# MINIMAL SESSION STATE (ONLY FOR PREDICTION DISPLAY)
# ============================================================
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False
if 'prediction_probs' not in st.session_state:
    st.session_state.prediction_probs = None
if 'prediction_most_likely' not in st.session_state:
    st.session_state.prediction_most_likely = None
if 'prediction_week' not in st.session_state:
    st.session_state.prediction_week = None
if 'prediction_size' not in st.session_state:
    st.session_state.prediction_size = None

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #f5f7fb; }
    h1, h2, h3, .stMarkdown, p, div, span, label {
        color: #212529;
    }
    h1 { color: #1e466e; font-family: 'Segoe UI', sans-serif; }
    .subtitle { color: #2c6e9e; font-size: 1.2rem; margin-bottom: 1rem; }
    .author { color: #6c757d; font-size: 0.9rem; margin-bottom: 2rem; }
    
    .stButton button {
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
        .stApp, .main, .stAppViewContainer {
            background-color: #0a0a0a !important;
        }
        body, p, div, span, label, .stMarkdown, h1, h2, h3, .subtitle, .author {
            color: #ffffff !important;
        }
        .stButton button {
            background-color: #2c6e9e !important;
        }
        .stButton button:hover {
            background-color: #1e466e !important;
        }
        .streamlit-expanderHeader {
            background-color: #1e1e1e !important;
            color: white !important;
        }
        [data-testid="stMetricValue"] { color: #79c2ff !important; }
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

This tool uses a **Stochastic Model** (STEV) built the Tumor Evolution based on clinical data from Lynch Syndrome Colorectal Cancer patients treated with Immunotherapy (Dostarlimab) without other previous treatments.

#### 1. 🔍 Size -> Genotype
Given tumor size and treatment time (weeks), returns most likely Tumor Genotype (POLE, MLH1, MSH2, MSI-H, MSH6).

#### 2. 🔮 Genotype -> Size
Given tumor genotype and treatment time (weeks), predicts expected tumor size and range with 95% Credible Intervals.

#### 3. 📈 Growth and Immunotherapy Curves
Trajectories of the Evolution of Geometrical Dimension Changes in Time, from tiny to 30mm/60mm followed by volume shrinkage, with 90% Credible Bands.

#### 4. 🕰️ Two-Hit Dynamics
Incubation, latency, conditional and unconditional probability plots.

#### 5.  📐 Mathematical Formulation
Stochastic Tumor EVolution model (STEV).

#### 6.  📋 Clinical Case 
Colon Benign Poly Response to Immunotheraoy

""")

# ============================================================
# EXPANDER 1: GROWTH CURVES
# ============================================================
with st.expander("📈 Tumor Growth and Immunotherapy Response", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("30mm.png"):
            st.image("30mm.png", caption="Treatment initiated at 30 mm", use_container_width=True)
    with col2:
        if os.path.exists("60mm.png"):
            st.image("60mm.png", caption="Treatment initiated at 60 mm", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 MLH1 Tumor Validation (TMB = 55.4)")
    col1, col2 = st.columns([2, 1])
    with col1:
        if os.path.exists("MLH1_55TMB.png"):
       #     st.image("MLH1_55TMB.png", caption="~20mm to ~1.5mm in 9.3 weeks", width=400)
            st.image("MLH1_55TMB.png", caption="STEV projection vs. actual measurements", use_container_width=True)
    with col2:
        st.markdown("**MLH1 | TMB=55.4** | Faster than population mean")

# ============================================================
# EXPANDER 2: TWO-HIT DYNAMICS
# ============================================================
with st.expander("🕰️ Two-Hit Dynamics: Tumor Incubation, Latency, Detection Phase, Risk Assement", expanded=False):

#st.markdown("""
    ### 📖 What is "First Hit" and "Second Hit"?
    
    - **First hit (inherited mutation):** A person with Lynch syndrome is born with **one faulty copy** of an MMR gene (e.g., MLH1, MSH2) inherited from a parent. This alone does not cause cancer – it only creates a **predisposition**.
      
    - **Second hit (acquired mutation):** At some point later in life, the **second healthy copy** of that MMR gene is damaged or lost (due to random chance, environment, or aging). When this happens, the cell can no longer repair DNA mistakes, leading to microsatellite instability (MSI) and eventually **tumor formation**.
 #   """)    
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("incubation.png"):
            st.image("incubation.png", caption="Incubation (birth to second hit)", use_container_width=True)
    with col2:
        if os.path.exists("latency.png"):
            st.image("latency.png", caption="Latency (second hit to detectable)", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("detection_age_conditional.png"):
            st.image("detection_age_conditional.png", caption="Detection age (conditional)", use_container_width=True)
    with col2:
        if os.path.exists("probability_conditional.png"):
            st.image("probability_conditional.png", caption="Probability (conditional)", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("detection_age_unconditional.png"):
            st.image("detection_age_unconditional.png", caption="Detection age (unconditional)", use_container_width=True)
    with col2:
        if os.path.exists("probability_uconditional.png"):
            st.image("probability_uconditional.png", caption="Probability (unconditional)", use_container_width=True)

# ============================================================
# EXPANDER 3: MATHEMATICAL FRAMEWORK
# ============================================================

with st.expander("📐 Mathematical Framework of the STEV Model", expanded=False):
    st.markdown(r"""
    ### A True Stochastic Process
    
    The STEV model is a **purely stochastic simulation**. At each weekly cycle, tumor growth or shrinkage is random. The mean path is only the average of many random realizations - no single patient follows the mean exactly.
    
    ---
    
    **Equation 1: Logit Transformation**
    $$
    Z = \ln\left(\frac{S - L}{U - S}\right)
    $$
    **Inverse:** $S = L + \frac{U - L}{1 + e^{-Z}}$
    - $L = 1.0$ mm (lower bound), $U = 60.0$ mm (upper bound)
    
    ---
    
    **Equation 2: Sensitivity Coefficient**
    $$
    \kappa(S) = \frac{dZ}{dS} = \frac{U - L}{(S - L)(U - S)}
    $$
    
    ---
    
    **Equation 3: Stochastic Process at Each Cycle**
    $$
    Z_{t+1} = Z_t + \Delta Z_t, \quad \mathbb{E}[\Delta Z_t] = r, \quad \text{Var}(\Delta Z_t) = \sigma_{\text{cycle}}^2(S_t)
    $$
    
    ---
    
    **Equation 4: Growth Phase - Mean Path**
    $$
    \mu_Z(t) = \alpha + r \cdot t
    $$
    - $r = 0.0426$ per week, $\alpha \approx -6.44$
    
    ---
    
    **Equation 5: Per-Cycle Noise (mm space)**
    $$
    \sigma_{S}(S) = \max(0.5,\; 0.20 \cdot S) \text{ mm}
    $$
    
    ---
    
    **Equation 6: Convert Noise to Logit Variance**
    $$
    \sigma_{\text{cycle}}^2(S) = [\kappa(S) \cdot \sigma_S(S)]^2
    $$
    
    ---
    
    **Equation 7: Biological Modulation (TMB and MMR)**
    $$
    \text{strength} = \frac{\ln(1 + \text{TMB})}{\ln(11)} \times f_{\text{MMR}}
    $$
    - dMMR/MSI-H: $f=1.3$, MSS: $f=1.0$, POLE: $f=1.5$
    
    ---
    
    **Equation 8: Immunotherapy Delay**
    $$
    t_{\text{delay}} = \max\left(1.5,\; 3.0 \times \left(1 - 0.30 \cdot \frac{\text{strength}}{1 + \text{strength}}\right)\right)
    $$
    
    ---
    
    **Equation 9: Cure Phase - 4PL Mean Path**
    $$
    S_{\text{cure}}(t) = K_c + \frac{L_c - K_c}{1 + e^{-k_c (t - t_{\text{delay}} - x_{0c})}}, \quad t \ge t_{\text{delay}}
    $$
    - $L_c$ = starting size, $K_c \approx 1.1$ mm (cure floor)
    
    ---
    
    **Equation 10: Total Variance - Growth Phase**
    $$
    \text{Var}[Z(t)] = t^2 \cdot \text{Var}(r) + \sum_{i=1}^{t} \sigma_{\text{cycle}}^2(\mu_S(i))
    $$
    
    ---
    
    **Equation 11: Total Variance - Cure Phase**
    $$
    \text{Var}[Z_{\text{cure}}(t)] = (t - t_{\text{delay}})^2 \cdot \text{Var}(r_{\text{decay}}) + \sum_{i=1}^{t - t_{\text{delay}}} \sigma_{\text{cycle}}^2(S_{\text{cure}}(i))
    $$
    
    ---
    
    **Equation 12: Biological Variance Fraction**
    $$
    \phi_{\text{bio}} = \min\left(0.70,\; \frac{\text{strength}}{1 + \text{strength}}\right)
    $$
    
    ---
    
    **Equation 13: CLT Confidence Bands (90% Credible Intervals)**
    $$
    Z_{\text{lo}}(t) = \mu_Z(t) - 1.645 \cdot \sqrt{\text{Var}[Z(t)]}, \quad
    Z_{\text{hi}}(t) = \mu_Z(t) + 1.645 \cdot \sqrt{\text{Var}[Z(t)]}
    $$
    $$
    S_{\text{lo}}(t) = L + \frac{U - L}{1 + e^{-Z_{\text{lo}}(t)}}, \quad
    S_{\text{hi}}(t) = L + \frac{U - L}{1 + e^{-Z_{\text{hi}}(t)}}
    $$
    
    ---
    
    ### Two-Hit Dynamics: Mathematical Formulation
    
    ---
    
    **Equation 14: Incubation (Birth to Second Hit)**
    $$
    f_{\text{inc}}(t) = \frac{t^{k-1} e^{-t/\theta}}{\theta^k \, \Gamma(k)}, \quad t \ge 0
    $$
    - Gamma distribution, shape $k \approx 4-6$, scale $\theta \approx 5-8$ years
    
    ---
    
    **Equation 15: Latency (Second Hit to Detectable Tumor)**
    $$
    f_{\text{lat}}(t) = \frac{t^{k_{\text{lat}}-1} e^{-t/\theta_{\text{lat}}}}{\theta_{\text{lat}}^{k_{\text{lat}}} \, \Gamma(k_{\text{lat}})}, \quad t \ge 0
    $$
    
    ---
    
    **Equation 16: Convolution (Incubation + Latency)**
    $$
    T_{\text{detection}} = T_{\text{inc}} + T_{\text{lat}}, \quad
    f_{\text{det}}(t) = \int_{0}^{t} f_{\text{inc}}(\tau) \, f_{\text{lat}}(t - \tau) \, d\tau
    $$
    
    ---
    
    **Equation 17: Conditional Probability of Detection by Age**
    $$
    P_{\text{cond}}(t) = \int_{0}^{t} f_{\text{det}}(\tau) \, d\tau
    $$
    
    ---
    
    **Equation 18: Unconditional Probability of Detection by Age**
    $$
    P_{\text{uncond}}(t) = R_{\text{lifetime}} \cdot P_{\text{cond}}(t)
    $$
    - $R_{\text{lifetime}} \approx 0.70-0.80$ for MLH1/MSH2
    
    ---
    
    **Lifetime Risk by Gene**
    - MLH1: 70-80%
    - MSH2: 70-80%
    - MSH6: 50-60%
    - PMS2: 15-20%
    
    ---
    
    ### Summary of All Parameters
    
    | Parameter | Meaning | Value |
    |-----------|---------|-------|
    | $L$ | Lower bound | 1.0 mm |
    | $U$ | Upper bound | 60.0 mm |
    | $r$ | Growth rate | 0.0426 /week |
    | $\sigma_{\text{floor}}$ | Minimum noise | 0.5 mm |
    | $\sigma_{\text{rel}}$ | Relative noise | 0.20 |
    | $t_{\text{delay}}$ | Immunotherapy delay | 1.5-3.0 weeks |
    | $k_{\text{inc}}$ | Incubation shape | 4-6 |
    | $\theta_{\text{inc}}$ | Incubation scale | 5-8 years |
    | $k_{\text{lat}}$ | Latency shape | 2-4 |
    | $\theta_{\text{lat}}$ | Latency scale | 1-3 years |
    
    *Model parameters calibrated to published trial data (GARNET, KEYNOTE-177) and Lynch syndrome epidemiology.*
    """)

# ============================================================
# EXPANDER 4: CLINICAL CASE
# ============================================================
with st.expander("📋 Clinical Case: Benign Polyp", expanded=False):
    st.markdown("""
    **Patient:** Lynch Syndrome on Immunotheraopy (Dostarlimab Infusion 500mg IV every treeb weeks). A Sessile Polyp of the descending colon (712728006), estimated initial size (~50-55mm), and about 30cm from anal verge. Could not be removed (ESD failed).
    
    **Outcome:** Polyp shrank progressively (slower rate than Population mean). Third colonoscopy removed it successfully after shrinkage. No dysplasia or malignancy.
    
    **Validation:** Both fast shrinkage rate for MLH1 tumor and slower shrinkage rate for benign polyp fell within model's predictions 90% Credible Interval.
    """)
    if os.path.exists("benign_polyp_STEV.png"):
        st.image("benign_polyp_STEV.png", caption="STEV projection vs. actual measurements", use_container_width=True)
  #     st.image("benign_polyp_STEV.png", caption="~20mm to ~1.5mm in 9.3 weeks", width=400)
# ============================================================
# PARAMETERS
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

def predict_forward(genotype, week):
    mu = means[week][genotype]
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
    st.image(buf.getvalue(), width=150, caption="Scan with phone camera")
    
    st.markdown("**Or copy link:**")
    st.code(app_url, language="text")
    
    st.markdown("### ℹ️ How to use")
    st.markdown("- **Size -> Genotype:** Enter Time and Size, get Genotype")
    st.markdown("- **Genotype -> Size:** Enter Time and Select Genotype, get Size Mean/Spread")
    st.markdown("- **Expanders:** Click to view curves, dynamics, math, and cases")
    st.markdown("---")
    st.markdown("**STEV model** - Lynch Syndrome")
    st.markdown("*Horatio Quinones / Sherry Johnson et al*")

# ============================================================
# MAIN APP WITH TWO TABS
# ============================================================
tab1, tab2 = st.tabs(["🔍 Size -> Genotype", "🔮 Genotype -> Size"])

# ========== TAB 1: SIZE -> GENOTYPE ==========
with tab1:
    col_left, col_right = st.columns(2)
    with col_left:
        week = st.selectbox("📅 Week", weeks, index=8)
    with col_right:
        size = st.slider("📏 Tumor size (mm)", 0.0, 30.0, 1.4, 0.1)

    if st.button("Predict Genotype", use_container_width=True):
        probs = predict_inverse(size, week)
        most_likely = max(probs, key=probs.get)
        st.session_state.show_prediction = True
        st.session_state.prediction_probs = probs
        st.session_state.prediction_most_likely = most_likely
        st.session_state.prediction_week = week
        st.session_state.prediction_size = size

    # Display prediction if it exists
    if st.session_state.show_prediction and st.session_state.prediction_probs:
        probs = st.session_state.prediction_probs
        most_likely = st.session_state.prediction_most_likely
        week = st.session_state.prediction_week
        size = st.session_state.prediction_size
        
        col_a, col_b = st.columns(2)
        col_a.metric("🧬 Most likely Genotype", most_likely)
        col_b.metric("📊 Probability", f"{probs[most_likely]:.1%}")
        
        df = pd.DataFrame(list(probs.items()), columns=['Genotype', 'Probability'])
        fig = px.bar(df, x='Genotype', y='Probability', color='Genotype',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title=f'Week {week}, size = {size} mm')
        fig.update_layout(yaxis_title='Posterior probability', xaxis_title='genotype')
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("⚠️ Research & education only - not medical advice")

# ========== TAB 2: GENOTYPE -> SIZE ==========
with tab2:
    col_left, col_right = st.columns(2)
    with col_left:
        week = st.selectbox("📅 Week", weeks, index=8, key="forward_week")
    with col_right:
        genotype = st.selectbox("🧬 Genotype", names, index=1)

    if st.button("Predict Size", use_container_width=True):
        mu, sigma, ci = predict_forward(genotype, week)
        
        col_a, col_b = st.columns(2)
        col_a.metric("📏 Predicted mean size", f"{mu:.2f} mm")
        col_b.metric("📊 95% credible interval", f"[{ci[0]:.2f}, {ci[1]:.2f}] mm")
        
        x_vals = np.linspace(max(0, mu - 4*sigma), mu + 4*sigma, 200)
        y_vals = norm.pdf(x_vals, mu, sigma)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', line_color='#1e466e', name='Density'))
        fig.add_vline(x=mu, line_dash="dash", line_color="red", annotation_text=f"Mean = {mu:.2f} mm")
        fig.update_layout(title=f'{genotype} at week {week}',
                          xaxis_title='Tumor size (mm)',
                          yaxis_title='Probability density')
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("⚠️ Research & education only - not medical advice")
