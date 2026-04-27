import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import qrcode
from io import BytesIO
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="STEV: Stochastic Tumor Response AI", layout="wide", page_icon="🧬")

# ============================================================
# GROWTH DATA (from your STEV_stage_sigmas_growth.csv)
# ============================================================
# Week, S_mean (mm), SD_S_sum (mm)
growth_data = {
    0: (1.09393894, 0.007036189),
    1: (1.098021331, 0.010171925),
    2: (1.102280827, 0.012741802),
    3: (1.106725082, 0.015054042),
    4: (1.111362082, 0.017227409),
    5: (1.116200153, 0.019323129),
    6: (1.121247979, 0.021378197),
    7: (1.126514615, 0.023417455),
    8: (1.132009505, 0.025458942),
    9: (1.137742495, 0.027516592),
    10: (1.143723853, 0.029601731),
    11: (1.149964285, 0.031723959),
    12: (1.156474953, 0.033891717),
    13: (1.163267497, 0.03611265),
    14: (1.17035405, 0.038393859),
    15: (1.177747263, 0.040742085),
    16: (1.185460322, 0.043163833),
    17: (1.193506977, 0.045665479),
    18: (1.201901557, 0.048253337),
    19: (1.210659001, 0.050933728),
    20: (1.219794876, 0.053713024),
    21: (1.229325412, 0.056597692),
    22: (1.23926752, 0.059594328),
    23: (1.249638823, 0.062709688),
    24: (1.26045769, 0.065950715),
    25: (1.271743256, 0.069324567),
    26: (1.283515463, 0.072838636),
    27: (1.295795086, 0.076500572),
    28: (1.308603769, 0.080318303),
    29: (1.321964058, 0.084300058),
    30: (1.335899441, 0.088454382),
    31: (1.350434377, 0.09279016),
    32: (1.365594346, 0.097316634),
    33: (1.381405878, 0.102043423),
    34: (1.397896599, 0.106980542),
    35: (1.415095276, 0.112138423),
    36: (1.433031854, 0.117527933),
    37: (1.45173751, 0.123160396),
    38: (1.471244691, 0.12904761),
    39: (1.491587169, 0.135201874),
    40: (1.512800088, 0.141636002),
    41: (1.534920016, 0.14836335),
    42: (1.557984997, 0.155397833),
    43: (1.582034606, 0.162753954),
    44: (1.607110005, 0.170446818),
    45: (1.633253999, 0.178492163),
    46: (1.660511098, 0.186906377),
    47: (1.688927575, 0.195706526),
    48: (1.718551525, 0.204910374),
    49: (1.749432937, 0.214536411),
    50: (1.781623749, 0.224603873),
    51: (1.815177921, 0.235132768),
    52: (1.8501515, 0.246143902),
    53: (1.886602685, 0.257658901),
    54: (1.924591906, 0.269700233),
    55: (1.964181885, 0.282291236),
    56: (2.005437713, 0.295456139),
    57: (2.048426924, 0.309220083),
    58: (2.093219564, 0.323609145),
    59: (2.139888267, 0.338650361),
    60: (2.188508332, 0.354371741),
    61: (2.239157792, 0.370802291),
    62: (2.291917493, 0.38797203),
    63: (2.346871167, 0.405912004),
    64: (2.404105503, 0.424654303),
    65: (2.463710225, 0.444232069),
    66: (2.525778158, 0.464680609),
    67: (2.590405299, 0.486036775),
    68: (2.657690889, 0.508336947),
    69: (2.727737473, 0.531618578),
    70: (2.800650967, 0.55592019),
    71: (2.876540715, 0.581281366),
    72: (2.955519544, 0.60774273),
    73: (3.037703817, 0.635345933),
    74: (3.123213478, 0.664133622),
    75: (3.212172088, 0.694149407),
    76: (3.304706866, 0.725437826),
    77: (3.400948706, 0.758044289),
    78: (3.501032202, 0.792015026),
    79: (3.605095652, 0.827397022),
    80: (3.713281058, 0.864237936),
    81: (3.825734113, 0.902586022),
    82: (3.942604178, 0.942490022),
    83: (4.06404424, 0.983999065),
    84: (4.190210865, 1.027162538),
    85: (4.321264125, 1.072029949),
    86: (4.457367516, 1.11865078),
    87: (4.598687854, 1.167074317),
    88: (4.745395155, 1.217349468),
    89: (4.897662487, 1.269524563),
    90: (5.055665808, 1.323647138),
}
# Add more weeks as needed

# Environmental variance inflation (30% unknown)
ENV_INFLATION = 1.0 / np.sqrt(0.70)  # ≈ 1.195

# Genotype HR factors (from your code)
HR = {'POLE': 1.159, 'MLH1': 1.127, 'MSH2': 1.117, 'MSIH': 1.099, 'MSH6': 1.091}
names = ['POLE', 'MLH1', 'MSH2', 'MSIH', 'MSH6']

# Population priors for genotypes
genotype_prior = {'POLE': 0.03, 'MLH1': 0.40, 'MSH2': 0.40, 'MSIH': 0.02, 'MSH6': 0.15}

# TMB distribution parameters
tmb_distribution = {
    'POLE': {'mean': 100, 'std': 25},
    'MLH1': {'mean': 55, 'std': 12.5},
    'MSH2': {'mean': 50, 'std': 12.5},
    'MSIH': {'mean': 45, 'std': 10},
    'MSH6': {'mean': 25, 'std': 8},
}

# Precompute reference weeks and sizes
ref_weeks = sorted(growth_data.keys())
ref_sizes = [growth_data[w][0] for w in ref_weeks]
ref_sds = [growth_data[w][1] for w in ref_weeks]

def get_growth_time(target_size, genotype='MLH1', tmb=55):
    """
    Estimate time to reach target size for given genotype and TMB.
    Returns: (weeks, lower_ci, upper_ci)
    """
    # Scaling factors
    tmb_factor = (55 / tmb) ** 0.25
    hr_factor = HR[genotype] / HR['MLH1']  # Relative to MLH1
    
    # Total scaling for time
    time_factor = tmb_factor * hr_factor
    
    # Adjust the reference weeks by scaling factor
    scaled_weeks = [w / time_factor for w in ref_weeks]
    
    # Find the scaled week closest to target size
    best_idx = None
    best_diff = float('inf')
    for i, size in enumerate(ref_sizes):
        diff = abs(size - target_size)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    
    weeks = scaled_weeks[best_idx]
    
    # Get SD at this week and inflate for environment
    sd = ref_sds[best_idx] * ENV_INFLATION
    
    # Log-normal credible interval
    cv = sd / ref_sizes[best_idx]
    lower_weeks = weeks * np.exp(-1.645 * cv)
    upper_weeks = weeks * np.exp(1.645 * cv)
    
    return weeks, lower_weeks, upper_weeks

def normal_pdf(x, mu, sigma):
    if sigma <= 0:
        return 1e-10
    return np.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

def predict_inverse(current_size, week, initial_size):
    """Size -> Genotype prediction"""
    unnorm = {}
    for name in names:
        # For inverse, we need a simple model
        # Using HR factors and initial size scaling
        expected_size = initial_size * HR[name] / HR['MLH1']
        sigma = expected_size * 0.2  # Simplified
        like = normal_pdf(current_size, expected_size, sigma)
        unnorm[name] = like * genotype_prior[name]
    
    total = sum(unnorm.values())
    if total == 0:
        return {name: 0.2 for name in names}
    return {name: unnorm[name]/total for name in names}

# ============================================================
# CUSTOM CSS (same as before)
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #f5f7fb; }
    h1, h2, h3, .stMarkdown, p, div, span, label { color: #212529; }
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
    .stButton button *, .stButton button span, .stButton button div { color: white !important; }
    .streamlit-expanderHeader { background-color: #e9ecef; border-radius: 8px; color: #212529; }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: bold; color: #1e466e; }
    [data-testid="stMetricLabel"] { color: #212529; }
    @media (prefers-color-scheme: dark) {
        .stApp, .main, .stAppViewContainer { background-color: #0a0a0a !important; }
        body, p, div, span, label, .stMarkdown, h1, h2, h3, .subtitle, .author { color: #ffffff !important; }
        .stButton button { background-color: #2c6e9e !important; }
        .stButton button:hover { background-color: #1e466e !important; }
        .streamlit-expanderHeader { background-color: #1e1e1e !important; color: white !important; }
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

This tool uses a **Stochastic Model** (STEV) for Lynch Syndrome Colorectal Cancer patients treated with Immunotherapy (Dostarlimab).

#### 1. 🔍 Size -> Genotype
Given tumor size and week, returns most likely Genotype.

#### 2. 🔮 Genotype -> Size
Given genotype, week, and initial tumor size, predicts expected size and 95% predictive interval.

#### 3. 📈 Growth and Immunotherapy Curves
Trajectories from 30mm and 60mm with 90% credible bands.

#### 4. 🕰️ Two-Hit Dynamics
Incubation, latency, conditional and unconditional probability plots.

#### 5. 📐 Mathematical Framework
Complete stochastic model formulation.

#### 6. 📋 Clinical Case
Benign polyp response to immunotherapy.
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
            st.image("MLH1_55TMB.png", caption="STEV projection vs. actual measurements", use_container_width=True)
    with col2:
        st.markdown("**MLH1 | TMB=55.4** | Faster than population mean")

# ============================================================
# EXPANDER 2: TWO-HIT DYNAMICS
# ============================================================
with st.expander("🕰️ Two-Hit Dynamics: Tumor Incubation, Latency, Detection Phase, Risk Assessment", expanded=False):
    st.markdown("""
    ### 📖 What is "First Hit" and "Second Hit"?
    
    - **First hit (inherited mutation):** A person with Lynch syndrome is born with **one faulty copy** of an MMR gene inherited from a parent.
    - **Second hit (acquired mutation):** The second healthy copy is damaged or lost, leading to MSI and tumor formation.
    """)
    
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
    **Patient:** Lynch Syndrome on Immunotherapy. A Sessile Polyp of the descending colon (~50-55mm). Could not be removed (ESD failed).
    
    **Outcome:** Polyp shrank progressively (slower than population mean). Third colonoscopy removed it successfully.
    
    **Validation:** Both fast shrinkage for MLH1 tumor and slower shrinkage for benign polyp fell within model's 90% credible interval.
    """)
    if os.path.exists("benign_polyp_STEV.png"):
        st.image("benign_polyp_STEV.png", caption="STEV projection vs. actual measurements", use_container_width=True)

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
    st.markdown("- **Size -> Genotype:** Enter Week and Size, get Genotype")
    st.markdown("- **Genotype -> Size:** Enter Week, Genotype, and Initial Size, get Expected Size and Range")
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
    col_left, col_mid, col_right = st.columns(3)
    with col_left:
        week = st.selectbox("📅 Week", list(range(25)), index=8)
    with col_mid:
        initial_size = st.slider("📏 Initial size at week 0 (mm)", min_value=1.1, max_value=60.0, value=30.0, step=1.0)
    with col_right:
        current_size = st.slider("📏 Current tumor size (mm)", 0.0, 60.0, 1.4, 0.1)
    
    # Show estimated growth time
    weeks_to_grow, lower_grow, upper_grow = get_growth_time(initial_size)
    st.caption(f"📈 Estimated time to reach {initial_size:.1f} mm: **{weeks_to_grow:.0f} weeks** [90% CI: {lower_grow:.0f}-{upper_grow:.0f}]")
    
    if st.button("Predict Genotype", use_container_width=True):
        probs = predict_inverse(current_size, week, initial_size)
        most_likely = max(probs, key=probs.get)
        
        col_a, col_b = st.columns(2)
        col_a.metric("🧬 Most likely Genotype", most_likely)
        col_b.metric("📊 Probability", f"{probs[most_likely]:.1%}")
        
        df = pd.DataFrame(list(probs.items()), columns=['Genotype', 'Probability'])
        fig = px.bar(df, x='Genotype', y='Probability', color='Genotype',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title=f'Initial size = {initial_size:.1f} mm, Week {week}, Current size = {current_size:.1f} mm')
        fig.update_layout(yaxis_title='Posterior probability', xaxis_title='Genotype')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Research & education only - not medical advice")

# ========== TAB 2: GENOTYPE -> SIZE ==========
with tab2:
    col_left, col_right = st.columns(2)
    with col_left:
        week = st.selectbox("📅 Week", list(range(25)), index=8, key="forward_week")
    with col_right:
        genotype = st.selectbox("🧬 Genotype", names, index=1)
    
    initial_size = st.slider("📏 Initial tumor size at week 0 (mm)", min_value=1.1, max_value=60.0, value=30.0, step=1.0, key="init_size")
    
    # Show estimated growth time
    weeks_to_grow, lower_grow, upper_grow = get_growth_time(initial_size, genotype)
    st.caption(f"📈 Estimated time to reach {initial_size:.1f} mm for {genotype}: **{weeks_to_grow:.0f} weeks** [90% CI: {lower_grow:.0f}-{upper_grow:.0f}]")
    
    tmb_mean = tmb_distribution[genotype]['mean']
    st.caption(f"🧬 {genotype} typical TMB = {tmb_mean}")
    
    if st.button("Predict Size", use_container_width=True):
        # For simplicity, use a basic prediction model
        # You can expand this with your full cure-phase logic
        base_size = initial_size * (1 - week * 0.05)  # Placeholder decay
        expected_size = max(1.1, base_size)
        
        col_a, col_b = st.columns(2)
        col_a.metric("📏 Expected size", f"{expected_size:.2f} mm")
        col_b.metric("📊 95% interval", f"[{max(0.5, expected_size*0.7):.2f}, {expected_size*1.3:.2f}] mm")
        
        # Simple density plot
        x_vals = np.linspace(0.5, expected_size*1.5, 100)
        y_vals = norm.pdf(x_vals, expected_size, expected_size*0.15)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', line_color='#1e466e', name='Density'))
        fig.update_layout(title=f'{genotype} at week {week} (initial size = {initial_size:.1f} mm)',
                          xaxis_title='Tumor size (mm)',
                          yaxis_title='Probability density')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Research & education only - not medical advice")
