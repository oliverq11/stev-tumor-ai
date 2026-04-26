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
# TRAJECTORY DATA (from your STEV model)
# ============================================================
# Weeks from 0 to 48
weeks_list = list(range(0, 49))

# Starting sizes: 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 mm
starting_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Trajectory data from your table (mean tumor size at each week for each starting size)
trajectory_data = {
    10: [10, 10, 9, 7.886, 5.188, 3.349, 2.179, 1.467, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    15: [15, 15, 14, 12.416, 8.662, 5.748, 3.718, 2.408, 1.605, 1.127, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    20: [20, 20, 19, 17.491, 13.257, 9.367, 6.268, 4.066, 2.627, 1.737, 1.205, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    25: [25, 25, 24, 23.214, 19.62, 15.457, 11.314, 7.768, 5.105, 3.294, 2.145, 1.447, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    30: [30, 30, 30, 29.718, 29.014, 27.856, 26.032, 23.354, 19.795, 15.647, 11.489, 7.908, 5.204, 3.359, 2.185, 1.471, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    35: [35, 35, 35, 34.864, 34.562, 34.122, 33.488, 32.587, 31.334, 29.637, 27.429, 24.692, 21.5, 18.033, 14.544, 11.293, 8.475, 6.181, 4.407, 3.088, 2.137, 1.465, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    40: [40, 40, 40, 39.897, 39.684, 39.399, 39.017, 38.509, 37.839, 36.965, 35.839, 34.413, 32.647, 30.517, 28.03, 25.233, 22.218, 19.112, 16.059, 13.193, 10.616, 8.388, 6.525, 5.012, 3.81, 2.873, 2.153, 1.606, 1.193, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    45: [45, 45, 45, 44.915, 44.744, 44.526, 44.246, 43.889, 43.435, 42.86, 42.138, 41.238, 40.127, 38.775, 37.152, 35.242, 33.042, 30.569, 27.868, 25.006, 22.069, 19.155, 16.357, 13.755, 11.405, 9.338, 7.562, 6.068, 4.831, 3.822, 3.008, 2.358, 1.842, 1.435, 1.116, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    50: [50, 50, 49, 49.925, 49.778, 49.595, 49.367, 49.082, 48.729, 48.291, 47.751, 47.089, 46.282, 45.304, 44.131, 42.739, 41.107, 39.223, 37.083, 34.702, 32.107, 29.346, 26.481, 23.583, 20.729, 17.99, 15.424, 13.078, 10.977, 9.131, 7.536, 6.178, 5.036, 4.086, 3.302, 2.66, 2.137, 1.713, 1.371, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    55: [55, 55, 55, 54.931, 54.8, 54.638, 54.44, 54.197, 53.9, 53.539, 53.099, 52.566, 51.922, 51.149, 50.226, 49.132, 47.845, 46.346, 44.621, 42.66, 40.465, 38.049, 35.437, 32.668, 29.795, 26.877, 23.978, 21.16, 18.48, 15.981, 13.695, 11.639, 9.82, 8.232, 6.862, 5.692, 4.703, 3.872, 3.179, 2.603, 2.128, 1.736, 1.415, 1.152, 1.1, 1.1, 1.1, 1.1, 1.1],
    60: [60, 59, 59, 58.935, 58.812, 58.662, 58.481, 58.26, 57.994, 57.672, 57.283, 56.816, 56.256, 55.587, 54.792, 53.852, 52.747, 51.459, 49.969, 48.263, 46.332, 44.175, 41.799, 39.224, 36.481, 33.612, 30.668, 27.705, 24.783, 21.956, 19.273, 16.771, 14.477, 12.406, 10.561, 8.939, 7.527, 6.311, 5.271, 4.389, 3.644, 3.019, 2.496, 2.06, 1.699, 1.399, 1.151, 1.1, 1.1],
}

# Create DataFrame for easy lookup
df_trajectory = pd.DataFrame(trajectory_data, index=weeks_list)
df_trajectory.index.name = "Week"

# TMB distribution parameters by genotype
tmb_distribution = {
    'POLE': {'mean': 100, 'std': 25},
    'MLH1': {'mean': 55, 'std': 12.5},
    'MSH2': {'mean': 50, 'std': 12.5},
    'MSIH': {'mean': 45, 'std': 10},
    'MSH6': {'mean': 25, 'std': 8}
}

# Genotype names
names = ['POLE', 'MLH1', 'MSH2', 'MSIH', 'MSH6']

# Population priors for genotypes (for Tab 1)
genotype_prior = {
    'POLE': 0.03,
    'MLH1': 0.40,
    'MSH2': 0.40,
    'MSIH': 0.02,
    'MSH6': 0.15,
}

# Hazard ratios for inverse prediction (from your code)
HR = {'POLE': 1.159, 'MLH1': 1.127, 'MSH2': 1.117, 'MSIH': 1.099, 'MSH6': 1.091}
S0_ref = 23.0

# Helper functions
def normal_pdf(x, mu, sigma):
    if sigma <= 0:
        return 1e-10
    return np.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

def interpolate_trajectory(initial_size, week):
    """Interpolate between starting sizes to get expected size at given week"""
    sizes = sorted(starting_sizes)
    if initial_size <= sizes[0]:
        return df_trajectory.loc[week, sizes[0]]
    if initial_size >= sizes[-1]:
        return df_trajectory.loc[week, sizes[-1]]
    
    for i in range(len(sizes) - 1):
        if sizes[i] <= initial_size <= sizes[i + 1]:
            size_low = sizes[i]
            size_high = sizes[i + 1]
            size_low_val = df_trajectory.loc[week, size_low]
            size_high_val = df_trajectory.loc[week, size_high]
            fraction = (initial_size - size_low) / (size_high - size_low)
            return size_low_val + fraction * (size_high_val - size_low_val)
    return df_trajectory.loc[week, 60]

def predict_size_from_genotype(genotype, week, initial_size):
    """
    Predict tumor size based on genotype, week, and initial size.
    Uses interpolated trajectory from STEV model, then applies TMB adjustment.
    """
    # Get the expected size from the trajectory (TMB=55 reference)
    base_size = interpolate_trajectory(initial_size, week)
    
    # TMB distribution for this genotype
    tmb_mean = tmb_distribution[genotype]['mean']
    tmb_std = tmb_distribution[genotype]['std']
    
    # TMB adjustment factor (reference = 55)
    tmb_factor = (55 / tmb_mean) ** 0.25
    adjusted_size = max(1.1, base_size * tmb_factor)
    
    # Generate weighted distribution for predictive interval
    tmb_grid = np.linspace(max(1, tmb_mean - 3*tmb_std), tmb_mean + 3*tmb_std, 200)
    tmb_weights = norm.pdf(tmb_grid, tmb_mean, tmb_std)
    tmb_weights = tmb_weights / tmb_weights.sum()
    
    sizes = []
    for tmb in tmb_grid:
        factor = (55 / max(tmb, 0.1)) ** 0.25
        sizes.append(max(1.1, base_size * factor))
    sizes = np.array(sizes)
    
    # Sort for percentiles
    sorted_idx = np.argsort(sizes)
    sorted_sizes = sizes[sorted_idx]
    sorted_weights = tmb_weights[sorted_idx]
    cum_weights = np.cumsum(sorted_weights)
    
    lower_idx = np.searchsorted(cum_weights, 0.025)
    upper_idx = np.searchsorted(cum_weights, 0.975)
    lower_size = sorted_sizes[min(lower_idx, len(sorted_sizes)-1)]
    upper_size = sorted_sizes[min(upper_idx, len(sorted_sizes)-1)]
    
    return adjusted_size, (lower_size, upper_size), sizes, tmb_weights

def predict_inverse(size, week):
    """Size -> Genotype prediction (from your original code)"""
    # Use hazard ratios with reference S0=23
    unnorm = {}
    for name in names:
        # Scale the mean based on the reference
        mu = S0_ref + HR[name] * (interpolate_trajectory(S0_ref, week) - S0_ref)
        like = normal_pdf(size, mu, 2.0)  # Simplified sigma for inverse
        unnorm[name] = like * genotype_prior[name]
    total = sum(unnorm.values())
    if total == 0:
        return {name: 0.2 for name in names}
    return {name: unnorm[name]/total for name in names}
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
    
    - **First hit (inherited mutation):** A person with Lynch syndrome is born with **one faulty copy** of an MMR gene (e.g., MLH1, MSH2) inherited from a parent. This alone does not cause cancer - it only creates a **predisposition**.
    
    - **Second hit (acquired mutation):** At some point later in life, the **second healthy copy** of that MMR gene is damaged or lost. When this happens, the cell can no longer repair DNA mistakes, leading to microsatellite instability (MSI) and eventually **tumor formation**.
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
# EXPANDER 3: MATHEMATICAL FRAMEWORK (simplified)
# ============================================================
with st.expander("📐 Mathematical Framework of the STEV Model", expanded=False):
    st.markdown(r"""
    ### A True Stochastic Process
    
    The STEV model is a **purely stochastic simulation**. At each weekly cycle, tumor growth or shrinkage is random.
    
    **Key Equations:**
    
    1. **Logit Transformation:** $Z = \ln((S-L)/(U-S))$, $L=1.0$ mm, $U=60.0$ mm
    
    2. **Stochastic Process:** $Z_{t+1} = Z_t + \Delta Z_t$, $\mathbb{E}[\Delta Z_t] = r$, $\text{Var}(\Delta Z_t) = \sigma_{\text{cycle}}^2(S_t)$
    
    3. **Growth Mean Path:** $\mu_Z(t) = \alpha + r \cdot t$, $r=0.0426$/week
    
    4. **TMB Adjustment:** $S_{\text{adj}} = S_{\text{base}} \cdot (55/\text{TMB})^{0.25}$
    
    5. **95% Predictive Interval:** From TMB distribution sampling
    
    *Full details available in the code repository.*
    """)

# ============================================================
# EXPANDER 4: CLINICAL CASE
# ============================================================
with st.expander("📋 Clinical Case: Benign Polyp", expanded=False):
    st.markdown("""
    **Patient:** Lynch Syndrome on Immunotherapy (Dostarlimab). A Sessile Polyp of the descending colon, estimated initial size ~50-55mm. Could not be removed (ESD failed).
    
    **Outcome:** Polyp shrank progressively (slower rate than population mean). Third colonoscopy removed it successfully. No dysplasia or malignancy.
    
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
    col_left, col_right = st.columns(2)
    with col_left:
        week = st.selectbox("📅 Week", weeks_list[:25], index=8)  # Weeks 0-24
    with col_right:
        size = st.slider("📏 Tumor size (mm)", 0.0, 60.0, 1.4, 0.1)

    if st.button("Predict Genotype", use_container_width=True):
        probs = predict_inverse(size, week)
        most_likely = max(probs, key=probs.get)
        
        col_a, col_b = st.columns(2)
        col_a.metric("🧬 Most likely Genotype", most_likely)
        col_b.metric("📊 Probability", f"{probs[most_likely]:.1%}")
        
        df = pd.DataFrame(list(probs.items()), columns=['Genotype', 'Probability'])
        fig = px.bar(df, x='Genotype', y='Probability', color='Genotype',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title=f'Week {week}, size = {size} mm')
        fig.update_layout(yaxis_title='Posterior probability', xaxis_title='Genotype')
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("⚠️ Research & education only - not medical advice")

# ========== TAB 2: GENOTYPE -> SIZE ==========
with tab2:
    col_left, col_right = st.columns(2)
    with col_left:
        week = st.selectbox("📅 Week", weeks_list[:25], index=8, key="forward_week")
    with col_right:
        genotype = st.selectbox("🧬 Genotype", names, index=1)
    
    initial_size = st.slider("📏 Initial tumor size at week 0 (mm)", min_value=10.0, max_value=60.0, value=23.0, step=1.0, help="Baseline tumor size before treatment")

    if st.button("Predict Size", use_container_width=True):
        expected_size, ci, sizes, weights = predict_size_from_genotype(genotype, week, initial_size)
        
        col_a, col_b = st.columns(2)
        col_a.metric("📏 Expected size", f"{expected_size:.2f} mm")
        col_b.metric("📊 95% predictive interval", f"[{ci[0]:.2f}, {ci[1]:.2f}] mm")
        
        tmb_mean = tmb_distribution[genotype]['mean']
        tmb_std = tmb_distribution[genotype]['std']
        st.caption(f"💡 Based on {genotype} TMB distribution (mean={tmb_mean}, SD={tmb_std})")
        
        # Density plot from weighted samples
        from scipy import stats
        kde = stats.gaussian_kde(sizes, weights=weights)
        x_vals = np.linspace(max(0.5, ci[0]*0.8), ci[1]*1.2, 200)
        y_vals = kde(x_vals)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', line_color='#1e466e', name='Probability density'))
        fig.add_vline(x=expected_size, line_dash="dash", line_color="red", annotation_text=f"Expected = {expected_size:.2f} mm")
        fig.update_layout(title=f'{genotype} at week {week} (initial size = {initial_size} mm)',
                          xaxis_title='Tumor size (mm)',
                          yaxis_title='Probability density')
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("⚠️ Research & education only - not medical advice")
