import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="STEV Tumor Response AI", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .author {
        font-size: 14px;
        color: #666;
        margin-top: -20px;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        font-weight: normal;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 STEV: Stochastic Tumor Response to Immunotherapy")
st.markdown('<div class="subtitle">Lynch Syndrome Colorectal Tumors</div>', unsafe_allow_html=True)
st.markdown('<div class="author">Horatio Quinones / Sherry Johnson</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# 1. PARAMETERS
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

# ============================================================
# 2. INVERSE PREDICTION
# ============================================================
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

# ============================================================
# 3. FORWARD PREDICTION
# ============================================================
def predict_forward(biology, week):
    mu = means[week][biology]
    sigma = sigma_env[week]
    ci_95 = (mu - 1.96*sigma, mu + 1.96*sigma)
    return mu, sigma, ci_95

# ============================================================
# 4. UI
# ============================================================
mode = st.radio("Select prediction mode", ["Inverse: Size → Biology", "Forward: Biology → Size"])

if mode == "Inverse: Size → Biology":
    col1, col2 = st.columns(2)
    with col1:
        week = st.selectbox("Week", weeks, index=8)
    with col2:
        size = st.number_input("Tumor size (mm)", min_value=0.0, max_value=50.0, value=1.4, step=0.1)
    
    if st.button("Predict Biology"):
        probs = predict_inverse(size, week)
        most_likely = max(probs, key=probs.get)
        st.metric("Most likely biology", most_likely)
        
        fig, ax = plt.subplots()
        ax.bar(names, [probs[n] for n in names], color='skyblue')
        ax.set_ylabel('Posterior probability')
        ax.set_title(f'Week {week}, size = {size} mm')
        st.pyplot(fig)

else:
    col1, col2 = st.columns(2)
    with col1:
        week = st.selectbox("Week", weeks, index=8)
    with col2:
        biology = st.selectbox("Biology", names, index=1)
    
    if st.button("Predict Size"):
        mu, sigma, ci = predict_forward(biology, week)
        st.metric("Predicted mean size", f"{mu:.2f} mm")
        st.write(f"**95% credible interval:** [{ci[0]:.2f}, {ci[1]:.2f}] mm")
        
        x = np.linspace(max(0, mu - 4*sigma), mu + 4*sigma, 200)
        y = norm.pdf(x, mu, sigma)
        fig, ax = plt.subplots()
        ax.plot(x, y, color='blue')
        ax.fill_between(x, y, where=(x>=ci[0]) & (x<=ci[1]), color='blue', alpha=0.3)
        ax.axvline(mu, color='red', linestyle='--', label=f'Mean = {mu:.2f} mm')
        ax.set_xlabel('Tumor size (mm)')
        ax.set_ylabel('Density')
        ax.set_title(f'{biology} at week {week}')
        ax.legend()
        st.pyplot(fig)