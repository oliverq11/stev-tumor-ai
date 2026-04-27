import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, cumulative_trapezoid

np.random.seed(42)
# HQS 04/27/2026 morning early
# ------------------------------
# Second hit distribution (Gamma)
# ------------------------------
inc_mean = 44.0
inc_std = 10.0
inc_shape = (inc_mean / inc_std)**2
inc_scale = inc_std**2 / inc_mean
def sample_second_hit(n):
    return gamma.rvs(inc_shape, scale=inc_scale, size=n)

# ------------------------------
# Latency distribution (your table)
# ------------------------------
x_lat = np.array([0, 1, 2, 2.5, 3, 3.7, 5, 5.8, 7, 7.5, 8.8, 10, 11.25, 12.5, 13.25, 15, 17.5, 20])
y_lat = np.array([0.001, 0.01, 0.04, 0.11, 0.135, 0.12, 0.08, 0.09, 0.12, 0.115, 0.08, 0.05, 0.034, 0.03, 0.026, 0.022, 0.01, 0.001])

# Normalize density
density_interp = interp1d(x_lat, y_lat, kind='linear', fill_value=0, bounds_error=False)
x_dense = np.linspace(0, 20, 1000)
y_dense = density_interp(x_dense)
area = trapezoid(y_dense, x_dense)
y_norm = y_lat / area
density_norm = interp1d(x_lat, y_norm, kind='linear', fill_value=0, bounds_error=False)

# Build empirical CDF for sampling
cdf_x = np.linspace(0, 20, 1000)
pdf_vals = density_norm(cdf_x)
cdf_y = cumulative_trapezoid(pdf_vals, cdf_x, initial=0)
cdf_y = cdf_y / cdf_y[-1]
cdf_func = interp1d(cdf_y, cdf_x, kind='linear', fill_value=(0, 20), bounds_error=False)
def sample_latency(n):
    return cdf_func(np.random.rand(n))

# ------------------------------
# Monte Carlo convolution
# ------------------------------
n_samples = 1_000_000
second_hit = sample_second_hit(n_samples)
latency = sample_latency(n_samples)
detect_age = second_hit + latency

# ------------------------------
# Statistics
# ------------------------------
mean_age = np.mean(detect_age)
std_age = np.std(detect_age)
se_mean = std_age / np.sqrt(n_samples)
ci_low = mean_age - 1.96 * se_mean
ci_high = mean_age + 1.96 * se_mean

p2_5 = np.percentile(detect_age, 2.5)
p5 = np.percentile(detect_age, 5)
p25 = np.percentile(detect_age, 25)
p50 = np.percentile(detect_age, 50)
p75 = np.percentile(detect_age, 75)
p95 = np.percentile(detect_age, 95)
p97_5 = np.percentile(detect_age, 97.5)

print("--- Detection age distribution (individual variability) ---")
print(f"Mean: {mean_age:.1f} years")
print(f"Std dev: {std_age:.1f} years")
print(f"5th percentile: {p5:.1f} years")
print(f"Median (50th): {p50:.1f} years")
print(f"95th percentile: {p95:.1f} years")
print(f"95% prediction interval (2.5th–97.5th): {p2_5:.1f}–{p97_5:.1f} years")
print(f"\n95% CI for the mean: [{ci_low:.2f}, {ci_high:.2f}] years (very narrow)")

# ------------------------------
# Plot histogram with wide spread
# ------------------------------
plt.figure(figsize=(10, 6))
plt.hist(detect_age, bins=100, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Detection age distribution')
# Mark percentiles
plt.axvline(p5, color='orange', linestyle='--', linewidth=2, label='5th percentile')
plt.axvline(p50, color='green', linestyle='--', linewidth=2, label='Median (50th)')
plt.axvline(p95, color='orange', linestyle='--', linewidth=2, label='95th percentile')
# Shade the 95% prediction interval (2.5th–97.5th)
plt.axvspan(p2_5, p97_5, alpha=0.15, color='gray', label='95% prediction interval')
# Mean (the narrow CI is invisible at this scale)
plt.axvline(mean_age, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_age:.1f} y')
plt.xlabel('Age at detection (years)', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.title('Detection age distribution for MLH1 Lynch syndrome (individual variability)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('detection_age_wide_spread.png', dpi=150)
plt.show()