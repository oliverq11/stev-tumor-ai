# standalone_genotype_predictor.py
# Uses ALL genotypes, sigma=0.75, no Bayesian priors

import numpy as np
import pandas as pd

# ============================================================
# DATA (copied from your app)
# ============================================================

# Starting sizes for cure data
starting_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Cure data: dictionary week -> list of sizes for each starting size
cure_data = {
    0: [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    1: [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 59],
    2: [9, 14, 19, 24, 30, 35, 40, 45, 49, 55, 59],
    3: [7.886, 12.416, 17.491, 23.214, 29.718, 34.864, 39.897, 44.915, 49.925, 54.931, 58.935],
    4: [5.188, 8.662, 13.257, 19.62, 29.014, 34.562, 39.684, 44.744, 49.778, 54.8, 58.812],
    5: [3.349, 5.748, 9.367, 15.457, 27.856, 34.122, 39.399, 44.526, 49.595, 54.638, 58.662],
    6: [2.179, 3.718, 6.268, 11.314, 26.032, 33.488, 39.017, 44.246, 49.367, 54.44, 58.481],
    7: [1.467, 2.408, 4.066, 7.768, 23.354, 32.587, 38.509, 43.889, 49.082, 54.197, 58.26],
    8: [1.1, 1.605, 2.627, 5.105, 19.795, 31.334, 37.839, 43.435, 48.729, 53.9, 57.994],
    9: [1.1, 1.127, 1.737, 3.294, 15.647, 29.637, 36.965, 42.86, 48.291, 53.539, 57.672],
    10: [1.1, 1.1, 1.205, 2.145, 11.489, 27.429, 35.839, 42.138, 47.751, 53.099, 57.283],
    11: [1.1, 1.1, 1.1, 1.447, 7.908, 24.692, 34.413, 41.238, 47.089, 52.566, 56.816],
    12: [1.1, 1.1, 1.1, 1.1, 5.204, 21.5, 32.647, 40.127, 46.282, 51.922, 56.256],
    13: [1.1, 1.1, 1.1, 1.1, 3.359, 18.033, 30.517, 38.775, 45.304, 51.149, 55.587],
    14: [1.1, 1.1, 1.1, 1.1, 2.185, 14.544, 28.03, 37.152, 44.131, 50.226, 54.792],
    15: [1.1, 1.1, 1.1, 1.1, 1.471, 11.293, 25.233, 35.242, 42.739, 49.132, 53.852],
    16: [1.1, 1.1, 1.1, 1.1, 1.1, 8.475, 22.218, 33.042, 41.107, 47.845, 52.747],
    17: [1.1, 1.1, 1.1, 1.1, 1.1, 6.181, 19.112, 30.569, 39.223, 46.346, 51.459],
    18: [1.1, 1.1, 1.1, 1.1, 1.1, 4.407, 16.059, 27.868, 37.083, 44.621, 49.969],
    19: [1.1, 1.1, 1.1, 1.1, 1.1, 3.088, 13.193, 25.006, 34.702, 42.66, 48.263],
    20: [1.1, 1.1, 1.1, 1.1, 1.1, 2.137, 10.616, 22.069, 32.107, 40.465, 46.332],
    21: [1.1, 1.1, 1.1, 1.1, 1.1, 1.465, 8.388, 19.155, 29.346, 38.049, 44.175],
    22: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 6.525, 16.357, 26.481, 35.437, 41.799],
    23: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 5.012, 13.755, 23.583, 32.668, 39.224],
    24: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 3.81, 11.405, 20.729, 29.795, 36.481],
    25: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 2.873, 9.338, 17.99, 26.877, 33.612],
    26: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 2.153, 7.562, 15.424, 23.978, 30.668],
    27: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.606, 6.068, 13.078, 21.16, 27.705],
    28: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.193, 4.831, 10.977, 18.48, 24.783],
    29: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 3.822, 9.131, 15.981, 21.956],
    30: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 3.008, 7.536, 13.695, 19.273],
    31: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 2.358, 6.178, 11.639, 16.771],
    32: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.842, 5.036, 9.82, 14.477],
    33: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.435, 4.086, 8.232, 12.406],
    34: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.116, 3.302, 6.862, 10.561],
    35: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 2.66, 5.692, 8.939],
    36: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 2.137, 4.703, 7.527],
    37: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.713, 3.872, 6.311],
    38: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.371, 3.179, 5.271],
    39: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 2.603, 4.389],
    40: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 2.128, 3.644],
    41: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.736, 3.019],
    42: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.415, 2.496],
    43: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.152, 2.06],
    44: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.699],
    45: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.399],
    46: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.151],
    47: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    48: [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
}

# Genotype hazard ratios (relative growth/scaling factors)
HR = {
    'POLE': 1.159,
    'MLH1': 1.127,
    'MSH2': 1.117,
    'MSIH': 1.099,
    'MSH6': 1.091,
    'PMS2': 1.000   # Reference (as per literature, PMS2 has slower progression)
}

# List of all genotypes we have data for
all_genotypes = ['POLE', 'MLH1', 'MSH2', 'MSIH', 'MSH6', 'PMS2']

# Standard deviation for likelihood (fixed at 0.75 as requested)
SIGMA = 0.75

def get_expected_size(week, initial_size, genotype):
    """
    Return expected tumor size (mm) for given week, initial_size, and genotype.
    Uses the cure_data table and applies genotype-specific scaling.
    """
    if week not in cure_data:
        week = max(cure_data.keys())
    
    week_data = cure_data[week]
    
    # Interpolate for initial_size (starting sizes 10-60 mm)
    if initial_size <= 10:
        base_expected = week_data[0]
    elif initial_size >= 60:
        base_expected = week_data[-1]
    else:
        # Linear interpolation between starting sizes
        for i in range(len(starting_sizes) - 1):
            if starting_sizes[i] <= initial_size <= starting_sizes[i+1]:
                low_s = starting_sizes[i]
                high_s = starting_sizes[i+1]
                low_v = week_data[i]
                high_v = week_data[i+1]
                frac = (initial_size - low_s) / (high_s - low_s)
                base_expected = low_v + frac * (high_v - low_v)
                break
    
    # Apply genotype scaling (relative to MLH1)
    # You can also scale relative to a reference; here we use MLH1 as reference
    hr_factor = HR[genotype] / HR['MLH1']
    expected = base_expected * hr_factor
    
    # Ensure minimum size is 1.1 mm (cure floor)
    expected = max(1.1, expected)
    
    return expected

def likelihood(current_size, expected_size, sigma=SIGMA):
    """
    Normal likelihood (probability density) given observed size and expected size.
    sigma = 0.75 mm.
    """
    return np.exp(-0.5 * ((current_size - expected_size) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def predict_genotype(week, initial_size, current_size):
    """
    Predict the most likely genotype based on observed current size.
    Returns: dict of likelihoods (not normalized to probabilities)
    """
    results = {}
    for g in all_genotypes:
        exp_size = get_expected_size(week, initial_size, g)
        like = likelihood(current_size, exp_size)
        results[g] = like
    return results

# ============================================================
# EXAMPLE PREDICTIONS AND TEST CASES
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Genotype Predictor (sigma=0.75, no Bayesian prior)")
    print("=" * 60)
    
    # Test case 1: MLH1-like tumor at week 8, initial size 30mm, current size ~15mm
    print("\n--- Test 1: MLH1-like shrinking tumor ---")
    week = 8
    init = 30
    current = 16  # typical size for MLH1 at week 8 from table (~19.8mm, but let's use 16)
    results = predict_genotype(week, init, current)
    
    # Sort by likelihood (highest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(f"Week={week}, initial_size={init}mm, current_size={current}mm")
    print("Likelihoods (relative, not normalized):")
    for g, like in sorted_results:
        print(f"  {g}: {like:.6e}")
    print(f"Most likely: {sorted_results[0][0]}")
    
    # Test case 2: POLE-like tumor (expected to be larger at same week)
    print("\n--- Test 2: POLE-like large tumor ---")
    week = 8
    init = 30
    current = 28  # larger than MLH1 expected (~19.8mm)
    results = predict_genotype(week, init, current)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(f"Week={week}, initial_size={init}mm, current_size={current}mm")
    print("Likelihoods (relative, not normalized):")
    for g, like in sorted_results:
        print(f"  {g}: {like:.6e}")
    print(f"Most likely: {sorted_results[0][0]}")
    
    # Test case 3: at week 0 (no shrinkage yet)
    print("\n--- Test 3: Week 0 (no shrinkage) ---")
    week = 0
    init = 30
    current = 30
    results = predict_genotype(week, init, current)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(f"Week={week}, initial_size={init}mm, current_size={current}mm")
    print("Likelihoods (relative, not normalized):")
    for g, like in sorted_results:
        print(f"  {g}: {like:.6e}")
    print(f"Most likely: {sorted_results[0][0]}")
    
    # Test case 4: a small residual tumor after many weeks
    print("\n--- Test 4: Small residual tumor (cure nearly complete) ---")
    week = 30
    init = 30
    current = 3.0
    results = predict_genotype(week, init, current)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(f"Week={week}, initial_size={init}mm, current_size={current}mm")
    print("Likelihoods (relative, not normalized):")
    for g, like in sorted_results:
        print(f"  {g}: {like:.6e}")
    print(f"Most likely: {sorted_results[0][0]}")
    
    print("\n" + "=" * 60)
    print("To test your own values, modify the script or call predict_genotype()")
    print("with (week, initial_size, current_size).")
What this script does:
Uses all 6 genotypes (POLE, MLH1, MSH2, MSIH, MSH6, PMS2).

Uses sigma = 0.75 for the likelihood.

Uses your cure_data dictionary exactly as in your app.

Computes expected size for each genotype by interpolating the cure table and applying the genotype‑specific hazard ratio.

Computes a normal likelihood (no Bayesian prior).

Returns the most likely genotype (highest likelihood).

How to run:
Save the code as standalone_genotype_predictor.py

Run it with python standalone_genotype_predictor.py

It will print the results for 4 test cases.

To test your own values:
Modify the __main__ section or call the function directly:

python
results = predict_genotype(week=8, initial_size=30, current_size=16)
This script is independent of your Streamlit app. Once you confirm the results match your expectations, we can debug why the app version was giving incorrect answers.


