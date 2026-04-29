import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import qrcode
from io import BytesIO
import os
# HQ
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
    91: (5.219583776, 1.379763699),
    92: (5.389597532, 1.437919469),
    93: (5.565890463, 1.498158115),
    94: (5.748647928, 1.560521454),
    95: (5.938056964, 1.625049141),
    96: (6.134305952, 1.691778339),
    97: (6.33758426, 1.760743364),
    98: (6.548081841, 1.831975315),
    99: (6.765988811, 1.90550168),
    100: (6.991494978, 1.981345929),
    101: (7.224789342, 2.059527084),
    102: (7.466059555, 2.140059276),
    103: (7.715491344, 2.222951285),
    104: (7.973267898, 2.308206065),
    105: (8.239569208, 2.395820265),
    106: (8.514571383, 2.485783728),
    107: (8.798445913, 2.578079003),
    108: (9.091358909, 2.672680832),
    109: (9.393470296, 2.769555659),
    110: (9.704932979, 2.868661132),
    111: (10.02589198, 2.969945615),
    112: (10.35648352, 3.073347726),
    113: (10.69683413, 3.178795877),
    114: (11.04705968, 3.286207859),
    115: (11.4072644, 3.395490445),
    116: (11.77753994, 3.506539037),
    117: (12.15796434, 3.619237361),
    118: (12.54860105, 3.733457205),
    119: (12.94949797, 3.849058227),
    120: (13.36068639, 3.965887819),
    121: (13.78218012, 4.08378105),
    122: (14.21397446, 4.202560686),
    123: (14.65604536, 4.322037296),
    124: (15.10834851, 4.442009453),
    125: (15.5708185, 4.562264026),
    126: (16.04336813, 4.682576581),
    127: (16.52588764, 4.802711883),
    128: (17.01824413, 4.922424514),
    129: (17.520281, 5.041459595),
    130: (18.03181754, 5.159553622),
    131: (18.55264857, 5.276435418),
    132: (19.08254422, 5.391827187),
    133: (19.62124981, 5.505445677),
    134: (20.16848587, 5.617003448),
    135: (20.72394832, 5.726210222),
    136: (21.28730869, 5.832774332),
    137: (21.85821458, 5.936404234),
    138: (22.43629024, 6.03681009),
    139: (23.02113722, 6.1337054),
    140: (23.61233528, 6.226808668),
    141: (24.20944335, 6.31584509),
    142: (24.81200067, 6.400548248),
    143: (25.41952808, 6.480661786),
    144: (26.03152936, 6.555941063),
    145: (26.64749281, 6.62615475),
    146: (27.26689284, 6.691086366),
    147: (27.8891917, 6.750535723),
    148: (28.51384135, 6.804320283),
    149: (29.14028528, 6.852276385),
    150: (29.76796055, 6.89426035),
    151: (30.39629973, 6.93014944),
    152: (31.02473302, 6.959842658),
    153: (31.65269027, 6.983261393),
    154: (32.27960303, 7.000349878),
    155: (32.90490667, 7.011075493),
    156: (33.52804234, 7.015428865),
    157: (34.14845897, 7.01342381),
    158: (34.76561523, 7.005097083),
    159: (35.37898131, 6.990507964),
    160: (35.98804074, 6.969737673),
    161: (36.59229203, 6.942888633),
    162: (37.19125022, 6.91008358),
    163: (37.78444833, 6.871464544),
    164: (38.37143867, 6.827191708),
    165: (38.951794, 6.777442156),
    166: (39.52510858, 6.722408536),
    167: (40.09099909, 6.66229765),
    168: (40.64910534, 6.597328978),
    169: (41.19909093, 6.527733169),
    170: (41.74064369, 6.453750499),
    171: (42.27347598, 6.375629331),
    172: (42.79732496, 6.293624566),
    173: (43.31195258, 6.207996127),
    174: (43.81714551, 6.119007469),
    175: (44.31271501, 6.026924139),
    176: (44.79849656, 5.932012384),
    177: (45.27434951, 5.834537838),
    178: (45.74015652, 5.734764268),
    179: (46.19582301, 5.632952411),
    180: (46.64127646, 5.529358895),
    181: (47.07646571, 5.424235242),
    182: (47.50136009, 5.317826974),
    183: (47.91594862, 5.210372806),
    184: (48.32023911, 5.102103936),
    185: (48.71425719, 4.993243423),
    186: (49.09804539, 4.884005666),
    187: (49.47166211, 4.774595964),
    188: (49.83518072, 4.665210164),
    189: (50.18868847, 4.55603439),
    190: (50.53228555, 4.447244856),
    191: (50.8660841, 4.339007739),
    192: (51.19020724, 4.231479135),
    193: (51.50478808, 4.124805065),
    194: (51.80996886, 4.019121541),
    195: (52.10589997, 3.914554692),
    196: (52.39273912, 3.811220924),
    197: (52.67065046, 3.70922713),
    198: (52.9398038, 3.608670929),
    199: (53.2003738, 3.509640942),
    200: (53.45253925, 3.412217092),
    201: (53.69648237, 3.316470921),
    202: (53.9323881, 3.22246593),
    203: (54.16044353, 3.130257934),
    204: (54.38083728, 3.039895415),
    205: (54.59375895, 2.951419897),
    206: (54.7993986, 2.864866311),
    207: (54.99794629, 2.780263369),
    208: (55.18959163, 2.697633929),
    209: (55.37452336, 2.616995365),
    210: (55.55292901, 2.538359919),
    211: (55.72499452, 2.461735053),
    212: (55.89090396, 2.387123792),
    213: (56.05083923, 2.314525051),
    214: (56.20497985, 2.24393395),
    215: (56.35350269, 2.175342128),
    216: (56.49658177, 2.108738025),
    217: (56.63438816, 2.044107166),
    218: (56.76708972, 1.981432423),
    219: (56.89485107, 1.920694263),
    220: (57.01783342, 1.861870986),
    221: (57.13619449, 1.804938941),
    222: (57.25008847, 1.749872732),
    223: (57.35966591, 1.69664541),
    224: (57.46507373, 1.64522865),
    225: (57.56645514, 1.595592911),
    226: (57.66394966, 1.547707589),
    227: (57.75769312, 1.501541148),
    228: (57.84781762, 1.457061248),
    229: (57.9344516, 1.414234854),
    230: (58.01771981, 1.373028336),
    231: (58.0977434, 1.33340756),
    232: (58.1746399, 1.295337966),
    233: (58.2485233, 1.258784636),
    234: (58.31950409, 1.22371236),
    235: (58.38768931, 1.190085686),
    236: (58.4531826, 1.157868969),
    237: (58.5160843, 1.127026413),
    238: (58.57649147, 1.097522104),
    239: (58.63449796, 1.069320048),
    240: (58.69019453, 1.042384197),
    241: (58.74366884, 1.016678479),
    242: (58.7950056, 0.992166821),
    243: (58.8442866, 0.968813179),
    244: (58.8915908, 0.946581564),
    245: (58.93699438, 0.925436069),
    246: (58.98057085, 0.905340899),
    247: (59.02239111, 0.886260402),
}
# Add more weeks as needed

# Environmental variance inflation (30% unknown)
ENV_INFLATION = 1.0 / np.sqrt(0.70)  # ≈ 1.195

# Genotype HR factors (from your code)
HR = {'POLE': 1.159, 'MLH1': 1.127, 'MSH2': 1.117, 'MSIH': 1.099, 'MSH6': 1.091, 'PMS2': 1.000}
names = ['POLE', 'MLH1', 'MSH2', 'MSIH', 'MSH6' , 'PMS2']

# Population priors for genotypes
genotype_prior = {'POLE': 0.03, 'MLH1': 0.40, 'MSH2': 0.40, 'MSIH': 0.02, 'MSH6': 0.15, 'PMS2': 0.11}

# TMB distribution parameters
tmb_distribution = {
    'POLE': {'mean': 100, 'std': 25},
    'MLH1': {'mean': 55, 'std': 12.5},
    'MSH2': {'mean': 50, 'std': 12.5},
    'MSIH': {'mean': 45, 'std': 10},
    'MSH6': {'mean': 25, 'std': 8},
    'PMS2': {'mean': 18, 'std': 6},
}


# ============================================================
# CURE PHASE DATA (shrinkage during immunotherapy)
# ============================================================
starting_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

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
   #= HR[genotype] / HR['MLH1']  # Relative to MLH1
    hr_factor = HR['MLH1'] / HR[genotype]
    
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
    """
    Returns posterior probability for each genotype.
    Uses inverted HR scaling for cure phase (week >= 3) and growth_data SD for early weeks.
    """
    unnorm = {}
    
    # Determine sigma based on week
    if week <= 2:
        # Growth phase: use SD from growth_data
        if week in growth_data:
            sigma = growth_data[week][1]
        else:
            sigma = 0.5
    else:
        # Cure phase: use normalized SD model
        norm_size = current_size / initial_size
        x = max(0.01, min(norm_size, 0.99))
        p = 0.45
        peak_sd = 3.75
        sigma = peak_sd * (x / p) * np.exp(1 - x / p) * (1 - x) / (1 - p)
        sigma = max(0.0, sigma)
    
    for name in names:
        # Get expected size from cure_data
        week_data = cure_data[week]
        
        if initial_size <= 10:
            expected_raw = week_data[0]
        elif initial_size >= 60:
            expected_raw = week_data[-1]
        else:
            for i in range(len(starting_sizes)-1):
                if starting_sizes[i] <= initial_size <= starting_sizes[i+1]:
                    low_s = starting_sizes[i]
                    high_s = starting_sizes[i+1]
                    low_v = week_data[i]
                    high_v = week_data[i+1]
                    frac = (initial_size - low_s) / (high_s - low_s)
                    expected_raw = low_v + frac * (high_v - low_v)
                    break
        
        # Apply HR scaling (INVERTED for cure phase)
        if week <= 2:
            # Growth phase: higher HR = larger expected (faster growth)
            hr_factor = HR[name] / HR['MLH1']
        else:
            # Cure phase: higher HR = smaller expected (faster shrinkage)
            hr_factor = HR['MLH1'] / HR[name]
        
        expected = expected_raw * hr_factor
        expected = max(1.1, expected)
        
        # Calculate likelihood (normal PDF)
        diff = current_size - expected
        z_score = diff / sigma
        likelihood = np.exp(-0.5 * z_score * z_score) / (sigma * np.sqrt(2 * np.pi))
        
        # Apply prior
        unnorm[name] = likelihood * genotype_prior[name]
    
    total = sum(unnorm.values())
    if total == 0:
        return {name: 1.0/len(names) for name in names}
    
    # Normalize to get posterior probabilities
    posterior = {name: unnorm[name]/total for name in names}
    
    return posterior
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
    
    - **First hit (inherited mutation):** Germline Mutation Inherited from a parent. Present at birth (age 0) with **one faulty copy** of an MMR gene inherited from a parent.
    - **Second hit (acquired mutation):** Somatic inactivation of the remaining normal allele, second healthy copy is damaged or lost, leading to MSI and tumor early phase evolution.
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
    - $L_c$ =  size, $K_c \approx 1.1$ mm (cure floor)
    
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
    
    # ============================================================
    # ALERT: Tumor not responding to immunotherapy
    # ============================================================
    if current_size > initial_size:
        st.error(
            "🚨 **ALERT: TUMOR NOT RESPONDING TO IMMUNOTHERAPY** 🚨\n\n"
            f"Current size ({current_size:.1f} mm) is LARGER than Initial size ({initial_size:.1f} mm).\n\n"
            "Consider: Prompt Clinical Review, Alternate Treatment."
        )
    elif current_size > initial_size * 0.9:
        st.warning(
            f"⚠️ **Minimal Response:** Current size ({current_size:.1f} mm) is >90% of Initial size ({initial_size:.1f} mm). Monitor closely."
        )
    # ============================================================
    
    # Show estimated growth time
    # Show estimated growth time
    weeks_to_grow, lower_grow, upper_grow = get_growth_time(initial_size, 'MLH1')
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
        
        # ============================================================
        # GENOTYPE CLUSTERING (adds below the histogram)
        # ============================================================
        
        # Sort probabilities
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        # Clustering threshold (5% - genotypes within 5% probability are grouped)
        threshold = 0.05
        
        # Build clusters
        clusters = []
        i = 0
        while i < len(sorted_probs):
            cluster_names = [sorted_probs[i][0]]
            cluster_total = sorted_probs[i][1]
            j = i + 1
            while j < len(sorted_probs) and sorted_probs[j][1] >= sorted_probs[i][1] - threshold:
                cluster_names.append(sorted_probs[j][0])
                cluster_total += sorted_probs[j][1]
                j += 1
            clusters.append((" + ".join(cluster_names), cluster_total))
            i = j
        
        # Display clusters as bullet points
        st.markdown("---")
        st.markdown("### 🧬 Genotype Clusters")
        st.markdown("*Genotypes within 5% probability are grouped as indistinguishable*")
        
        for idx, (names, total) in enumerate(clusters, 1):
            if idx == 1:
                st.markdown(f"**Most likely cluster ({total:.1%})**: {names}")
            elif idx == 2:
                st.markdown(f"**Second cluster ({total:.1%})**: {names}")
            elif idx == 3:
                st.markdown(f"**Third cluster ({total:.1%})**: {names}")
            else:
                st.markdown(f"**Cluster {idx} ({total:.1%})**: {names}")
        
        st.caption("⚠️ Research & education only - not medical advice")
