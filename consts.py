FEATURES = [
    "pitch_mean_log",
    "pitch_range_log",
    "intensity_mean_vcd",
    #"jitter",
    #"shimmer",
    "nhr_vcd",
    "rate",
]

FEATURES_NORM = [f"{x}_norm" for x in FEATURES]
FEATURES_DATASET_NORM = [f"{x}_dataset_norm" for x in FEATURES]