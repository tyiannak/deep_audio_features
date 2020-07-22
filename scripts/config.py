# Configuration parameters for audio feature model

# Sampling settings
SAMPLING_RATE = 16000
WIN_SIZE = round(50 * 1e-3 * SAMPLING_RATE)  # 50 msec
HOP_LENGTH = round(50 * 1e-3 * SAMPLING_RATE)  # 50 msec -- no overlapping
