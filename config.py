# Configuration parameters for audio feature model

# Sampling settings
SAMPLING_RATE = 16000
WINDOW_LENGTH = round(50 * 1e-3 * SAMPLING_RATE)  # 50 msec
HOP_LENGTH = round(50 * 1e-3 * SAMPLING_RATE)  # 50 msec -- no overlapping

# Dataloader
BATCH_SIZE = 16
