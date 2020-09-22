# Configuration parameters for audio feature model

# Training
EPOCHS = 500
CNN_BOOLEAN = True

# .pkl files
VARIABLES_FOLDER = "pkl/"
# Sampling settings
SAMPLING_RATE = 8000
WINDOW_LENGTH = round(50 * 1e-3 * SAMPLING_RATE)  # 50 msec
HOP_LENGTH = round(50 * 1e-3 * SAMPLING_RATE)  # 50 msec -- no overlapping

# Dataloader
BATCH_SIZE = 16
OVERSAMPLING = False
FEATURE_EXTRACTION_METHOD = "MEL_SPECTROGRAM"
