# Configuration parameters for audio feature model

# Training
EPOCHS = 500
CNN_BOOLEAN = True

# .pkl files
VARIABLES_FOLDER = "pkl/"
# Sampling settings
WINDOW_LENGTH = (50 * 1e-3)
HOP_LENGTH = (50 * 1e-3)

# Dataloader
BATCH_SIZE = 16
OVERSAMPLING = False
FEATURE_EXTRACTION_METHOD = "MEL_SPECTROGRAM"
FUSED_SPECT = False
ZERO_PAD = False
FORCE_SIZE = False
SPECTOGRAM_SIZE = (51, 128)