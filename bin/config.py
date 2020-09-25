# Configuration parameters for audio feature model

# Training
EPOCHS = 500
CNN_BOOLEAN = True

# .pkl files
VARIABLES_FOLDER = "pkl/"
# Sampling settings
WINDOW_LENGTH = round(50 * 1e-3)  # 50 msec
HOP_LENGTH = round(10 * 1e-3 )    # 10 msec step

# Dataloader
BATCH_SIZE = 16
OVERSAMPLING = False
FEATURE_EXTRACTION_METHOD = "MEL_SPECTROGRAM"
