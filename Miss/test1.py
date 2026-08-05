import numpy as np
TIME_DATA_PATH = "Troi_epochs_stride_0.126s_time.npy"
time_data = np.load(TIME_DATA_PATH)
print("Time data shape:", time_data.shape)

X = time_data.transpose(0, 2, 1)
print("Transposed time data shape (X):", X.shape)
EVENTS_PATH    = "Troi_epochs_stride_0.126s_events.npy"
events = np.load(EVENTS_PATH)
print("Events data shape:", events.shape)
np.unique(events, axis=0)