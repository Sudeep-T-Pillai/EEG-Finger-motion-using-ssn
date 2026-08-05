import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. HYPERPARAMETERS
# ==========================================
batch_size = 64
num_epochs = 450 # Increased epochs (SNNs learn slower)
learning_rate = 2e-3 # Slightly higher LR for SNNs
beta = 0.95 # Decay rate (High beta = "Memory" of previous time steps)
time_steps = 100 #  window size
num_inputs = 32 #  32 EEG channels
num_hidden = 128 # Hidden layer size
num_outputs = 4 # 4 Classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

TIME_DATA_PATH = "Troi_epochs_stride_0.126s_time.npy"
EVENTS_PATH = "Troi_epochs_stride_0.126s_events.npy"


# ==========================================
# 2. DATA LOADING
# ==========================================
# Seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
print(np.load(EVENTS_PATH).shape)

# Reshape data toprint("--- Loading and Preparing Data ---")
time_data = np.load(TIME_DATA_PATH)
events = np.load(EVENTS_PATH)
labels = events[:, 2] #(N, T, C) - Samples, Timesteps, Channels
X = time_data.transpose(0, 2, 1)
y = labels

thumb_vs_other_indices = np.where((y == 1) | (y == 4))[0]
X_binary = X[thumb_vs_other_indices]
y_binary = y[thumb_vs_other_indices]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = [str(c) for c in label_encoder.classes_]

# Split data into training and validation sets first
X_train, X_val, y_train, y_val = train_test_split(
 X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardize the data
# The scaler needs 2D data, so we reshape: (N, T, C) -> (N*T, C)
n_samples_train, n_timesteps, n_features = X_train.shape
n_samples_val = X_val.shape[0]

X_train_reshaped = X_train.reshape(-1, n_features)
X_val_reshaped = X_val.reshape(-1, n_features)

# Fit the scaler ONLY on the training data to prevent data leakage
scaler = StandardScaler()
scaler.fit(X_train_reshaped)

# Apply the scaler to both train and validation data
X_train_scaled = scaler.transform(X_train_reshaped)
X_val_scaled = scaler.transform(X_val_reshaped)

# Reshape the data back to the (N, T, C) format
X_train = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
X_val = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)

print("Data has been standardized.")

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create DataLoaders

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# time_data = np.load(TIME_DATA_PATH)
# events = np.load(EVENTS_PATH)
# labels = events[:, 2]
# X = time_data.transpose(0, 2, 1)

# print("time_data shape:", time_data.shape)
# print("events shape:", events.shape)
# print("X shape (N, T, C):", X.shape)
# print("Unique labels:", np.unique(labels))


# ==========================================
# 3. THE "DIRECT INPUT" SNN ARCHITECTURE
# ==========================================
# class DirectSNN(nn.Module):
# def __init__(self):
# super().__init__()

# # Surrogate Gradient (The "Trick" to train spikes)
# spike_grad = surrogate.fast_sigmoid(slope=25)

# # Layer 1: The Encoder (Analog Input -> Spikes)
# # We inject current directly. The neuron integrates the voltage.
# self.fc1 = nn.Linear(num_inputs, num_hidden)
# self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

# # Layer 2: Deep Spiking Layer (Spikes -> Spikes)
# self.fc2 = nn.Linear(num_hidden, num_hidden)
# self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

# # Layer 3: Output (Spikes -> Classification)
# self.fc3 = nn.Linear(num_hidden, num_outputs)
# self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad,
# init_hidden=True, output=True)

# def forward(self, x):
# # Initialize hidden states
# self.lif1.init_leaky()
# self.lif2.init_leaky()
# self.lif3.init_leaky()

# # Record output spikes
# spk3_rec = []

# # CRITICAL CHANGE: Permute to (Time, Batch, Channels)
# # x is originally (Batch, Time, Channels)
# x = x.permute(1, 0, 2)

# # Loop over time
# for step in range(x.size(0)):
# current_input = x[step] # Shape: (Batch, 32)

# # Layer 1: Encode
# cur1 = self.fc1(current_input)
# spk1 = self.lif1(cur1)

# # Layer 2: Process
# cur2 = self.fc2(spk1)
# spk2 = self.lif2(cur2)

# # Layer 3: Decode
# cur3 = self.fc3(spk2)
# spk3, _ = self.lif3(cur3)

# spk3_rec.append(spk3)

# # Stack -> (Time, Batch, Output)
# return torch.stack(spk3_rec, dim=0)


# ==========================================
# RECURRENT SNN ARCHITECTURE (The "LSTM-SNN")
# ==========================================
# 

# class ConvSNN(nn.Module):
# def __init__(self):
# super().__init__()

# # Use a gentler slope (10 instead of 25) to help gradients flow 
# # even if spikes are rare initially.
# spike_grad = surrogate.fast_sigmoid(slope=10) 
 
# # Layer 1: Convolution (Spatial Feature Extraction)
# # Input: 32 Channels -> Output: 64 Features
# self.conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
 
# # THIS IS YOUR FIX: Batch Norm "powers up" the signal automatically.
# self.bn1 = nn.BatchNorm1d(64) 
 
# self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
# learn_beta=True, learn_threshold=True)

# # Layer 2: Dense Layer
# self.fc2 = nn.Linear(64, 128)
# self.bn2 = nn.BatchNorm1d(128) # Powers up the second layer too
# self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
# learn_beta=True, learn_threshold=True)

# # Layer 3: Output
# self.fc3 = nn.Linear(128, num_outputs)
# # Note: We don't usually Batch Norm the output layer of a classifier
# self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, 
# output=True, learn_beta=True, learn_threshold=True)

# def forward(self, x):
# # Reset memory
# self.lif1.init_leaky()
# self.lif2.init_leaky()
# self.lif3.init_leaky()

# spk3_rec = []

# # Data format: [Batch, Time, Channels]
# # Permute for time loop: [Time, Batch, Channels]
# x = x.permute(1, 0, 2) 

# for step in range(x.size(0)):
# current_input = x[step] # [Batch, 32]
 
# # Conv1d expects [Batch, Channels, Length]
# # We treat the instantaneous data as a signal of length 1
# conv_in = current_input.unsqueeze(2) # [Batch, 32, 1]
 
# # Layer 1: Conv -> BatchNorm -> Spike
# cur1 = self.conv1(conv_in)
# cur1 = self.bn1(cur1).squeeze(2) # Normalize and remove length dim
# spk1 = self.lif1(cur1) # Get spikes

# # Layer 2: Linear -> BatchNorm -> Spike
# cur2 = self.fc2(spk1)
# cur2 = self.bn2(cur2)
# spk2 = self.lif2(cur2)

# # Layer 3: Output
# cur3 = self.fc3(spk2)
# spk3, _ = self.lif3(cur3)

# spk3_rec.append(spk3)

# return torch.stack(spk3_rec, dim=0)


# class TemporalConvSNN(nn.Module):
# def __init__(self):
# super().__init__()
 
# # Hyperparameters
# spike_grad = surrogate.fast_sigmoid(slope=25) # Steeper slope = sharper decisions
 
# # ---------------------------------------------------------
# # BLOCK 1: TEMPORAL FEATURE EXTRACTION (Standard CNN)
# # ---------------------------------------------------------
# # We use a larger kernel (size 5 or 7) to see "time context"
# # Input: 32 Channels. Output: 64 Features.
# # Padding ensures the time dimension stays at 100.
# self.conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
# self.bn1 = nn.BatchNorm1d(64) # Powers up the signal
# self.act1 = nn.LeakyReLU(0.1) # Standard ANN activation for the feature extractor
 
# # ---------------------------------------------------------
# # BLOCK 2: SPIKING NEURAL NETWORK
# # ---------------------------------------------------------
# # Now we feed the "Features" into the SNN
 
# # Layer 2: Dense SNN Layer
# self.fc2 = nn.Linear(64, 128)
# self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
# learn_beta=True, learn_threshold=True)
 
# # Layer 3: Output SNN Layer
# self.fc3 = nn.Linear(128, num_outputs)
# self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, 
# output=True, learn_beta=True, learn_threshold=True)

# def forward(self, x):
# # Initialize SNN memories
# self.lif2.init_leaky()
# self.lif3.init_leaky()

# # x shape: [Batch, Time, Channels] -> [64, 100, 32]
 
# # 1. PROCESS TIME FIRST (Parallelized CNN)
# # Permute for Conv1d: [Batch, Channels, Time] -> [64, 32, 100]
# x_cnn = x.permute(0, 2, 1) 
 
# # The Conv1d now sees the whole 100-step sequence!
# features = self.conv1(x_cnn) # [64, 64, 100]
# features = self.bn1(features)
# features = self.act1(features)
 
# # 2. PREPARE FOR SNN LOOP
# # Permute back to Time-First: [Time, Batch, Features] -> [100, 64, 64]
# features = features.permute(2, 0, 1)
 
# spk3_rec = []

# # 3. RUN SNN LOOP (Now feeding it rich features, not raw data)
# for step in range(features.size(0)):
# current_input = features[step] # [Batch, 64]

# # Layer 2
# cur2 = self.fc2(current_input)
# spk2 = self.lif2(cur2)

# # Layer 3
# cur3 = self.fc3(spk2)
# spk3, _ = self.lif3(cur3)

# spk3_rec.append(spk3)

# return torch.stack(spk3_rec, dim=0)

# # ==========================================
# # OPTIMIZER TWEAK (Shock Therapy)
# # ==========================================
# model = TemporalConvSNN().to(device)
# # Increase LR to 5e-3 to force the "Lazy Learner" to move
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.999)) 
# loss_fn = SF.ce_rate_loss()
# history = {'loss': [], 'acc': []}


# ==========================================
# RECURRENT SNN 
# ==========================================
# 
# class RecurrentNormSNN(nn.Module):
# def __init__(self):
# super().__init__()

# # Hyperparameters
# spike_grad = surrogate.fast_sigmoid(slope=25) 
 
# # ---------------------------------------------------------
# # BLOCK 1: INPUT ENCODING (The "Signal Booster")
# # ---------------------------------------------------------
# self.encoder = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1)
# self.bn = nn.BatchNorm1d(128)
 
# # ---------------------------------------------------------
# # BLOCK 2: RECURRENT SNN (The "Memory")
# # ---------------------------------------------------------
# self.rlif = snn.RLeaky(
# beta=beta,
# spike_grad=spike_grad,
# init_hidden=True,
# linear_features=128,
# learn_beta=True,
# learn_threshold=True
# )

# # ---------------------------------------------------------
# # BLOCK 3: OUTPUT
# # ---------------------------------------------------------
# self.fc_out = nn.Linear(128, num_outputs)
# self.lif_out = snn.Leaky(
# beta=beta,
# spike_grad=spike_grad,
# init_hidden=True,
# output=True,
# learn_beta=True,
# learn_threshold=True
# )

# def forward(self, x):
# # Initialize Memories
# self.rlif.init_rleaky()
# self.lif_out.init_leaky()
 
# # x: [Batch, Time, Channels] -> [Batch, Channels, Time]
# x = x.permute(0, 2, 1)
 
# # Encode and normalize
# encoded_seq = self.encoder(x) # [Batch, 128, Time]
# encoded_seq = self.bn(encoded_seq)
 
# # Permute to [Time, Batch, Channels] for time-step iteration
# encoded_seq = encoded_seq.permute(2, 0, 1) # [Time, Batch, 128]
 
# spk_rec = []
 
# # Process each time step
# for step in range(encoded_seq.size(0)):
# current_input = encoded_seq[step] # [Batch, 128]
 
# # Recurrent layer - returns only spike output
# spk1 = self.rlif(current_input) # [Batch, 128]
 
# # Output layer
# cur_out = self.fc_out(spk1) # [Batch, 4]
# spk_out, _ = self.lif_out(cur_out) # Leaky returns (spk, mem)
 
# spk_rec.append(spk_out)
 
# # Stack along time dimension: [Time, Batch, Classes]
# result = torch.stack(spk_rec, dim=0) # [100, 64, 4]
 
# # Debug check
# if result.size(0) == 100 and result.size(1) != 64:
# print(f"ERROR: Expected [100, 64, 4], got {result.shape}")
# print(f"encoded_seq.size(0) = {encoded_seq.size(0)}")
# print(f"len(spk_rec) = {len(spk_rec)}")
# print(f"spk_rec[0].shape = {spk_rec[0].shape}")
 
# return result

# 
# class SpikingSelfAttention(nn.Module):
# def __init__(self, embed_dim, num_heads):
# super().__init__()
# self.embed_dim = embed_dim
# self.num_heads = num_heads
# self.head_dim = embed_dim // num_heads
# self.scale = self.head_dim ** -0.5

# # Linear projections for Query, Key, Value
# self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
# self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
# self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
# self.out_proj = nn.Linear(embed_dim, embed_dim)

# # The "Spiking Softmax" - Learnable Threshold
# # We replace the standard Softmax with a LIF neuron to keep it neuromorphic
# self.attn_lif = snn.Leaky(
# beta=0.9, 
# spike_grad=surrogate.fast_sigmoid(slope=5), 
# init_hidden=True,
# learn_beta=True, 
# learn_threshold=True
# )

# def forward(self, x):
# # x shape: [Batch, Time, Embed_Dim]
# B, T, C = x.shape
 
# # 1. Project Q, K, V
# q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
# k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
# v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

# # 2. Compute Raw Attention Scores
# attn = (q @ k.transpose(-2, -1)) * self.scale # [Batch, Heads, Time, Time]
 
# # 3. SPIKING ATTENTION MECHANISM
# # Flatten to feed into LIF: [Batch*Heads, Time*Time]
# attn_flat = attn.reshape(B * self.num_heads, T * T)
 
# self.attn_lif.init_leaky()
# # The neuron fires 1 if the attention score is high, 0 otherwise
# attn_spikes = self.attn_lif(attn_flat) 
 
# # Reshape back to attention map
# attn_spikes = attn_spikes.reshape(B, self.num_heads, T, T)

# # 4. Apply Spiking Mask to Values
# out = (attn_spikes @ v).transpose(1, 2).reshape(B, T, C)
# return self.out_proj(out)

# # ==========================================
# # 2. THE ARCHITECTURE: ST-PLIF
# # ==========================================
# class ST_PLIF(nn.Module):
# def __init__(self):
# super().__init__()
# spike_grad = surrogate.fast_sigmoid(slope=5)
 
# # BLOCK 1: SPLIT POLARITY ENCODER
# # Input 64 because we split 32 channels into (Pos + Neg)
# self.fc_in = nn.Linear(64, 128)
# self.lif_in = snn.Leaky(
# beta=0.9, spike_grad=spike_grad, init_hidden=True,
# learn_beta=True, learn_threshold=True
# )
# self.drop1 = nn.Dropout(0.4) # <--- Heavy regularization
 
# # BLOCK 2: SPIKING TRANSFORMER (The Novelty)
# # 128 dimensions, 4 attention heads
# self.attention = SpikingSelfAttention(embed_dim=128, num_heads=4)
# self.bn_attn = nn.BatchNorm1d(128) # Stabilize attention outputs
# self.drop2 = nn.Dropout(0.4)
 
# # BLOCK 3: OUTPUT DECODER
# self.fc_out = nn.Linear(128, 4) # 4 Classes
# self.lif_out = snn.Leaky(
# beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True,
# learn_beta=True, learn_threshold=True
# )

# def forward(self, x):
# # x: [Batch, Time, 32]
 
# # 1. SPLIT POLARITY (The "Retina" Trick)
# pos = torch.relu(x)
# neg = torch.relu(-x)
# x_split = torch.cat([pos, neg], dim=2) # [Batch, Time, 64]
 
# # 2. ENCODE
# self.lif_in.init_leaky()
# spk_rec = []
 
# # We process the linear encoder step-by-step
# # (Alternatively, we could use nn.Linear on the whole sequence if memory allows, 
# # but let's stick to the loop for safety)
# encoded_spikes = []
# for step in range(x_split.size(1)):
# cur = self.fc_in(x_split[:, step, :])
# spk = self.lif_in(cur)
# encoded_spikes.append(spk)
 
# # Stack: [Batch, Time, 128]
# seq_spikes = torch.stack(encoded_spikes, dim=1)
# seq_spikes = self.drop1(seq_spikes)
 
# # 3. ATTENTION (Processes the whole window at once!)
# # This is where the magic happens. It looks at the whole 1-sec sequence.
# attn_out = self.attention(seq_spikes)
 
# # Reshape for BatchNorm [Batch, Channels, Time]
# attn_out = attn_out.permute(0, 2, 1)
# attn_out = self.bn_attn(attn_out)
# attn_out = attn_out.permute(0, 2, 1) # Back to [Batch, Time, Channels]
# attn_out = self.drop2(attn_out)
 
# # 4. DECODE / CLASSIFY
# self.lif_out.init_leaky()
# out_spikes = []
 
# for step in range(attn_out.size(1)):
# cur = self.fc_out(attn_out[:, step, :])
# spk, _ = self.lif_out(cur)
# out_spikes.append(spk)
 
# return torch.stack(out_spikes, dim=0) # [Time, Batch, 4]

import torch
import torch.nn as nn
class SpikingTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Attention Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Spiking Neuron for Attention Map
        self.attn_lif = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid(slope=5),
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True
        )

        # Feed Forward Part (Reasoning step)
        self.ff_1 = nn.Linear(embed_dim, embed_dim * 2)
        self.ff_lif = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid(slope=5),
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True
        )
        self.ff_2 = nn.Linear(embed_dim * 2, embed_dim)

        # Normalization & Dropout
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # x: [Batch, Time, Embed_Dim]
        B, T, C = x.shape

        # --- SUB-BLOCK 1: Attention ---
        residual = x

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot Product
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Spiking Softmax
        attn_flat = attn.reshape(B * self.num_heads, T * T)
        self.attn_lif.init_leaky()
        attn_spikes = self.attn_lif(attn_flat)
        attn_spikes = attn_spikes.reshape(B, self.num_heads, T, T)

        # Apply & Project
        out = (attn_spikes @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        out = self.drop(out)

        # Add Residual + BatchNorm
        out = out.permute(0, 2, 1)
        out = self.bn1(out)
        out = out.permute(0, 2, 1)

        x = residual + out

        # --- SUB-BLOCK 2: Feed Forward ---
        residual = x

        out = self.ff_1(x)
        self.ff_lif.init_leaky()
        out = self.ff_lif(out)
        out = self.ff_2(out)
        out = self.drop(out)

        # Add Residual + BatchNorm
        out = out.permute(0, 2, 1)
        out = self.bn2(out)
        out = out.permute(0, 2, 1)

        return residual + out

# ==========================================
# 2. THE SCALED ARCHITECTURE: DEEP ST-PLIF
# ==========================================
class Deep_ST_PLIF(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=5)

        # CONFIGURATION
        self.embed_dim = 256  # Increased from 128
        self.depth = 2        # Increased from 1 (Stacking layers)

        # BLOCK 1: SPLIT POLARITY SPATIAL ENCODER
        # Using Conv1d to mix the 64 split-channels spatially first
        self.spatial_encoder = nn.Conv1d(64, self.embed_dim, kernel_size=1)
        self.bn_in = nn.BatchNorm1d(self.embed_dim)
        self.lif_in = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad,
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True
        )
        self.drop_in = nn.Dropout(0.3)

        # BLOCK 2: STACKED TRANSFORMER BLOCKS
        self.layers = nn.ModuleList([
            SpikingTransformerBlock(self.embed_dim, num_heads=4)
            for _ in range(self.depth)
        ])

        # BLOCK 3: OUTPUT DECODER
        self.fc_out = nn.Linear(self.embed_dim, 4)
        self.lif_out = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
            learn_beta=True,
            learn_threshold=True
        )

    def forward(self, x):
        # x: [Batch, Time, 32]

        # 1. SPLIT POLARITY
        pos = torch.relu(x)
        neg = torch.relu(-x)
        x_split = torch.cat([pos, neg], dim=2)  # [Batch, Time, 64]

        # 2. SPATIAL ENCODE
        # Permute for Conv1d: [Batch, 64, Time]
        x_spatial = x_split.permute(0, 2, 1)
        x_spatial = self.spatial_encoder(x_spatial)  # [Batch, 256, Time]
        x_spatial = self.bn_in(x_spatial)

        # Permute back: [Batch, Time, 256]
        x_spatial = x_spatial.permute(0, 2, 1)

        # Initial Spike Generation
        self.lif_in.init_leaky()

        # Loop over time for stateful LIF
        encoded_list = []
        for step in range(x_spatial.size(1)):
            spk = self.lif_in(x_spatial[:, step, :])
            encoded_list.append(spk)

        x_seq = torch.stack(encoded_list, dim=1)  # [Batch, Time, 256]
        x_seq = self.drop_in(x_seq)

        # 3. DEEP TRANSFORMER LOOP
        for layer in self.layers:
            x_seq = layer(x_seq)

        # 4. DECODE / CLASSIFY
        self.lif_out.init_leaky()
        out_spikes = []

        for step in range(x_seq.size(1)):
            cur = self.fc_out(x_seq[:, step, :])
            spk, _ = self.lif_out(cur)
            out_spikes.append(spk)

        return torch.stack(out_spikes, dim=0)  # [Time, Batch, 4]
# ==========================================
# OPTIMIZER
# ==========================================
model = Deep_ST_PLIF().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
history = {'loss': [], 'acc': []}
loss_fn = SF.ce_rate_loss()

print("--- Starting Training ---")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    train_correct = 0
    train_total = 0

    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # --- MIXUP IMPLEMENTATION ---
        if torch.rand(1).item() < 0.5:  # Apply MixUp 50% of the time
            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(data.size(0)).to(device)

            mixed_data = lam * data + (1 - lam) * data[index]

            # Forward pass with mixed data
            spk_rec = model(mixed_data)  # [Time, Batch, 4]

            # MixUp loss
            loss = (
                lam * loss_fn(spk_rec, targets)
                + (1 - lam) * loss_fn(spk_rec, targets[index])
            )
        else:
            # Normal forward pass
            spk_rec = model(data)  # [Time, Batch, 4]
            loss = loss_fn(spk_rec, targets)

        # --- DEBUG CHECK (first batch of first epoch only) ---
        if epoch == 0 and i == 0:
            print(f"DEBUG: spk_rec shape: {spk_rec.shape}")
            print(f"DEBUG: targets shape: {targets.shape}")

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Training accuracy
        acc = SF.accuracy_rate(spk_rec, targets)
        train_correct += acc * targets.size(0)
        train_total += targets.size(0)

    scheduler.step()

    # ---------- Validation ----------
    model.eval()
    val_total = 0
    val_correct = 0

    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)

            spk_rec = model(data)
            acc = SF.accuracy_rate(spk_rec, targets)

            val_correct += acc * targets.size(0)
            val_total += targets.size(0)

    val_acc = val_correct / val_total
    train_acc = train_correct / train_total
    best_val_acc = 0.0

    # Inside the loop, after validation
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_checkpoint.pth")
        print(f"--> New Best Model Saved! (Acc: {best_val_acc*100:.2f}%)")

    # Also save "latest" every 50 epochs just in case
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        history["loss"].append(epoch_loss)
        history["acc"].append(val_acc)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch+1} | "
            f"Loss: {epoch_loss:.2f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Train Acc: {train_acc*100:.2f}%"
        )

PATH = "450_epoch_SNN_transformer.pth"
torch.save(model.state_dict(),PATH)
# ==========================================
# 5. FINAL PLOT
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(history['acc'], label="Validation Accuracy")
plt.title("SNN Training Progress (Direct Input)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
