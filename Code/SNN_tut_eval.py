import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

# ==========================================
# CONFIG
# ==========================================
BATCH_SIZE = 64
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "450_epoch_SNN_transformer_2class_thumb_pinky_.pth"

TIME_DATA_PATH = "Troi_epochs_stride_0.126s_time.npy"
EVENTS_PATH    = "Troi_epochs_stride_0.126s_events.npy"


def calculate_theoretical_energy(spike_tensor, mac_operations_ann, time_steps=100):
    """
    Calculates theoretical energy consumption based on 45nm CMOS hardware.
    spike_tensor: The binary output tensor from your PLIF neurons (Shape: [Batch, Channels, Time])
    mac_operations_ann: The number of baseline MACs your Conv1D layer uses before spiking
    """
    # 45nm Hardware Constants (in picoJoules)
    ENERGY_MAC = 3.2  # 32-bit Floating Point Multiply-Accumulate
    ENERGY_AC = 0.1   # 32-bit Floating Point Accumulate (Addition)
    
    # 1. Calculate Sparsity (Firing Rate)
    # Total possible spikes = batch_size * neurons * timesteps
    total_possible_spikes = spike_tensor.numel() 
    actual_spikes = torch.sum(spike_tensor).item()
    
    firing_rate = actual_spikes / total_possible_spikes
    print(f"Network Firing Rate (Sparsity): {firing_rate * 100:.2f}%")
    
    # 2. Calculate SNN Energy
    # Only active spikes trigger an AC operation. Non-spikes (0) do nothing.
    snn_ac_operations = actual_spikes 
    snn_energy_pj = (mac_operations_ann * ENERGY_MAC) + (snn_ac_operations * ENERGY_AC)
    
    # 3. Calculate equivalent continuous ANN/LSTM energy (Worst Case)
    # An ANN performs a MAC for every single neuron at every single timestep
    ann_mac_operations = mac_operations_ann + total_possible_spikes
    ann_energy_pj = ann_mac_operations * ENERGY_MAC
    
    # 4. Results
    efficiency_ratio = ann_energy_pj / snn_energy_pj
    print(f"Theoretical ANN Energy: {ann_energy_pj:.2f} pJ")
    print(f"Theoretical SNN Energy: {snn_energy_pj:.2f} pJ")
    print(f"The SNN is {efficiency_ratio:.2f}x more energy-efficient than the continuous baseline.")
    
    return snn_energy_pj, efficiency_ratio
# ==========================================
# MODEL DEFINITION 
# ==========================================
class SpikingTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_lif = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid(slope=5),
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True
        )

        self.ff_1 = nn.Linear(embed_dim, embed_dim * 2)
        self.ff_lif = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid(slope=5),
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True
        )
        self.ff_2 = nn.Linear(embed_dim * 2, embed_dim)

        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        B, T, C = x.shape
        residual = x

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_flat = attn.reshape(B * self.num_heads, T * T)

        self.attn_lif.init_leaky()
        attn_spikes = self.attn_lif(attn_flat)
        attn_spikes = attn_spikes.reshape(B, self.num_heads, T, T)

        out = (attn_spikes @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        out = self.drop(out)

        out = self.bn1(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = residual + out

        residual = x
        out = self.ff_1(x)
        self.ff_lif.init_leaky()
        out = self.ff_lif(out)
        out = self.ff_2(out)
        out = self.drop(out)

        out = self.bn2(out.permute(0, 2, 1)).permute(0, 2, 1)
        return residual + out


class Deep_ST_PLIF(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=5)
        self.embed_dim = 256
        self.depth = 2

        self.spatial_encoder = nn.Conv1d(64, self.embed_dim, kernel_size=1)
        self.bn_in = nn.BatchNorm1d(self.embed_dim)
        self.lif_in = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, learn_beta=True, learn_threshold=True)

        self.layers = nn.ModuleList([
            SpikingTransformerBlock(self.embed_dim, num_heads=4)
            for _ in range(self.depth)
        ])

        self.fc_out = nn.Linear(self.embed_dim, NUM_CLASSES)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        all_spikes = [] # To track spikes for energy calculation
        
        pos = torch.relu(x)
        neg = torch.relu(-x)
        x = torch.cat([pos, neg], dim=2)

        x = self.spatial_encoder(x.permute(0, 2, 1))
        x = self.bn_in(x).permute(0, 2, 1)

        self.lif_in.init_leaky()
        spikes_in = []
        for t in range(x.size(1)):
            spk = self.lif_in(x[:, t, :])
            spikes_in.append(spk)
        
        x = torch.stack(spikes_in, dim=1)
        all_spikes.append(x) 

        for layer in self.layers:
            # Note: For strict energy tracking, you'd modify the Block to return its spikes too
            x = layer(x)
            all_spikes.append(x)

        self.lif_out.init_leaky()
        out_spikes = []
        for t in range(x.size(1)):
            spk, _ = self.lif_out(self.fc_out(x[:, t, :]))
            out_spikes.append(spk)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack, all_spikes

# ==========================================
# LOAD DATA (SAME PREPROCESSING)
# ==========================================
time_data = np.load(TIME_DATA_PATH)
events = np.load(EVENTS_PATH)
labels = events[:, 2]

X = time_data.transpose(0, 2, 1)
y = labels

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = [str(c) for c in label_encoder.classes_]

#========2 class filtering ========
thumb_vs_other_indices = np.where((y == 1) | (y == 4))[0]
X = X[thumb_vs_other_indices]
y = y[thumb_vs_other_indices]
y = LabelEncoder().fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
#========2 class balancing (thumb vs rest)========

# y = y.copy()
# y[y > 1] = 0  # Thumb = 1, Rest = 0

# thumb_indices = np.where(y == 1)[0]
# rest_indices  = np.where(y == 0)[0]

# n_thumb = len(thumb_indices)   # ✅ DEFINE FIRST

# print(f"Thumb trials: {len(thumb_indices)}")
# print(f"Rest trials:  {len(rest_indices)}")

# np.random.seed(42)  # reproducibility
# rest_indices_truncated = np.random.choice(
#     rest_indices,
#     size=n_thumb,
#     replace=False
# )

# balanced_indices = np.concatenate([thumb_indices, rest_indices_truncated])
# np.random.shuffle(balanced_indices)

# X_balanced = X[balanced_indices]
# y_balanced = y[balanced_indices]

# print("Balanced class counts:", np.bincount(y_balanced))

# #Split data into training and validation sets first
# X_train, X_val, y_train, y_val = train_test_split(
#     X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
# )
# unique, counts = np.unique(y_balanced, return_counts=True)
# print("Class distribution (original labels):")
# for cls, cnt in zip(unique, counts):
#      print(f"  Class {cls}: {cnt} samples")

#===========3 Class Filtering================
# three_class = np.where((y == 1) | (y == 2) | (y == 3) | (y == 4))[0]
# X = X[three_class]
# y = y[three_class]

# unique, counts = np.unique(y, return_counts=True)

# print("Class distribution (original labels):")
# for cls, cnt in zip(unique, counts):
#     print(f"  Class {cls}: {cnt} samples")

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# class_names = [str(c) for c in label_encoder.classes_]

# # Split data into training and validation sets first
# X_train, X_val, y_train, y_val = train_test_split(
#  X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )

# X_train, X_val, y_train, y_val = train_test_split(
#         X,
#         y_encoded,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
X_val_scaled  = scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)

# Convert the SCALED data to tensors, not the raw X_val
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
test_loader = DataLoader(
    TensorDataset(X_val_tensor, y_val_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False
)
# ==========================================
# LOAD MODEL
# ==========================================
model = Deep_ST_PLIF().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==========================================
# INFERENCE + METRICS
# ==========================================
# ==========================================
# INFERENCE + METRICS + ENERGY
# ==========================================
all_preds = []
all_targets = []
total_time = 0
num_samples = 0
total_spike_activity = []

# Estimate baseline MACs (roughly: Input_dim * Output_dim * Timesteps)
# For your Conv1D(64, 256) and transformer projections
ESTIMATED_MAC_BASE = (64 * 256) + (256 * 256 * 3 * 2) 

model.eval()
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        start = time.time()
        # Updated forward returns spikes
        spk_rec, layer_spikes = model(data) 
        
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        end = time.time()

        total_time += (end - start)
        num_samples += data.size(0)

        # Rate coding: average spikes over time
        rate = spk_rec.mean(dim=0)
        preds = rate.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
        # Flatten all spikes in this batch to pass to your energy function
        batch_spikes = torch.cat([s.flatten() for s in layer_spikes])
        total_spike_activity.append(batch_spikes)

# ==========================================
# FINAL ENERGY ANALYSIS
# ==========================================
print("\n===== ENERGY ANALYSIS =====")
all_spikes_tensor = torch.cat(total_spike_activity)
calculate_theoretical_energy(all_spikes_tensor, ESTIMATED_MAC_BASE)

# ==========================================
# RESULTS
# ==========================================
y_true = np.concatenate(all_targets)
y_pred = np.concatenate(all_preds)

# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
    yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - SNN Transformer  thumbs vs index vs middle vs pinky")
plt.tight_layout()
#plt.show()
plt.savefig("test_confusion_matrixSNN Transformer  Thumb va Index vs Middle vs Pinky.png")


acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro")
rec = recall_score(y_true, y_pred, average="macro")

print("\n===== TEST RESULTS =====")
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {prec*100:.2f}%")
print(f"Recall    : {rec*100:.2f}%")
print(f"Avg inference time per sample: {total_time/num_samples*1000:.3f} ms")

print("\nPer-class report:")
print(classification_report(y_true, y_pred))


# Normalized Confusion Matrix
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
    yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix - SNN Transformer  Thumb vs Index vs Middle vs Pinky")   
plt.tight_layout()
#plt.show()
plt.savefig("test_normalized_confusion_matrix SNN Transformer  Thumb vs Index vs Middle vs Pinky.png") 



