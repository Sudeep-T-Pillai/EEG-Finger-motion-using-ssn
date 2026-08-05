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
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_model_checkpoint.pth"

TIME_DATA_PATH = "/home/labadmin/Desktop/SpikeArm/Finger_cmu/SendAnywhere_LL5TEPFJ/Troi_epochs_stride_0.126s_time.npy"
EVENTS_PATH    = "/home/labadmin/Desktop/SpikeArm/Finger_cmu/SendAnywhere_LL5TEPFJ/Troi_epochs_stride_0.126s_events.npy"

# ==========================================
# MODEL DEFINITION (MUST MATCH TRAINING)
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
        self.lif_in = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad,
            init_hidden=True,
            learn_beta=True,
            learn_threshold=True
        )

        self.layers = nn.ModuleList([
            SpikingTransformerBlock(self.embed_dim, num_heads=4)
            for _ in range(self.depth)
        ])

        self.fc_out = nn.Linear(self.embed_dim, NUM_CLASSES)
        self.lif_out = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
            learn_beta=True,
            learn_threshold=True
        )

    def forward(self, x):
        pos = torch.relu(x)
        neg = torch.relu(-x)
        x = torch.cat([pos, neg], dim=2)

        x = self.spatial_encoder(x.permute(0, 2, 1))
        x = self.bn_in(x).permute(0, 2, 1)

        self.lif_in.init_leaky()
        spikes = [self.lif_in(x[:, t, :]) for t in range(x.size(1))]
        x = torch.stack(spikes, dim=1)

        for layer in self.layers:
            x = layer(x)

        self.lif_out.init_leaky()
        out = []
        for t in range(x.size(1)):
            spk, _ = self.lif_out(self.fc_out(x[:, t, :]))
            out.append(spk)

        return torch.stack(out, dim=0)

# ==========================================
# LOAD DATA (SAME PREPROCESSING)
# ==========================================
time_data = np.load(TIME_DATA_PATH)
events = np.load(EVENTS_PATH)
labels = events[:, 2]

X = time_data.transpose(0, 2, 1)
y = LabelEncoder().fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
X_test  = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
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
all_preds = []
all_targets = []
total_time = 0
num_samples = 0

with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        start = time.time()
        spk_rec = model(data)
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        end = time.time()

        total_time += (end - start)
        num_samples += data.size(0)

        # Rate coding: average spikes over time
        rate = spk_rec.mean(dim=0)      # [Batch, Classes]
        preds = rate.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

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
plt.title("Confusion Matrix - SNN Transformer")
plt.tight_layout()
plt.show()


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
plt.title("Normalized Confusion Matrix - SNN Transformer")
plt.tight_layout()
plt.show()

