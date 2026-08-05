import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate

import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# 1. DEFINE THE MODEL 
# ==========================================
# CONFIG
# ==========================================
batch_size = 64
NUM_CLASSES = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

MODEL_PATH = "450_epoch_SNN_transformer.pth"
LAMDA_SMOOTH = 0.1
TIME_DATA_PATH = "roi_epochs_stride_0.126s_time.npy"
EVENTS_PATH    = "Troi_epochs_stride_0.126s_events.npy"

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


class RobustRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.weights = torch.tensor([1.0,1.0,2.0,3.0]).to(device) 

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        weighted_loss =  loss * self.weights
        diff = pred[1:]-pred[:-1]
        smoothness = torch.mean(diff ** 2)
        return weighted_loss.mean() + (0.1 * smoothness)

# ==========================================
# LOAD DATA (SAME PREPROCESSING)
# ==========================================
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


# 2. LOAD PRE-TRAINED WEIGHTS
model = Deep_ST_PLIF().to(device)
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
) # Load your 75% acc model
model.load_state_dict(checkpoint)
print("Loaded Pre-Trained Classification Weights!")

# 3. MODIFY THE HEAD FOR REGRESSION
# We re-initialize the final layer because 'Classes' are different from 'Angles'
# But we keep the Feature Extractor (Transformer) frozen initially!
model.fc_out = nn.Linear(model.embed_dim, 4).to(device)
nn.init.orthogonal_(model.fc_out.weight)
nn.init.constant_(model.fc_out.bias,0.0)
# Note: 4 Outputs = [Thumb Angle, Index Angle, Middle Angle, Pinky Angle]

# 4. FREEZE THE BACKBONE (Optional but recommended)
# This forces the model to use the "Concepts" it already learned
for param in model.parameters():
    param.requires_grad = False
# Unfreeze ONLY the last layer
for param in model.fc_out.parameters():
    param.requires_grad = True
for param in model.lif_out.parameters():
    param.requires_grad = True

# 5. DEFINE REGRESSION LOSS
# We count spikes.
# 0 spikes = 0 degrees (Open)
# 100 spikes = 90 degrees (Closed)
# optimizer = torch.optim.Adam([
#     {'params': model.spatial_encoder.parameters(), 'lr': 1e-5},
#     {'params': model.layers.parameters(), 'lr': 1e-5},
#     {'params': model.lif_in.parameters(), 'lr': 1e-5},
#     {'params': model.fc_out.parameters(), 'lr': 1e-3},
#     {'params': model.lif_out.parameters(), 'lr': 1e-3}
# ])
optimizer =  torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-3)
loss_fn = RobustRegressionLoss()
print("------Starting Regression Fine tunining-------")
for epoch in range(100):
    model.train()
    batch_loss = 0
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        # One-hot regression targets
        regression_targets = torch.zeros(targets.size(0), 4, device=device)
        regression_targets.scatter_(1, targets.unsqueeze(1), 1.0)

        spk_rec = model(data)            # [Time, Batch, 4]
        rate = spk_rec.mean(dim=0)       # [Batch, 4]

        loss = loss_fn(rate, regression_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()

    epoch_loss = batch_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "450Test_unfreezed_differential_snn_regression_final.pth")