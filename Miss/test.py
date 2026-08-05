import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA
print("--- LOADING DATA FOR LABEL AUDIT ---")
EVENTS_PATH = "Troi_epochs_stride_0.126s_events.npy"
TIME_DATA_PATH = "Troi_epochs_stride_0.126s_time.npy"

events = np.load(EVENTS_PATH) # [N, 3]
time_data = np.load(TIME_DATA_PATH) # [N, C, T]

# Check Raw Labels
raw_labels = events[:, 2]
unique_labels = np.unique(raw_labels)
print(f"Found Raw Labels in dataset: {unique_labels}")

# 2. DEFINE A TINY AUDIT MODEL (Fast Linear Classifier)
# We don't need the full Transformer to check labels. Linear is enough to see separability.
class TinyAudit(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.flat_dim = input_dim
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# 3. RUN PAIRWISE COMPARISONS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pairwise_check(label_A, label_B):
    # Filter Data
    indices = np.where((raw_labels == label_A) | (raw_labels == label_B))[0]
    X_sub = time_data[indices]
    y_sub = raw_labels[indices]
    
    # Remap labels to 0 and 1
    y_binary = (y_sub == label_B).astype(int) 
    
    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_binary, test_size=0.2, random_state=42)
    
    # Convert to Tensor
    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    test_ds = TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # Train Tiny Model
    model = TinyAudit(input_dim=X_tr.shape[1]*X_tr.shape[2]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(5): # Quick 5 epochs
        for x, y in train_loader:
            optim.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            optim.step()
            
    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in DataLoader(test_ds, batch_size=32):
            preds = model(x.to(device)).argmax(dim=1)
            correct += (preds == y.to(device)).sum().item()
            total += y.size(0)
            
    return correct / total

print("\n--- PAIRWISE SEPARABILITY REPORT ---")
import itertools
pairs = list(itertools.combinations(unique_labels, 2))

for (a, b) in pairs:
    acc = run_pairwise_check(a, b)
    print(f"Label {a} vs Label {b} Accuracy: {acc*100:.1f}%")