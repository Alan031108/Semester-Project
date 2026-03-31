import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 資料路徑
DATA_PATH = r"C:\Users\方安政\Desktop\程式作業\semester_project\Dataset\DynamicData"
VICTIMS_PATH = os.path.join(DATA_PATH, "Victims")
ATTACKERS_PATH = os.path.join(DATA_PATH, "Attackers")
LOG_DIR = "./logs"

# 特徵欄位
COLUMN_NAMES = [
    "Start-X", "Start-Y", "End-X", "End-Y", "Distance-T", "Distance-X", "Distance-Y", "Path-T",
    "Elapsed-Time", "TouchID", "Tangential", "Curvature-T", "Velocity", "Acceleration",
    "Touch-Size", "Touch-Pressure", "Orientation-X", "Orientation-Y", "Orientation-XY",
    "Velocity-X", "Velocity-Y", "Velocity-Z", "Velocity-XY", "Acceleration-X", "Acceleration-Y",
    "Acceleration-Z", "Acceleration-XY", "Orientation-X Avg", "Orientation-Y Avg",
    "Orientation-XY Avg", "Velocity-X Avg", "Velocity-Y Avg", "Velocity-Z Avg", "Velocity-XY Avg",
    "Acceleration-X Avg", "Acceleration-Y Avg", "Acceleration-Z Avg", "Acceleration-XY Avg",
    "Orientation-X StDev", "Orientation-Y StDev", "Orientation-XY StDev", "Velocity-X StDev",
    "Velocity-Y StDev", "Velocity-Z StDev", "Velocity-XY StDev", "Acceleration-X StDev",
    "Acceleration-Y StDev", "Acceleration-Z StDev", "Acceleration-XY StDev"
]
class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def get_victim_file(victim_id):
    for f in os.listdir(VICTIMS_PATH):
        if f.startswith(f"{victim_id}-") and f.endswith(".xlsx"):
            return os.path.join(VICTIMS_PATH, f)
    raise FileNotFoundError("Victim 檔案不存在")

def get_attacker_files(victim_id):
    return [os.path.join(ATTACKERS_PATH, f) for f in os.listdir(ATTACKERS_PATH) if f.startswith(f"{victim_id}-")]

def read_featuredata(file_path):
    try:
        print(f"正在讀取檔案: {file_path}")
        df = pd.read_excel(file_path, sheet_name="featuredata").dropna()
        return df.reindex(columns=COLUMN_NAMES, fill_value=0)
    except Exception as e:
        print(f"錯誤：{e}")
        return None

def safe_concat(dfs):
    return pd.concat(dfs, ignore_index=True)

def get_all_negative_data(victims_files, current_victim_file, pos_count):
    all_neg = safe_concat([read_featuredata(f) for f in victims_files if f != current_victim_file and read_featuredata(f) is not None])
    return all_neg.sample(n=pos_count, random_state=42)

def get_attacker_negative_data(victim_id, pos_count):
    dfs = [read_featuredata(f) for f in get_attacker_files(victim_id) if read_featuredata(f) is not None]
    return safe_concat(dfs).sample(n=pos_count, random_state=42)

def train_model(train_pos, train_neg, device):
    train_pos["label"] = 1
    train_neg["label"] = 0
    df = pd.concat([train_pos, train_neg], ignore_index=True)

    X = df[COLUMN_NAMES].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    model = MLPModel(input_dim=X.shape[1]).to(device)
    pos_weight = torch.tensor([len(train_neg) / len(train_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
    return model, scaler

def test_model(model, scaler, test_pos, test_neg, device, threshold=0.4):
    test_pos["label"] = 1
    test_neg["label"] = 0
    df = pd.concat([test_pos, test_neg], ignore_index=True)

    X = scaler.transform(df[COLUMN_NAMES])
    y = df["label"].values
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).cpu().numpy().flatten()
    pred = (torch.sigmoid(torch.tensor(logits)) >= threshold).int().numpy()

    return classification_report(y, pred, zero_division=0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(LOG_DIR, exist_ok=True)
    victim_id = int(input("請輸入 Victim ID (3 到 12): "))

    victim_path = get_victim_file(victim_id)
    victim_data = read_featuredata(victim_path)
    train_pos, test_pos = train_test_split(victim_data, test_size=0.3, random_state=42)

    print(f"Train Positive Size: {len(train_pos)}")
    print(f"Test Positive Size: {len(test_pos)}")

    victim_files = [os.path.join(VICTIMS_PATH, f) for f in os.listdir(VICTIMS_PATH) if f.endswith(".xlsx")]
    train_neg = get_all_negative_data(victim_files, victim_path, len(train_pos))
    test_neg = get_attacker_negative_data(victim_id, len(test_pos))

    print(f"Train Negative Size: {len(train_neg)}")
    print(f"Test Negative Size: {len(test_neg)}")

    model, scaler = train_model(train_pos, train_neg, device)
    result = test_model(model, scaler, test_pos, test_neg, device, threshold=0.4)

    log_path = os.path.join(LOG_DIR, f"victim_{victim_id}_MLP_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=========================Test Results=========================\n")
        f.write(result)
    print(f"\n測試結果已儲存至 {log_path}")
