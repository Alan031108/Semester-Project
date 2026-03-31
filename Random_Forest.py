import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = r"C:\Users\方安政\Desktop\程式作業\semester_project\Dataset\DynamicData"
VICTIMS_PATH = os.path.join(DATA_PATH, "Victims")
ATTACKERS_PATH = os.path.join(DATA_PATH, "Attackers")
LOG_DIR = "./logs"

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

def get_victim_file(victim_id):
    for file in os.listdir(VICTIMS_PATH):
        if file.startswith(f"{victim_id}-") and file.endswith(".xlsx"):
            return os.path.join(VICTIMS_PATH, file)
    raise FileNotFoundError(f"未找到 Victim ID {victim_id} 對應的檔案！")

def get_attacker_files(victim_id):
    return [
        os.path.join(ATTACKERS_PATH, file)
        for file in os.listdir(ATTACKERS_PATH)
        if file.startswith(f"{victim_id}-") and file.endswith(".xlsx")
    ]

def read_featuredata(file_path):
    print(f"正在讀取檔案: {file_path}")
    try:
        data = pd.read_excel(file_path, sheet_name="featuredata")
        data.dropna(inplace=True)
        return data.reindex(columns=COLUMN_NAMES, fill_value=0)
    except Exception as e:
        print(f"讀取檔案錯誤: {e}")
        return None

def safe_concat(dataframes):
    return pd.concat(dataframes, ignore_index=True)

def get_all_negative_data(victims_files, current_victim_file, positive_count):
    negative_pool = [
        read_featuredata(f) for f in victims_files
        if f != current_victim_file and read_featuredata(f) is not None
    ]
    all_negative = safe_concat(negative_pool)
    return all_negative.sample(n=positive_count * 2, random_state=42) 

def get_attacker_negative_data(victim_id):
    attacker_files = get_attacker_files(victim_id)
    data = [read_featuredata(file) for file in attacker_files if read_featuredata(file) is not None]
    return safe_concat(data)

def train_model(train_positive, train_negative):
    train_positive["label"] = "positive"
    train_negative["label"] = "negative"
    train_data = pd.concat([train_positive, train_negative], ignore_index=True)
    
    X_train = train_data[COLUMN_NAMES]
    y_train = train_data["label"]

    # 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    log = []
    log.append("=========================Original Data=========================")
    log.append(f"Positive Samples: {len(train_positive)}")
    log.append(f"Negative Samples: {len(train_negative)}")

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, log

def test_model(model, scaler, test_positive, test_negative):
    test_positive["label"] = "positive"
    test_negative["label"] = "negative"
    test_data = pd.concat([test_positive, test_negative], ignore_index=True)
    X_test = test_data[COLUMN_NAMES]
    y_test = test_data["label"]

    # 套用訓練時的 scaler
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    report = classification_report(y_test, y_pred, zero_division=0)
    return report

if __name__ == "__main__":
    try:
        victim_id = int(input("請輸入 Victim ID (3 到 12): "))
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, f"victim_{victim_id}_RandomForest_log.txt")

        victim_file_path = get_victim_file(victim_id)
        victim_data = read_featuredata(victim_file_path)
        if victim_data is None or len(victim_data) == 0:
            raise ValueError("Victim 資料為空或讀取失敗！")

        train_positive, test_positive = train_test_split(victim_data, test_size=0.3, random_state=42)
        print(f"Train Positive Data Size: {len(train_positive)}")
        print(f"Test Positive Data Size: {len(test_positive)}")

        victims_files = [os.path.join(VICTIMS_PATH, f) for f in os.listdir(VICTIMS_PATH) if f.endswith(".xlsx")]
        train_negative = get_all_negative_data(victims_files, victim_file_path, len(train_positive))
        test_negative = get_attacker_negative_data(victim_id)

        model, scaler, train_log = train_model(train_positive, train_negative)
        test_log = test_model(model, scaler, test_positive, test_negative)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_log))
            f.write("\n=========================Test Results=========================\n")
            f.write(test_log)
        print(f"\n測試結果已儲存至 {log_path}")

    except Exception as e:
        print(f" 發生錯誤：{e}")