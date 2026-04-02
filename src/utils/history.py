import json
import os
import pandas as pd
from datetime import datetime, timedelta

HISTORY_FILE = "data/prediction_history.json"

def save_history(record: dict):
    os.makedirs("data", exist_ok=True)
    history = load_history()
    
    # Add timestamp if not exists
    if "timestamp" not in record:
        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    history.append(record)
    
    # Keep only records from last 30 days (optional, can be more)
    # but for simplicity, let's just save everything and filter on load
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def get_recent_history(days=7):
    history = load_history()
    if not history:
        return pd.DataFrame()
        
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    threshold = datetime.now() - timedelta(days=days)
    recent_df = df[df["timestamp"] >= threshold].sort_values("timestamp")
    return recent_df

def generate_synthetic_history():
    """테스트용 가상 데이터 생성 (현재 시점 기준 지난 7일치)"""
    import random
    
    history = []
    types = ["콧물·재채기 우세형", "코막힘 우세형", "복합 과민형"]
    now = datetime.now()
    
    for i in range(7, 0, -1):
        dt = now - timedelta(days=i)
        # Random but slightly stable type
        label = types[0] if i > 3 else types[1] # Change type mid-week
        record = {
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "cluster_label": label,
            "symptom_rhinorrhea": random.randint(3, 8),
            "symptom_congestion": random.randint(3, 9),
            "symptom_sneezing": random.randint(2, 7),
            "symptom_ocular": random.randint(1, 6),
            "pm10": random.randint(20, 100),
            "humidity": random.randint(30, 80)
        }
        history.append(record)
    
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return pd.DataFrame(history)
