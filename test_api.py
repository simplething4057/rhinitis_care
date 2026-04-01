"""
API 테스트 스크립트 (PowerShell curl 대체용)
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def print_response(title, response):
    print(f"\n{'='*50}")
    print(f"[{title}]  status: {response.status_code}")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

# ── 1. 헬스체크 ──────────────────────────────────────
res = requests.get(f"{BASE_URL}/health")
print_response("GET /health", res)

# ── 2. 전체 유형 목록 ─────────────────────────────────
res = requests.get(f"{BASE_URL}/clusters")
print_response("GET /clusters", res)

# ── 3. 예측 테스트 3가지 케이스 ──────────────────────
cases = [
    {
        "name": "호흡기 알레르기형 예상",
        "data": {
            "has_asthma": 0, "has_atopic_derm": 0,
            "has_food_allergy": 0, "food_allergy_count": 0,
            "rhinitis_onset_age": 8.0, "rhinitis_duration": 2.0,
            "atopic_march": 0
        }
    },
    {
        "name": "비염+천식 복합형 예상",
        "data": {
            "has_asthma": 1, "has_atopic_derm": 0,
            "has_food_allergy": 0, "food_allergy_count": 0,
            "rhinitis_onset_age": 7.0, "rhinitis_duration": 3.0,
            "atopic_march": 0
        }
    },
    {
        "name": "아토픽 마치형 예상",
        "data": {
            "has_asthma": 1, "has_atopic_derm": 1,
            "has_food_allergy": 1, "food_allergy_count": 2,
            "rhinitis_onset_age": 5.0, "rhinitis_duration": 4.0,
            "atopic_march": 1
        }
    },
]

for case in cases:
    res = requests.post(f"{BASE_URL}/predict", json=case["data"])
    print(f"\n{'='*50}")
    print(f"[POST /predict] {case['name']}  status: {res.status_code}")
    data = res.json()
    print(f"  → 유형:   {data['result']['cluster_label']}")
    print(f"  → 신뢰도: {data['result']['confidence']*100:.1f}%")
    print(f"  → 요약:   {data['summary']}")
    print(f"  → 가이드 첫줄: {data['result']['guide'][0]}")
