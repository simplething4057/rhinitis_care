"""
공공데이터 API 수집 모듈
- 에어코리아: 미세먼지(PM10, PM2.5), 꽃가루 위험지수 실시간 수집
- 기상청: 기온, 습도 수집

캐싱 정책:
  - 에어코리아: TTL 60분 (station_name 기준)
  - 기상청:     TTL 60분 (nx, ny 격자 기준)
  - clear_cache() 로 강제 초기화 가능
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

AIRKOREA_KEY = os.getenv("AIRKOREA_API_KEY", "")
KMA_KEY = os.getenv("KMA_API_KEY", "")
POLLEN_KEY = os.getenv("POLLEN_API_KEY") or AIRKOREA_KEY  # 꽃가루 전용 키 없으면 에어코리아 키로 폴백

# ──────────────────────────────────────────────
# 간단한 TTL 인메모리 캐시
# ──────────────────────────────────────────────
_CACHE_TTL_SEC = 3600  # 1시간

_cache: dict[str, tuple[float, pd.DataFrame]] = {}  # key → (stored_at, df)


def _cache_get(key: str) -> pd.DataFrame | None:
    """캐시 히트 시 DataFrame 반환, 만료/미존재 시 None."""
    if key not in _cache:
        return None
    stored_at, df = _cache[key]
    if time.time() - stored_at > _CACHE_TTL_SEC:
        del _cache[key]
        return None
    logger.debug(f"[cache HIT] {key}")
    return df


def _cache_set(key: str, df: pd.DataFrame) -> None:
    _cache[key] = (time.time(), df)
    logger.debug(f"[cache SET] {key}")


def clear_cache() -> None:
    """전체 캐시 초기화 (테스트·강제 갱신용)."""
    _cache.clear()
    logger.info("API 캐시 초기화 완료")


# ──────────────────────────────────────────────
# 에어코리아 대기오염 수집
# ──────────────────────────────────────────────
def fetch_airkorea(station_name: str = "종로구", date_str: str | None = None) -> pd.DataFrame:
    """
    에어코리아 시간대별 대기오염 데이터 수집.
    station_name: 측정소 이름 (예: '종로구', '강남구')
    date_str: 'YYYY-MM-DD' 형식. None이면 오늘.

    결과는 TTL 60분 인메모리 캐시에 보관됩니다.
    """
    if not AIRKOREA_KEY:
        raise EnvironmentError("AIRKOREA_API_KEY 환경변수가 설정되지 않았습니다.")

    cache_key = f"airkorea:{station_name}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    base_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    params = {
        "serviceKey": AIRKOREA_KEY,
        "returnType": "json",
        "numOfRows": 24,
        "pageNo": 1,
        "stationName": station_name,
        "dataTerm": "DAILY",
        "ver": "1.3",
    }

    resp = requests.get(base_url, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json()["response"]["body"]["items"]

    if not items:
        raise ValueError(
            f"'{station_name}' 측정소 데이터가 없습니다. "
            "측정소명이 정확한지 확인하세요 (예: '종로구', '인계동')."
        )

    records = []
    for item in items:
        records.append({
            "datetime": item.get("dataTime"),
            "pm10":     _safe_float(item.get("pm10Value")),
            "pm25":     _safe_float(item.get("pm25Value")),
            "o3":       _safe_float(item.get("o3Value")),
            "station":  station_name,
        })

    df = pd.DataFrame(records)
    # 에어코리아는 자정을 "YYYY-MM-DD 24:00"으로 반환 → 다음날 00:00으로 변환
    mask_24 = df["datetime"].str.contains("24:00", na=False)
    df["datetime"] = pd.to_datetime(
        df["datetime"].str.replace(" 24:00", " 00:00", regex=False),
        format="%Y-%m-%d %H:%M",
    )
    df.loc[mask_24, "datetime"] += pd.Timedelta(days=1)
    df["date"] = df["datetime"].dt.date
    logger.info(f"에어코리아 수집 완료: {len(df)}건 (station={station_name})")
    _cache_set(cache_key, df)
    return df


# ──────────────────────────────────────────────
# 기상청 단기예보 수집
# ──────────────────────────────────────────────
def fetch_kma_forecast(nx: int = 60, ny: int = 127) -> pd.DataFrame:
    """
    기상청 단기예보 (기온, 습도) 수집.
    nx, ny: 격자 좌표 (서울 기본값)

    결과는 TTL 60분 인메모리 캐시에 보관됩니다.
    """
    if not KMA_KEY:
        raise EnvironmentError("KMA_API_KEY 환경변수가 설정되지 않았습니다.")

    cache_key = f"kma:{nx}:{ny}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # KMA API는 KST(UTC+9) 기준으로 동작 — 서버 타임존 무관하게 KST 사용
    from datetime import timezone, timedelta as _td
    _KST = timezone(_td(hours=9))
    now = datetime.now(_KST).replace(tzinfo=None)

    # 기상청 예보 발표 시간 (02, 05, 08, 11, 14, 17, 20, 23) — 발표 후 약 10분 지연
    _ISSUE_HOURS = [2, 5, 8, 11, 14, 17, 20, 23]
    cur_hour = now.hour - (1 if now.minute < 10 else 0)
    past = [h for h in _ISSUE_HOURS if h <= cur_hour]
    if past:
        base_date = now.strftime("%Y%m%d")
        base_time = f"{past[-1]:02d}00"
    else:
        # 자정~02:10 사이 → 전날 23시 예보
        from datetime import timedelta
        prev = now - timedelta(days=1)
        base_date = prev.strftime("%Y%m%d")
        base_time = "2300"

    base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": KMA_KEY,
        "pageNo": 1,
        "numOfRows": 300,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }

    resp = requests.get(base_url, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json()["response"]["body"]["items"]["item"]

    target_categories = {"TMP": "temperature", "REH": "humidity", "PCP": "precipitation"}
    records = {}
    for item in items:
        cat = item["category"]
        if cat in target_categories:
            key = f"{item['fcstDate']}_{item['fcstTime']}"
            if key not in records:
                records[key] = {"date": item["fcstDate"], "time": item["fcstTime"]}
            records[key][target_categories[cat]] = _safe_float(item["fcstValue"])

    df = pd.DataFrame(list(records.values()))
    logger.info(f"기상청 예보 수집 완료: {len(df)}건")
    _cache_set(cache_key, df)
    return df


# ──────────────────────────────────────────────
# 꽃가루 위험지수 수집 (기상청 HealthWthrIdxServiceV3)
# ──────────────────────────────────────────────

# 기상청 꽃가루 V3 API 설정
_KMA_POLLEN_BASE = "https://apis.data.go.kr/1360000/HealthWthrIdxServiceV3"

# 시도명 → 10자리 행정구역코드
_KMA_POLLEN_AREA: dict[str, str] = {
    "서울": "1100000000",
    "부산": "2600000000",
    "대구": "2700000000",
    "인천": "2800000000",
    "광주": "2900000000",
    "대전": "3000000000",
    "울산": "3100000000",
    "세종": "3611000000",
    "경기": "4100000000",
    "강원": "5100000000",
    "충북": "4300000000",
    "충남": "4400000000",
    "전북": "4500000000",
    "전남": "4600000000",
    "경북": "4700000000",
    "경남": "4800000000",
    "제주": "5000000000",
}

# 꽃가루 종류 → API 오퍼레이션명
_KMA_POLLEN_OPS: dict[str, str] = {
    "소나무": "getPinePollenRiskIdxV3",
    "참나무": "getOakPollenRiskIdxV3",
    "자작나무": "getBirchPollenRiskIdxV3",
    "쑥":    "getWeedPollenRiskIdxV3",
}

# 수목·잡초 꽃가루 분류 (app.py 호환)
POLLEN_TREE_KEYS: dict[str, str] = {
    "소나무": "소나무",
    "참나무": "참나무",
    "자작나무": "자작나무",
}
POLLEN_WEED_KEYS: dict[str, str] = {
    "쑥": "쑥",
}
# 등급 → (레이블, 색상)
POLLEN_GRADE_MAP: dict[int, tuple[str, str]] = {
    1: ("낮음",    "#4CAF50"),
    2: ("보통",    "#FFC107"),
    3: ("높음",    "#FF5722"),
    4: ("매우높음", "#B71C1C"),
}


class PollenAPIUnavailableError(Exception):
    """꽃가루 API 서비스 미구독 또는 서버 오류."""


def fetch_pollen(sido: str = "서울") -> pd.DataFrame:
    """
    기상청 꽃가루농도위험지수(3.0) 수집 — HealthWthrIdxServiceV3.
    sido: 시도명 (예: '서울', '경기', '부산')
    반환: columns = ['sido', 'dataTime', '소나무', '참나무', '자작나무', '쑥'], 등급 1~4

    결과는 TTL 60분 인메모리 캐시에 보관됩니다.
    """
    if not POLLEN_KEY:
        raise EnvironmentError("POLLEN_API_KEY (또는 AIRKOREA_API_KEY) 환경변수가 설정되지 않았습니다.")

    cache_key = f"pollen:{sido}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    from datetime import timezone, timedelta as _td
    _KST = timezone(_td(hours=9))
    now = datetime.now(_KST).replace(tzinfo=None)
    # 기상청 V3: time 파라미터 YYYYMMDDHH (06 또는 18)
    hour = "06" if now.hour < 18 else "18"
    time_str = now.strftime("%Y%m%d") + hour

    area_code = _KMA_POLLEN_AREA.get(sido, "1100000000")
    row: dict = {"sido": sido, "dataTime": time_str}

    for pollen_name, operation in _KMA_POLLEN_OPS.items():
        try:
            resp = requests.get(
                f"{_KMA_POLLEN_BASE}/{operation}",
                params={
                    "serviceKey": POLLEN_KEY,
                    "pageNo": 1,
                    "numOfRows": 10,
                    "dataType": "JSON",
                    "areaNo": area_code,
                    "time": time_str,
                },
                timeout=10,
            )
            resp.raise_for_status()
            body = resp.json()["response"]["body"]
            items = body.get("items") or {}
            item_list = items.get("item", []) if isinstance(items, dict) else (items or [])
            if item_list:
                item = item_list[0] if isinstance(item_list, list) else item_list
                # h0 = 오늘 예측값 (등급 1~4)
                grade = item.get("h0") or item.get("today") or item.get("value")
                row[pollen_name] = _safe_float(grade)
            else:
                row[pollen_name] = None
        except Exception as e:
            logger.warning(f"꽃가루 {pollen_name} 수집 실패 ({operation}): {e}")
            row[pollen_name] = None

    df = pd.DataFrame([row])
    logger.info(f"기상청 꽃가루 수집 완료: sido={sido}, time={time_str}")
    _cache_set(cache_key, df)
    return df


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def save_to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"저장 완료: {path}")


if __name__ == "__main__":
    # 테스트 실행 (API 키 필요)
    air_df = fetch_airkorea(station_name="종로구")
    print(air_df.head())
