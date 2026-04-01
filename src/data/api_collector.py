"""
공공데이터 API 수집 모듈
- 에어코리아: 미세먼지(PM10, PM2.5) 실시간 수집
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
    df["datetime"] = pd.to_datetime(df["datetime"])
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

    now = datetime.now()
    base_date = now.strftime("%Y%m%d")
    # 기상청 예보 발표 시간 (02, 05, 08, 11, 14, 17, 20, 23)
    base_time = "0500"

    base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": KMA_KEY,
        "pageNo": 1,
        "numOfRows": 100,
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
