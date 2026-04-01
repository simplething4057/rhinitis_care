"""
환경별 설정 로더
APP_ENV 환경변수에 따라 config.yaml + config.{env}.yaml 을 병합하여 반환합니다.
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 자동 로드 (APP_ENV 포함)
load_dotenv()

_CONFIG_DIR = Path(__file__).parents[2] / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    """중첩 딕셔너리를 재귀적으로 병합합니다. override 값이 우선합니다."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> dict:
    """
    설정을 로드합니다.

    우선순위 (낮음 → 높음):
      config.yaml  →  config.{APP_ENV}.yaml  →  환경변수

    Returns:
        dict: 병합된 설정 딕셔너리
    """
    env = os.getenv("APP_ENV", "dev").lower()

    # 1. 공통 설정 로드
    base_file = _CONFIG_DIR / "config.yaml"
    with open(base_file, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 2. 환경별 오버라이드 병합
    env_file = _CONFIG_DIR / f"config.{env}.yaml"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        config = _deep_merge(config, overrides)

    # 3. API 키 환경변수 주입 (실제 값으로 치환)
    if "api" in config:
        config["api"]["airkorea_key"] = os.getenv(
            "AIRKOREA_API_KEY", config["api"].get("airkorea_key", "")
        )
        config["api"]["kma_key"] = os.getenv(
            "KMA_API_KEY", config["api"].get("kma_key", "")
        )

    config["_env"] = env  # 현재 환경 이름 포함
    return config


def get_env() -> str:
    """현재 APP_ENV 반환 (기본값: dev)"""
    return os.getenv("APP_ENV", "dev").lower()
