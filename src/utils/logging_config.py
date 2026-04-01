"""
중앙화된 로깅 설정
앱 시작 시 setup_logging()을 한 번만 호출하면
전체 모듈에 동일한 포맷/레벨/핸들러가 적용됩니다.
"""
import logging
import logging.handlers
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).parents[2] / "outputs" / "logs"
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_configured = False


def setup_logging(log_level: str = "INFO", enable_file: bool = True) -> None:
    """
    루트 로거를 설정합니다. 앱 시작 시 한 번만 호출하세요.

    Args:
        log_level:   로그 레벨 문자열 (DEBUG / INFO / WARNING / ERROR)
        enable_file: True이면 outputs/logs/app.log 에 로테이션 파일 핸들러 추가
    """
    global _configured
    if _configured:
        return

    level = getattr(logging, log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # 기존 핸들러 제거 (uvicorn 등이 미리 붙인 것 방지)
    root.handlers.clear()

    # 콘솔 핸들러
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_FORMATTER)
    root.addHandler(stream_handler)

    # 파일 핸들러 (rotating: 5MB × 3개)
    if enable_file:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_DIR / "app.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(_FORMATTER)
        root.addHandler(file_handler)

    # 외부 라이브러리 로그 레벨 억제
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).info(
        f"로깅 설정 완료 (level={log_level}, file={enable_file})"
    )
