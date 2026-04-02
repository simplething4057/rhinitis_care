import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

engine       = None
SessionLocal = None

if SQLALCHEMY_DATABASE_URL:
    try:
        _connect_args = {}
        # Supabase / 외부 PostgreSQL은 SSL 필요
        if "supabase" in SQLALCHEMY_DATABASE_URL or "postgresql" in SQLALCHEMY_DATABASE_URL:
            _connect_args = {"sslmode": "require"}
        engine = create_engine(
            SQLALCHEMY_DATABASE_URL,
            connect_args=_connect_args,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e:
        engine       = None
        SessionLocal = None
        print(f"⚠️ SQLAlchemy 연결 실패: {e}")
else:
    print("⚠️ DATABASE_URL 미설정 — 이력 저장 비활성화")

Base = declarative_base()

def get_db():
    if not SessionLocal:
        raise RuntimeError("DB 연결이 설정되지 않았습니다. DATABASE_URL을 확인하세요.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
