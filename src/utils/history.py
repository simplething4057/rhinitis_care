import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from src.database import SessionLocal, engine, Base
from src.models import PredictionHistory

# 앱 실행 시 테이블 자동 생성 (URL 설정이 된 경우에만)
if engine is not None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"⚠️ 테이블 생성 중 오류 발생 (DB가 실행 중인지 확인하세요): {e}")
else:
    print("⚠️ DB 연결이 설정되지 않아 테이블을 생성하지 않았습니다.")

def save_history(record: dict, user_id: str):
    if SessionLocal is None:
        print("⚠️ DB 연결이 설정되지 않았으므로 기록을 저장하지 않습니다.")
        return

    db: Session = SessionLocal()
    try:
        new_record = PredictionHistory(
            user_id=user_id,
            cluster_label=record["cluster_label"],
            symptom_rhinorrhea=record["symptom_rhinorrhea"],
            symptom_congestion=record["symptom_congestion"],
            symptom_sneezing=record["symptom_sneezing"],
            symptom_ocular=record["symptom_ocular"],
            pm10=record.get("pm10", 0),
            pm25=record.get("pm25", 0),
            humidity=record.get("humidity", 50),
            temperature=record.get("temperature", 20),
            created_at=datetime.now()
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
    except Exception as e:
        db.rollback()
        raise RuntimeError(f"이력 저장 실패: {e}") from e
    finally:
        db.close()

def get_recent_history(user_id: str, days=7):
    if SessionLocal is None:
        return pd.DataFrame()

    db: Session = SessionLocal()
    try:
        threshold = datetime.now() - timedelta(days=days)
        results = db.query(PredictionHistory).filter(
            PredictionHistory.user_id == user_id,
            PredictionHistory.created_at >= threshold
        ).order_by(PredictionHistory.created_at.asc()).all()
        
        if results:
            # SQLAlchemy 객체를 데이터프레임으로 변환
            data_list = []
            for r in results:
                data_list.append({
                    "timestamp": r.created_at,
                    "cluster_label": r.cluster_label,
                    "symptom_rhinorrhea": r.symptom_rhinorrhea,
                    "symptom_congestion": r.symptom_congestion,
                    "symptom_sneezing": r.symptom_sneezing,
                    "symptom_ocular": r.symptom_ocular,
                    "pm10": r.pm10,
                    "pm25": r.pm25,
                    "humidity": r.humidity,
                    "temperature": r.temperature
                })
            return pd.DataFrame(data_list)
        return pd.DataFrame()
    except Exception as e:
        print(f"데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def get_cluster_avg_history(cluster_label: str, days: int = 7) -> pd.DataFrame:
    """동일 유형(cluster_label) 전체 대상자의 날짜별 평균 증상 점수 반환.
    DB 연결이 없거나 해당 유형 데이터가 없으면 빈 DataFrame 반환.
    """
    if SessionLocal is None:
        return pd.DataFrame()

    db: Session = SessionLocal()
    try:
        threshold = datetime.now() - timedelta(days=days)
        # 구 명칭 호환 매핑 (이전 DB 레코드 포함)
        _label_compat = {
            "호흡기 알레르기형":  ["호흡기 알레르기형", "콧물·재채기 우세형"],
            "비염+천식 복합형":   ["비염+천식 복합형",  "코막힘 우세형"],
            "아토픽 마치형":      ["아토픽 마치형",     "복합 과민형"],
        }
        labels = _label_compat.get(cluster_label, [cluster_label])

        results = db.query(PredictionHistory).filter(
            PredictionHistory.cluster_label.in_(labels),
            PredictionHistory.created_at >= threshold
        ).order_by(PredictionHistory.created_at.asc()).all()

        if not results:
            return pd.DataFrame()

        data_list = [{
            "date": r.created_at.date(),
            "symptom_rhinorrhea": r.symptom_rhinorrhea,
            "symptom_congestion": r.symptom_congestion,
            "symptom_sneezing":   r.symptom_sneezing,
            "symptom_ocular":     r.symptom_ocular,
        } for r in results]

        df = pd.DataFrame(data_list)
        # 날짜별 평균 집계
        avg_df = df.groupby("date")[
            ["symptom_rhinorrhea", "symptom_congestion", "symptom_sneezing", "symptom_ocular"]
        ].mean().reset_index()
        avg_df["date"] = pd.to_datetime(avg_df["date"])
        return avg_df

    except Exception as e:
        print(f"동일 유형 평균 조회 중 오류: {e}")
        return pd.DataFrame()
    finally:
        db.close()


def generate_synthetic_history(user_id: str):
    """최초 실행 시 가상 데이터 생성 (SQLAlchemy 저장)"""
    if SessionLocal is None:
        print("⚠️ DB 연결이 설정되지 않았으므로 가상 데이터를 생성하지 않습니다.")
        return

    import random
    db: Session = SessionLocal()
    
    types = ["콧물·재채기 우세형", "코막힘 우세형", "복합 과민형"]
    now = datetime.now()
    
    try:
        # 이미 데이터가 있으면 생성하지 않음 (중복 생성 방지)
        if db.query(PredictionHistory).filter(PredictionHistory.user_id == user_id).first():
            return

        records = []
        for i in range(7, 0, -1):
            dt = now - timedelta(days=i)
            label = random.choice(types)
            new_record = PredictionHistory(
                user_id=user_id,
                created_at=dt,
                cluster_label=label,
                symptom_rhinorrhea=random.randint(3, 8),
                symptom_congestion=random.randint(3, 9),
                symptom_sneezing=random.randint(2, 7),
                symptom_ocular=random.randint(1, 6),
                pm10=random.randint(20, 100),
                humidity=random.randint(30, 80),
                temperature=random.randint(15, 25)
            )
            records.append(new_record)
        
        db.bulk_save_objects(records)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"가상 데이터 생성 중 오류 발생: {e}")
    finally:
        db.close()
