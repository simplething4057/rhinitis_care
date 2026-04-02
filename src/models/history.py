from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from src.database import Base

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, index=True)
    cluster_label = Column(String)
    symptom_rhinorrhea = Column(Integer)
    symptom_congestion = Column(Integer)
    symptom_sneezing = Column(Integer)
    symptom_ocular = Column(Integer)
    pm10 = Column(Float)
    pm25 = Column(Float)
    humidity = Column(Float)
    temperature = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
