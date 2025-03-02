from datetime import datetime
from sqlalchemy import Column, Integer, DateTime
from app.core.database import Base


class BaseModel(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        columns = [c.name for c in self.__table__.columns]
        values = [getattr(self, c) for c in columns]
        params = ", ".join(f"{c}={v!r}" for c, v in zip(columns, values))
        return f"{self.__class__.__name__}({params})"
