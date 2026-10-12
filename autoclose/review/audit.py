"""Audit trail in SQL. SQLite by default; set DATABASE_URL=postgresql+psycopg://... for PostgreSQL."""
import json, os, time, threading
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True)
    case_id = Column(String(40), index=True)
    step = Column(String(40))
    detail = Column(Text)
    ts = Column(Float)

class AuditLog:
    def __init__(self, url=None):
        url = url or os.getenv("DATABASE_URL", "sqlite://")      # in-memory unless told otherwise
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
        if url in ("sqlite://", "sqlite:///:memory:"):
            from sqlalchemy.pool import StaticPool
            kw["poolclass"] = StaticPool
        self.engine = create_engine(url, **kw)
        Base.metadata.create_all(self.engine)
        self.S = sessionmaker(self.engine)
        self.lock = threading.Lock()
    def log(self, case_id, step, detail):
        with self.lock, self.S() as s:
            s.add(AuditEvent(case_id=case_id, step=step, detail=json.dumps(detail, default=str), ts=time.time())); s.commit()
    def events(self, case_id=None, limit=500):
        with self.lock, self.S() as s:
            q = s.query(AuditEvent).order_by(AuditEvent.id)
            if case_id: q = q.filter(AuditEvent.case_id == case_id)
            return [{"id": e.id, "case_id": e.case_id, "step": e.step, "detail": json.loads(e.detail), "ts": e.ts} for e in q.limit(limit)]
    def clear(self):
        with self.lock, self.S() as s:
            s.query(AuditEvent).delete(); s.commit()
