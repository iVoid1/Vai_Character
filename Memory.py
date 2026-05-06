"""
memory.py
---------
نظام الذاكرة الكامل للـ Agent
  - Short-term  : آخر N رسالة (SQLite عبر SQLAlchemy)
  - Long-term   : vector store (Chroma + OpenAI embeddings)
  - Episodic    : ملخصات جلسات سابقة (SQLite)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column, String, Text, DateTime, Integer,
    create_engine, desc
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings          # ← بدّل للـ provider اللي تبيه
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# 1. Base ORM
# ─────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class EpisodicMemory(Base):
    """جلسات ملخّصة — كل صف = جلسة كاملة"""
    __tablename__ = "episodic_memory"

    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False, index=True)
    summary    = Column(Text,   nullable=False)
    mood       = Column(String, nullable=True)          # حالة الـ avatar وقتها
    created_at = Column(DateTime, default=datetime.utcnow)
    token_count= Column(Integer,  default=0)


class AgentState(Base):
    """حالة دائمة للـ agent بين الجلسات"""
    __tablename__ = "agent_state"

    key        = Column(String, primary_key=True)
    value      = Column(Text,   nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────────────────────────────────
# 2. Short-term — LangChain SQLChatMessageHistory
# ─────────────────────────────────────────────
def get_short_term_history(
    session_id: str,
    db_url: str = "sqlite:///vnyan_memory.db",
) -> SQLChatMessageHistory:
    """
    يرجع chat history مربوط بـ session_id.
    LangChain تدير الجدول تلقائيًا.
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=db_url,
    )


# ─────────────────────────────────────────────
# 3. Long-term — Chroma Vector Store
# ─────────────────────────────────────────────
class LongTermMemory:
    """
    تخزين نصوص مهمة كـ embeddings وبحث بالتشابه.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection: str  = "vnyan_longterm",
        embedding_model: str = "nomic-embed-text:latest",
    ):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.store = Chroma(
            collection_name=collection,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=60,
        )

    def save(self, text: str, metadata: Optional[dict] = None) -> List[str]:
        """يقطع النص ويحفظه كـ embeddings، يرجع IDs"""
        chunks = self.splitter.split_text(text)
        meta   = [metadata or {} for _ in chunks]
        ids    = self.store.add_texts(chunks, metadatas=meta)
        return ids

    def search(self, query: str, top_k: int = 4) -> List[str]:
        """يرجع أقرب N جمل للـ query"""
        docs = self.store.similarity_search(query, k=top_k)
        return [d.page_content for d in docs]

    def search_with_score(self, query: str, top_k: int = 4):
        """يرجع (نص, score) — مفيد للـ debug"""
        return self.store.similarity_search_with_score(query, k=top_k)


# ─────────────────────────────────────────────
# 4. Episodic — SQLAlchemy مباشرة
# ─────────────────────────────────────────────
class EpisodicStore:
    """حفظ واسترجاع ملخصات الجلسات"""

    def __init__(self, db_url: str = "sqlite:///vnyan_memory.db"):
        engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)

    # ── write ──
    def save_episode(
        self,
        session_id: str,
        summary: str,
        mood: Optional[str] = None,
        token_count: int = 0,
    ) -> str:
        with self.SessionLocal() as db:
            ep = EpisodicMemory(
                session_id=session_id,
                summary=summary,
                mood=mood,
                token_count=token_count,
            )
            db.add(ep)
            db.commit()
            return ep.id

    # ── read ──
    def get_recent_episodes(
        self,
        session_id: str,
        limit: int = 5,
    ) -> List[str]:
        with self.SessionLocal() as db:
            rows = (
                db.query(EpisodicMemory)
                .filter(EpisodicMemory.session_id == session_id)
                .order_by(desc(EpisodicMemory.created_at))
                .limit(limit)
                .all()
            )
        return [r.summary for r in reversed(rows)]

    def get_all_summaries(self, limit: int = 20) -> List[dict]:
        with self.SessionLocal() as db:
            rows = (
                db.query(EpisodicMemory)
                .order_by(desc(EpisodicMemory.created_at))
                .limit(limit)
                .all()
            )
        return [{"session": r.session_id, "summary": r.summary, "mood": r.mood} for r in rows]


# ─────────────────────────────────────────────
# 5. State Store — key/value دائم
# ─────────────────────────────────────────────
class StateStore:
    """حفظ حالة الـ agent (mood، persona، إلخ) بين الجلسات"""

    def __init__(self, db_url: str = "sqlite:///vnyan_memory.db"):
        engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)

    def set(self, key: str, value: str):
        with self.SessionLocal() as db:
            obj = db.get(AgentState, key)
            if obj:
                obj.value = value
                obj.updated_at = datetime.utcnow()
            else:
                db.add(AgentState(key=key, value=value))
            db.commit()

    def get(self, key: str, default: str = "") -> str:
        with self.SessionLocal() as db:
            obj = db.get(AgentState, key)
        return obj.value if obj else default

    def get_all(self) -> dict:
        with self.SessionLocal() as db:
            rows = db.query(AgentState).all()
        return {r.key: r.value for r in rows}


# ─────────────────────────────────────────────
# 6. MemoryManager — واجهة موحّدة
# ─────────────────────────────────────────────
class MemoryManager:
    """
    الكلاس الرئيسي — الـ Agent يتعامل معه فقط.
    
    مثال:
        mem = MemoryManager(session_id="session_001")
        mem.add_user_message("مرحبا")
        mem.add_ai_message("أهلاً وسهلاً!")
        context = mem.build_context("مرحبا")
    """

    def __init__(
        self,
        session_id: str,
        db_url:     str = "sqlite:///vnyan_memory.db",
        chroma_dir: str = "./chroma_db",
        max_short_term: int = 20,  # آخر N رسالة تُضاف للـ context
    ):
        self.session_id     = session_id
        self.max_short_term = max_short_term

        self.short_term  = get_short_term_history(session_id, db_url)
        self.long_term   = LongTermMemory(persist_dir=chroma_dir)
        self.episodic    = EpisodicStore(db_url)
        self.state       = StateStore(db_url)

    # ── write ──────────────────────────────────
    def add_user_message(self, text: str):
        self.short_term.add_user_message(text)

    def add_ai_message(self, text: str):
        self.short_term.add_ai_message(text)

    def save_important(self, text: str, metadata: Optional[dict] = None):
        """احفظ معلومة مهمة في الـ long-term"""
        return self.long_term.save(text, metadata)

    def close_session(self, summary: str, mood: Optional[str] = None):
        """استدعيه في نهاية الجلسة"""
        self.episodic.save_episode(self.session_id, summary, mood)
        self.short_term.clear()

    # ── read ───────────────────────────────────
    def get_recent_messages(self) -> List[BaseMessage]:
        msgs = self.short_term.messages
        return msgs[-self.max_short_term:]

    def build_context(self, query: str) -> str:
        """
        يبني context نصي كامل للـ LLM:
        - ملخصات episodic
        - نتائج long-term search
        """
        parts: list[str] = []

        # episodic
        episodes = self.episodic.get_recent_episodes(self.session_id, limit=3)
        if episodes:
            parts.append("## ملخص جلسات سابقة\n" + "\n---\n".join(episodes))

        # long-term search
        relevant = self.long_term.search(query, top_k=4)
        if relevant:
            parts.append("## معلومات ذات صلة\n" + "\n".join(f"- {r}" for r in relevant))

        return "\n\n".join(parts) if parts else ""

    # ── state shortcuts ─────────────────────────
    def get_mood(self)         -> str: return self.state.get("mood", "neutral")
    def set_mood(self, m: str)        : self.state.set("mood", m)
    def get_persona(self)      -> str: return self.state.get("persona", "friendly assistant")
    def set_persona(self, p: str)     : self.state.set("persona", p)