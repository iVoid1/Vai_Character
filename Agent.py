"""
agent.py
--------
AI Agent for vnyan using the new LangChain (LCEL + langgraph).
Controls:
  - Text responses
  - Facial expressions (blendshapes)
  - Head movement
  - Avatar mood state
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_ollama import ChatOllama         # ← Switch provider as needed
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from Memory import MemoryManager


# ══════════════════════════════════════════════
# 1. VMC Tools — Each tool controls an avatar aspect
# ══════════════════════════════════════════════

# Reference for VMCController, injected from main.py
_vmc_ref: Any = None

def set_vmc(vmc_instance):
    global _vmc_ref
    _vmc_ref = vmc_instance


@tool
def set_avatar_expression(expressions_json: str) -> str:
    """
    Sets avatar facial expressions.
    Input: JSON string like {"happy": 80, "surprised": 30}
    Values range from 0 to 100.
    """
    print(f"set_avatar_expression: {expressions_json}")
    if _vmc_ref is None:
        return "VMC not connected"
    try:
        exprs = json.loads(expressions_json)
        _vmc_ref.smooth_expression(exprs, normalize=True, smoothness=0.15)
        return f"Expression set: {exprs}"
    except Exception as e:
        return f"Error: {e}"


@tool
def move_head(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> str:
    """
    Moves the avatar's head.
    pitch: Forward/backward tilt (-30 to 30)
    yaw  : Left/right rotation (-45 to 45)
    roll : Side tilt (-20 to 20)
    """
    if _vmc_ref is None:
        return "VMC not connected"
    _vmc_ref.smooth_move("Head", {"pitch": pitch, "yaw": yaw, "roll": roll}, lerp_speed=0.15)
    return f"Head moved: pitch={pitch}, yaw={yaw}, roll={roll}"


# @tool
# def set_mood(mood: str) -> str:
#     """
#     Changes the agent's mood state.
#     Available values: happy, sad, surprised, angry, neutral, excited, calm
#     """
#     print(f"set_mood: {mood}")
#     MOOD_EXPRESSIONS = {
#         "happy"     : {"happy": 70, "relaxed": 30},
#         "sad"       : {"sad": 60, "relaxed": 20},
#         "surprised" : {"surprised": 80, "blink": 10},
#         "angry"     : {"angry": 60, "lookDown": 20},
#         "neutral"   : {"neutral": 100},
#         "excited"   : {"happy": 90, "surprised": 40},
#         "calm"      : {"relaxed": 70, "neutral": 30},
#     }
#     if _vmc_ref is None:
#         return "VMC not connected"

#     exprs = MOOD_EXPRESSIONS.get(mood.lower(), {"neutral": 100})
#     _vmc_ref.smooth_expression(exprs, normalize=True, smoothness=0.1)
#     return f"Mood set to: {mood}"


@tool
def save_to_memory(text: str) -> str:
    """
    Saves important information into long-term memory.
    Use this when the user mentions something important to remember.
    """
    print(f"save_to_memory: {text}")
    # _memory_ref is injected from VnyanAgent
    if _memory_ref is None:
        return "Memory not connected"
    ids = _memory_ref.save_important(text, metadata={"source": "user_info"})
    return f"Saved to long-term memory (ids: {ids[:2]}...)"


# noinspection PyUnresolvedReferences
_memory_ref: Optional[MemoryManager] = None

def set_memory_ref(mem: MemoryManager):
    global _memory_ref
    _memory_ref = mem


# ══════════════════════════════════════════════
# 2. Persona / System Prompt
# ══════════════════════════════════════════════

DEFAULT_PERSONA = """You are Aria, an intelligent assistant living inside a VRM avatar within the vnyan environment.
You have a warm and cheerful personality, and you speak naturally.

When responding:
1. Reply to the user naturally first.
2. Use expression and movement tools to embody your emotions.
3. Save important user information in memory.

Expression Rules:
- Joy        → set_mood("happy")
- Surprise   → set_mood("surprised") + move_head(pitch=-5)
- Thinking   → move_head(yaw=15, roll=5)
- Sadness    → set_mood("sad") + move_head(pitch=10)
- Excitement → set_mood("excited") + move_head(pitch=-10)

Do not mention tool names in your responses — just use them naturally.
"""


def build_system_prompt(persona: str, extra_context: str = "") -> str:
    base = persona
    if extra_context:
        base += f"\n\n{extra_context}"
    return base


# ══════════════════════════════════════════════
# 3. VnyanAgent — Main Class
# ══════════════════════════════════════════════

class VnyanAgent:
    """
    Main Agent class.

    Example:
        from vmc_controller import VMCController
        vmc = VMCController("127.0.0.1", 8000, "avatar.vrm")
        agent = VnyanAgent(vmc=vmc)
        response = agent.chat("Hello, how are you?")
    """

    TOOLS = [set_avatar_expression, move_head, save_to_memory]

    def __init__(
        self,
        vmc=None,
        session_id:     Optional[str] = None,
        persona:        str  = DEFAULT_PERSONA,
        model:          str  = "gpt-4o-mini",
        temperature:    float = 0.7,
        db_url:         str  = "sqlite:///vnyan_memory.db",
        chroma_dir:     str  = "./chroma_db",
        max_short_term: int  = 20,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.persona    = persona

        # ── Memory ──
        self.memory = MemoryManager(
            session_id=self.session_id,
            db_url=db_url,
            chroma_dir=chroma_dir,
            max_short_term=max_short_term,
        )

        # ── VMC ──
        if vmc:
            set_vmc(vmc)
        set_memory_ref(self.memory)

        # ── LLM ──
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            streaming=False,
        )

        # ── LangGraph Agent (LCEL + ReAct) ──
        self.checkpointer = MemorySaver()
        self.agent = create_agent(
            model=self.llm,
            tools=self.TOOLS,
            checkpointer=self.checkpointer,
        )

    # ─────────────────────────────────────────
    def _build_config(self) -> dict:
        """LangGraph thread config"""
        return {"configurable": {"thread_id": self.session_id}}

    def _get_system_message(self, user_query: str) -> SystemMessage:
        context = self.memory.build_context(user_query)
        mood    = self.memory.get_mood()
        full_prompt = build_system_prompt(
            self.persona,
            extra_context=f"Current state: {mood}\n\n{context}" if context else f"Current state: {mood}",
        )
        return SystemMessage(content=full_prompt)

    # ─────────────────────────────────────────
    def chat(self, user_input: str) -> str:
        """
        Sends a message and returns the agent's response.
        Updates memory automatically.
        """
        # Save to short-term
        self.memory.add_user_message(user_input)

        system_msg = self._get_system_message(user_input)

        # Run the agent
        result = self.agent.invoke(
            {
                "messages": [
                    system_msg,
                    HumanMessage(content=user_input),
                ]
            },
            config=self._build_config(),
        )

        # Extract final response
        ai_response = result["messages"][-1].content

        # Save to short-term
        self.memory.add_ai_message(ai_response)

        # Sync mood in state
        self._sync_mood_from_response(ai_response)

        return ai_response

    def stream_chat(self, user_input: str):
        """
        Same as chat but streaming — generator returns chunks.
        """
        self.memory.add_user_message(user_input)
        system_msg = self._get_system_message(user_input)

        full_response = []
        for chunk in self.agent.stream(
            {"messages": [system_msg, HumanMessage(content=user_input)]},
            config=self._build_config(),
            stream_mode="values",
        ):
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                delta = last_msg.content
                full_response.append(delta)
                yield delta

        final = "".join(full_response)
        self.memory.add_ai_message(final)
        self._sync_mood_from_response(final)

    # ─────────────────────────────────────────
    def _sync_mood_from_response(self, response: str):
        """
        Attempts to detect mood from keywords in the response and updates state.
        Simple logic — can be replaced with a classifier.
        """
        mood_keywords = {
            "happy"     : ["happy", "glad", "yay", "😄", "😊", "haaha", "haha"],
            "sad"       : ["sad", "unfortunately", "sorry", "😢", "😞"],
            "surprised" : ["woah", "didn't expect", "amazing", "😮", "!!", "wow"],
            "excited"   : ["awesome", "great", "let's go", "🎉", "💥"],
            "angry"     : ["mad", "not like that", "😠"],
        }
        response_lower = response.lower()
        for mood, keywords in mood_keywords.items():
            if any(kw in response_lower for kw in keywords):
                self.memory.set_mood(mood)
                return
        self.memory.set_mood("neutral")

    # ─────────────────────────────────────────
    def end_session(self, summary: Optional[str] = None):
        """
        Ends the session and saves a summary.
        If no summary is provided, the LLM generates one.
        """
        if summary is None:
            summary = self._auto_summarize()
        mood = self.memory.get_mood()
        self.memory.close_session(summary, mood)
        print(f"[Session {self.session_id}] ended. Summary saved.")

    def _auto_summarize(self) -> str:
        """Requests a session summary from the LLM"""
        messages = self.memory.get_recent_messages()
        if not messages:
            return "Short session with no conversation."

        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
            for m in messages[-10:]
        )
        prompt = f"Summarize this conversation in 2-3 sentences:\n\n{history_text}"
        result = self.llm.invoke([HumanMessage(content=prompt)])
        return result.content

    # ─────────────────────────────────────────
    def update_persona(self, new_persona: str):
        self.persona = new_persona
        self.memory.set_persona(new_persona)

    def inject_memory(self, text: str, metadata: Optional[dict] = None):
        """Add info directly to long-term memory without conversation"""
        return self.memory.save_important(text, metadata)

    def recall(self, query: str, top_k: int = 4) -> list[str]:
        """Search long-term memory"""
        return self.memory.long_term.search(query, top_k)

    def get_history(self):
        return self.memory.get_recent_messages()