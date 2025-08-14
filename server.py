#!/usr/bin/env python3
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---- Load .env for OPENAI_API_KEY / OPENAI_MODEL ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# LangChain (lazy init of client)
_llm_client = None
def get_llm():
    global _llm_client
    if _llm_client is None:
        from langchain_openai import ChatOpenAI
        _llm_client = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.7,
            api_key=OPENAI_API_KEY,
        )
    return _llm_client

# ----------------------------- Models -----------------------------
@dataclass
class Message:
    role: str  # "npc" | "user" | "system"
    speaker: str
    text: str
    ts: float

@dataclass
class Persona:
    key: str
    name: str
    style: str
    description: str

    def to_public(self) -> Dict:
        return {"key": self.key, "name": self.name, "style": self.style, "description": self.description}

# ------------------------- Demo Personas --------------------------
PERSONAS = [
    Persona(
        key="sangjae",
        name="금상재",
        style="거칠고 솔직한 말투, 슬랭·반말",
        description="겉은 거칠고 허세 있지만 의리는 있는 타입. 노가다 싫어하지만 친구한테는 의리파."
    ),
    Persona(
        key="sia",
        name="강시아",
        style="말수가 적고 간결, 간혹 시적인 표현",
        description="겉으론 무심하지만 관심 있는 것엔 섬세. 자연·풍경 얘기를 곧잘 함."
    ),
    Persona(
        key="sihyeon",
        name="모시현",
        style="정중하고 조리 있게, 이모지 절제",
        description="모범생. 책임감 강하고 담담. 가끔 귀여운 면모."
    ),
    Persona(
        key="inho",
        name="하인호",
        style="볓고 친화력 좋은 말투, 장난 적당히",
        description="분위기 메이커. 모두가 어색하면 먼저 말 거는 타입. 꿈 얘기 좋아함."
    ),
]

KEY2PERSONA = {p.key: p for p in PERSONAS}
NAME2PERSONA = {p.name: p for p in PERSONAS}
ORDER = ["sangjae", "sia", "sihyeon", "inho"]  # deterministic fallback order

# --------------------------- Utilities ----------------------------
def now_ts() -> float:
    return time.time()

def clean_text(s: str) -> str:
    return s.strip().replace("\\n", " ").strip()

def build_history_snippet(history: List[Message], limit: int = 14) -> str:
    recent = [m for m in history if m.role != "system"][-limit:]
    lines = []
    for m in recent:
        speaker = m.speaker or ("사용자" if m.role == "user" else "NPC")
        text = clean_text(m.text)
        if len(text) > 160:
            text = text[:160] + "…"
        lines.append(f"- {speaker}: {text}")
    return "\\n".join(lines) if lines else "(대화 시작 전)"

def next_round_robin(history: List[Message]) -> Persona:
    # choose next NPC deterministically based on last NPC in history
    last_npc = None
    for m in reversed(history):
        if m.role == "npc":
            last_npc = m.speaker
            break
    if last_npc and last_npc in NAME2PERSONA:
        last_key = NAME2PERSONA[last_npc].key
        idx = ORDER.index(last_key)
        next_key = ORDER[(idx + 1) % len(ORDER)]
    else:
        next_key = ORDER[0]
    return KEY2PERSONA[next_key]

# ---------------------- LLM-backed generation ---------------------
class NextTurn(BaseModel):
    speaker_key: Literal["sangjae","sia","sihyeon","inho"] = Field(..., description="다음 발화할 NPC의 key")
    text: str = Field(..., description="다음 차례에 해당 인물이 카톡에서 보낼 한 줄(90자 이내)")

async def generate_next_turn_llm(history: List[Message], day_context: str, room_label: str) -> NextTurn:
    # If no API key, use deterministic fallback
    if not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
        p = next_round_robin(history)
        return NextTurn(speaker_key=p.key, text=f"{day_context.split()[0] if day_context else '그거'} 내일 계속 하자")

    from langchain_core.prompts import ChatPromptTemplate
    llm = get_llm().with_structured_output(NextTurn)

    personas_text = "\\n".join([
        f"- {p.name} (key={p.key}) | 말투: {p.style} | 성격: {p.description}" for p in PERSONAS
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 고등학생 **그룹채팅방**의 **턴 디사이더**이자 **대사 작성자**다. "
         "이번 차례에 누가 말할지와 그 사람이 보낼 메시지를 **한 번에** 결정한다. "
         "반드시 아래 네 명 중 한 명만 선택하라:\n{personas_text}\n\n"
         "출력은 JSON으로 하고 스키마를 지켜라."),
        ("user",
         "대화방: {room_label}\n"
         "낮에 있었던 일 요약: {day_context}\n"
         "최근 대화:\n{history_snippet}\n\n"
         "규칙:\n"
         "1) speaker_key는 [sangjae, sia, sihyeon, inho] 중 하나만 사용.\n"
         "2) text는 **한국어 한 줄**, **90자 이내**, **자연스럽게**.\n"
         "3) 이름/말머리/따옴표/괄호/내레이션/해시태그 금지.\n"
         "4) 유해 표현 금지. 최근 맥락과 말투·성격을 반영.\n"
         "5) 플레이어(사용자)는 선택 대상이 아님.\n"
         "이제 다음 차례 한 턴을 산출하라.")
    ])

    try:
        result = await (prompt | llm).ainvoke({
            "room_label": room_label,
            "day_context": day_context or "(특이사항 없음)",
            "history_snippet": build_history_snippet(history),
            "personas_text": personas_text,
        })
        return result
    except Exception as e:
        # deterministic fallback (no randomness)
        p = next_round_robin(history)
        return NextTurn(speaker_key=p.key, text="그 이야기 내일 마저 하자")

# -------------------------- Room State ---------------------------
class RoomState:
    def __init__(self, label: str = "Night Time"):
        self.label = label
        self.day_context: str = ""
        self.history: List[Message] = []
        self.clients: Set[WebSocket] = set()
        self.running: bool = False
        self.chat_task: Optional[asyncio.Task] = None
        self.timer_task: Optional[asyncio.Task] = None
        self.time_left: int = 0  # seconds

    async def broadcast(self, payload: Dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def add_history(self, msg: Message):
        self.history.append(msg)
        if len(self.history) > 500:
            self.history = self.history[-500:]

    async def start(self, duration: int, day_context: str):
        if self.running:
            self.day_context = day_context
            self.time_left = duration
            await self.broadcast({"type": "timer", "time_left": self.time_left})
            return
        self.running = True
        self.day_context = day_context
        self.time_left = duration
        await self.broadcast({"type": "state", "running": self.running})
        self.chat_task = asyncio.create_task(self._chat_loop())
        self.timer_task = asyncio.create_task(self._timer_loop())

    async def stop(self):
        self.running = False
        await self.broadcast({"type": "state", "running": self.running})
        if self.chat_task:
            self.chat_task.cancel()
        if self.timer_task:
            self.timer_task.cancel()
        self.chat_task = None
        self.timer_task = None
        await self.broadcast({"type": "phase_end", "reason": "timer_end"})

    async def _timer_loop(self):
        try:
            while self.running and self.time_left > 0:
                await asyncio.sleep(1)
                self.time_left -= 1
                await self.broadcast({"type": "timer", "time_left": self.time_left})
            if self.running:
                await self.stop()
        except asyncio.CancelledError:
            pass

    async def _chat_loop(self):
        try:
            while self.running:
                delay = random.uniform(1.0, 3.0)  # UI 역동성
                await asyncio.sleep(delay)
                if not self.running:
                    break
                # LLM이 speaker + text를 **한 번에** 결정
                nt = await generate_next_turn_llm(self.history, self.day_context, self.label)
                p = KEY2PERSONA.get(nt.speaker_key, next_round_robin(self.history))
                text = clean_text(nt.text)[:110]
                msg = Message(role="npc", speaker=p.name, text=text, ts=time.time())
                self.add_history(msg)
                await self.broadcast({"type": "chat", "message": asdict(msg)})
        except asyncio.CancelledError:
            pass

ROOM = RoomState()

# ------------------------------ API ------------------------------
app = FastAPI(title="15일 반장 - Night Time Chat Demo (LLM speaker+text)", version="0.3.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))

class ControlPayload(BaseModel):
    action: str
    duration: Optional[int] = 180
    day_context: Optional[str] = ""

@app.websocket("/ws/night")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ROOM.clients.add(ws)
    try:
        await ws.send_text(json.dumps({
            "type": "init",
            "personas": [p.to_public() for p in PERSONAS],
            "history": [asdict(m) for m in ROOM.history],
            "running": ROOM.running,
            "time_left": ROOM.time_left,
            "label": ROOM.label,
            "day_context": ROOM.day_context,
        }, ensure_ascii=False))

        while True:
            text = await ws.receive_text()
            data = json.loads(text)

            if data.get("type") == "chat":
                speaker = data.get("speaker", "You")
                msg_text = clean_text(data.get("text", ""))[:500]
                if not msg_text:
                    continue
                msg = Message(role="user", speaker=speaker, text=msg_text, ts=time.time())
                ROOM.add_history(msg)
                await ROOM.broadcast({"type": "chat", "message": asdict(msg)})

            elif data.get("type") == "control":
                payload = ControlPayload(**data.get("payload", {}))
                if payload.action == "start":
                    await ROOM.start(duration=payload.duration or 180, day_context=payload.day_context or "")
                elif payload.action == "stop":
                    await ROOM.stop()
                elif payload.action == "context":
                    ROOM.day_context = payload.day_context or ""
                    await ROOM.broadcast({"type": "context", "day_context": ROOM.day_context})
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": f"unknown action: {payload.action}"}))

            elif data.get("type") == "clear":
                ROOM.history.clear()
                await ROOM.broadcast({"type": "cleared"})

            else:
                await ws.send_text(json.dumps({"type": "error", "message": "unknown message type"}))

    except WebSocketDisconnect:
        pass
    finally:
        ROOM.clients.discard(ws)

# Run tip: uvicorn server:app --reload --port 8000
