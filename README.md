# 15일 반장 · Night Time 채팅 웹 데모

멀티 에이전트(4명 NPC)가 **1~3초 랜덤 지연**으로 번갈아가며 채팅을 생성하고,
플레이어가 실시간으로 참여할 수 있는 **3분 타임어택** 채팅방 데모입니다.

> 이 데모는 로컬 규칙 기반 "LLM 스텁"으로 NPC 대사를 만듭니다. 실제 LLM API로 바꾸도록 설계되어 있습니다.

## 미리보기
- 좌측: 인물 소개, 낮(Context) 입력, 시작/중지/초기화
- 우측: 실시간 채팅, 플레이어 입력창
- 상단 바: 3분 타이머 + 진행 바

## 실행 방법

### 1) 요구 사항
- Python 3.10+
- 가상환경 권장

### 2) 설치 & 실행
```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```
브라우저에서 `http://127.0.0.1:8000` 접속

## 사용법
1. 좌측 패널의 **낮에 있었던 일(Context)** 텍스트 영역에 오늘 낮의 사건을 적습니다.
2. **시작(3분)** 버튼을 누르면 1~3초 간격으로 NPC가 말하기 시작합니다.
3. 플레이어도 하단 입력창으로 자유롭게 대화에 참여할 수 있습니다.
4. 3분이 지나면 타이머가 종료되고 오늘의 Night Time이 끝납니다.

## 설계 개요

### 구조
```
server.py         # FastAPI + WebSocket 서버
static/index.html # UI
static/styles.css # UI 스타일
static/app.js     # 클라이언트 로직 (WebSocket, UI 렌더)
```

### WebSocket 프로토콜
- 클라이언트 → 서버
  - `{"type":"chat","speaker":"나","text":"메시지 내용"}`
  - `{"type":"control","payload":{"action":"start","duration":180,"day_context":"…"}}`
  - `{"type":"control","payload":{"action":"stop"}}`
  - `{"type":"clear"}`

- 서버 → 클라이언트
  - `{"type":"init","personas":[…],"history":[…],"running":true,"time_left":172}`
  - `{"type":"chat","message":{role,speaker,text,ts}}`
  - `{"type":"timer","time_left":169}`
  - `{"type":"state","running":true|false}`
  - `{"type":"phase_end","reason":"timer_end"}`
  - `{"type":"context","day_context":"…"}`
  - `{"type":"cleared"}` / `{"type":"error","message":"…"}`

### LLM 스텁 교체하기
`server.py`의 `generate_npc_message(...)` 함수를 실제 LLM으로 교체하면 됩니다.
예시(의사 코드):
```python
def generate_npc_message(p, history, day_context, room_label):
    prompt = make_prompt(persona=p, history=history, context=day_context, room=room_label)
    # ex) OpenAI, Upstage 등 원하는 API 호출
    return call_llm(prompt)
```
- **발화자 선택 로직**: `choose_speaker`에서 최근 언급된 인물을 약하게 가중치로 올립니다.
- **1~3초 랜덤 지연**: `_chat_loop()`에서 `await asyncio.sleep(random.uniform(1.0, 3.0))`

### 멀티룸/멀티세션 확장
- `RoomState`를 여러 개 관리하도록 바꾸면 다양한 반/채팅방을 동시에 돌릴 수 있습니다.
- 현재 데모는 단일 룸(“Night Time”)만 존재합니다.

## 라이선스
MIT


## 🔌 실제 LLM 사용 설정


1) `.env.example`를 복사해 `.env` 생성 후 **OPENAI_API_KEY** 채워넣기
```bash
cp .env.example .env
# 편집기로 OPENAI_API_KEY 입력
```
2) (선택) `OPENAI_MODEL` 변경 가능: `gpt-4o-mini`(기본), `gpt-4o`, `o4-mini` 등
3) 서버 실행
```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```
4) 브라우저 접속: `http://127.0.0.1:8000`

### 프롬프트/체인 커스터마이즈
- `server.py`의 `generate_npc_message_llm(...)`에서 프롬프트/온도/토큰 제한 등을 조정하세요.
- 응답은 **한 줄**, **최대 90자**, **자연스러운 카톡체**를 강제합니다.
- API 오류·키 누락 시 자동으로 **스텁**으로 폴백합니다.


## 🤖 다음 발화자까지 LLM이 결정

- 더 이상 서버가 랜덤/규칙으로 발화자를 고르지 않습니다.
- LLM이 최근 대화/낮 컨텍스트/페르소나를 보고 **다음으로 말할 인물(speaker_key)** 과 **그 대사(text)** 를 **JSON**으로 한 번에 반환합니다.
- 실패 시에도 랜덤이 아니라 **라운드로빈(결정적 순서)** 으로 폴백합니다.
- 구현: `server.py`의 `generate_next_turn_llm(...)` + `NextTurn` 스키마(Pydantic). `with_structured_output` 사용.
