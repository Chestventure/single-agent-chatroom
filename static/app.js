(() => {
  const $ = (sel) => document.querySelector(sel);
  const chatEl = $("#chat");
  const inputEl = $("#input");
  const composerEl = $("#composer");
  const personaListEl = $("#personaList");
  const startBtn = $("#startBtn");
  const stopBtn = $("#stopBtn");
  const clearBtn = $("#clearBtn");
  const dayContextEl = $("#dayContext");
  const timerBarEl = $("#timerBar");
  const timerLabelEl = $("#timerLabel");
  const playerNameEl = $("#playerName");

  const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/night`;
  let ws = null;
  let running = false;
  let totalDur = 180;
  let timeLeft = 0;
  let personas = [];
  let label = "Night Time";
  let connected = false;

  function connect() {
    ws = new WebSocket(WS_URL);

    ws.addEventListener("open", () => {
      connected = true;
      console.log("WS connected");
      const saved = localStorage.getItem("night_day_context");
      const savedName = localStorage.getItem("night_player_name");
      if (saved) dayContextEl.value = saved;
      if (savedName) playerNameEl.value = savedName;
    });

    ws.addEventListener("message", (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "init") {
        personas = data.personas || [];
        label = data.label || label;
        running = data.running || false;
        timeLeft = data.time_left || 0;
        totalDur = timeLeft || 180;
        renderPersonas(personas);
        (data.history || []).forEach((m) => pushMessage(m));
        updateTimerUI();
      } else if (data.type === "chat") {
        pushMessage(data.message);
      } else if (data.type === "timer") {
        // totalDur sets on first tick if not set
        if (!totalDur) totalDur = data.time_left;
        timeLeft = data.time_left;
        updateTimerUI();
      } else if (data.type === "state") {
        running = data.running;
      } else if (data.type === "phase_end") {
        running = false;
        timeLeft = 0;
        updateTimerUI();
        pushSystem(`⏰ 타이머 종료 — 오늘의 Night Time이 끝났습니다.`);
      } else if (data.type === "context") {
        // updated day context (if changed mid-run)
      } else if (data.type === "cleared") {
        chatEl.innerHTML = "";
        pushSystem("대화 내역을 초기화했습니다.");
      } else if (data.type === "error") {
        alert(data.message);
      }
    });

    ws.addEventListener("close", () => {
      connected = false;
      console.log("WS disconnected; retrying in 1.5s");
      setTimeout(connect, 1500);
    });
  }

  connect();

  // --- UI helpers ---
  function pushSystem(text) {
    const msg = {
      role: "system",
      speaker: "System",
      ts: Date.now() / 1000,
      text
    };
    pushMessage(msg);
  }

  function pushMessage(m) {
    const line = document.createElement("div");
    const isUser = m.role === "user";
    line.className = "line" + (isUser ? " right" : "");

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const msg = document.createElement("p");
    msg.className = "msg";
    msg.textContent = m.text;
    const meta = document.createElement("div");
    meta.className = "meta";
    const d = new Date(m.ts * 1000);
    const speaker = m.speaker || (m.role === "npc" ? "NPC" : "나");
    meta.textContent = `${speaker} · ${d.toLocaleTimeString()}`;

    bubble.appendChild(msg);
    bubble.appendChild(meta);
    line.appendChild(bubble);
    chatEl.appendChild(line);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function renderPersonas(list) {
    personaListEl.innerHTML = "";
    list.forEach((p) => {
      const li = document.createElement("li");
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = p.name;
      const style = document.createElement("div");
      style.className = "style";
      style.textContent = p.style;
      const desc = document.createElement("div");
      desc.className = "desc";
      desc.textContent = p.description;
      li.appendChild(name);
      li.appendChild(style);
      li.appendChild(desc);
      personaListEl.appendChild(li);
    });
  }

  function updateTimerUI() {
    const t = Math.max(0, timeLeft);
    const mm = String(Math.floor(t / 60)).padStart(2, "0");
    const ss = String(t % 60).padStart(2, "0");
    timerLabelEl.textContent = `${mm}:${ss}`;
    const pct = totalDur > 0 ? Math.max(0, Math.min(1, t / totalDur)) : 0;
    timerBarEl.style.width = (pct * 100) + "%";
  }

  // --- Events ---
  composerEl.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!connected) return;
    const text = inputEl.value.trim();
    if (!text) return;
    const name = playerNameEl.value.trim() || "나";
    localStorage.setItem("night_player_name", name);
    ws.send(JSON.stringify({ type: "chat", speaker: name, text }));
    inputEl.value = "";
    inputEl.focus();
  });

  startBtn.addEventListener("click", () => {
    if (!connected) return;
    const ctx = dayContextEl.value.trim();
    localStorage.setItem("night_day_context", ctx);
    ws.send(JSON.stringify({ type: "control", payload: { action: "start", duration: 180, day_context: ctx } }));
  });

  stopBtn.addEventListener("click", () => {
    if (!connected) return;
    ws.send(JSON.stringify({ type: "control", payload: { action: "stop" } }));
  });

  clearBtn.addEventListener("click", () => {
    if (!connected) return;
    if (confirm("대화 내역을 모두 삭제할까요?")) {
      ws.send(JSON.stringify({ type: "clear" }));
    }
  });
})();
