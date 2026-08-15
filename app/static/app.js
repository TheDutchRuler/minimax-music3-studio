/* MiniMax Music 3 Studio — front end */

const $ = (id) => document.getElementById(id);
const api = (p, o) => fetch(p, o).then(async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
});

const state = {
  mode: "simple",
  tracks: [],
  jobs: new Map(),
  current: null,     // track id
  queue: [],         // playback order (track ids)
  waveCache: new Map(),
  filter: "",
};

/* ---------------- helpers ---------------- */

const fmtTime = (s) => {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

// Deterministic cover art: same track always gets the same colours.
function hashOf(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return Math.abs(h);
}
function artStyle(id) {
  const h = hashOf(id);
  const a = h % 360, b = (a + 40 + (h >> 8) % 120) % 360;
  const ang = (h >> 4) % 360;
  return `background:linear-gradient(${ang}deg,hsl(${a} 72% 56%),hsl(${b} 68% 44%))`;
}

function toast(msg, isErr) {
  const el = document.createElement("div");
  el.className = "toast" + (isErr ? " err" : "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), isErr ? 7000 : 3800);
}

const ICON = {
  play: `<svg viewBox="0 0 24 24"><path d="M8 5l11 7-11 7z"/></svg>`,
  pause: `<svg viewBox="0 0 24 24"><path d="M9 5v14M15 5v14"/></svg>`,
  dl: `<svg viewBox="0 0 24 24"><path d="M12 3v12m0 0l4-4m-4 4l-4-4M4 21h16"/></svg>`,
  trash: `<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M5 7l1 13h12l1-13M9 7V4h6v3"/></svg>`,
  redo: `<svg viewBox="0 0 24 24"><path d="M21 12a9 9 0 11-3-6.7M21 4v5h-5"/></svg>`,
};

/* ---------------- composer ---------------- */

document.querySelectorAll(".mode-btn").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".mode-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.mode = b.dataset.mode;
    $("pane-simple").classList.toggle("hidden", state.mode !== "simple");
    $("pane-custom").classList.toggle("hidden", state.mode !== "custom");
  };
});

document.querySelectorAll(".nav-item").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".nav-item").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    const lib = b.dataset.view === "library";
    $("composer").classList.toggle("hidden", lib);
    $("wsTitle").textContent = lib ? "Library" : "Your songs";
  };
});

// Insert a section tag at the caret, on its own line.
$("tagRow").addEventListener("click", (e) => {
  const btn = e.target.closest(".tag");
  if (!btn) return;
  const ta = $("lyrics");
  const tag = btn.dataset.tag;
  const s = ta.selectionStart, v = ta.value;
  const before = v.slice(0, s), after = v.slice(ta.selectionEnd);
  const pre = before && !before.endsWith("\n") ? "\n" : "";
  const ins = `${pre}${tag}\n`;
  ta.value = before + ins + after;
  ta.selectionStart = ta.selectionEnd = (before + ins).length;
  ta.focus();
  updateCounts();
});

const instBtn = $("instrumental");
instBtn.onclick = () => {
  const on = instBtn.getAttribute("aria-checked") === "true";
  instBtn.setAttribute("aria-checked", String(!on));
  // Lyrics are meaningless in instrumental mode; grey them out.
  $("lyrics").disabled = !on;
  $("lyrics").style.opacity = !on ? ".45" : "1";
};

$("duration").oninput = (e) => { $("durationOut").textContent = fmtTime(+e.target.value); };

function updateCounts() {
  $("simpleCount").textContent = $("simplePrompt").value.length;
  $("lyricsCount").textContent = $("lyrics").value.length;
}
$("simplePrompt").addEventListener("input", updateCounts);
$("lyrics").addEventListener("input", updateCounts);
updateCounts();

$("createBtn").onclick = async () => {
  const instrumental = instBtn.getAttribute("aria-checked") === "true";
  let prompt, lyrics, title;

  if (state.mode === "simple") {
    prompt = $("simplePrompt").value.trim();
    lyrics = "";
    title = "";
    if (!prompt) return toast("Describe the song you want first.", true);
  } else {
    prompt = buildCaption();
    lyrics = instrumental ? "" : $("lyrics").value.trim();
    title = $("title").value.trim();
    if (!prompt && !lyrics) return toast("Add lyrics or a style description.", true);
  }

  const seedRaw = $("seed").value.trim();
  const stepsRaw = $("steps").value.trim();
  const body = {
    prompt, lyrics, title, instrumental,
    duration: +$("duration").value,
    count: +$("count").value,
    seed: seedRaw === "" ? null : +seedRaw,
    steps: stepsRaw === "" ? null : +stepsRaw,
  };

  const btn = $("createBtn");
  btn.disabled = true;
  try {
    const res = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    res.jobs.forEach((j) => state.jobs.set(j.id, j));
    render();
    $("footNote").textContent = `Queued ${res.jobs.length} version${res.jobs.length > 1 ? "s" : ""}.`;
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
  }
};

/* ---------------- rendering ---------------- */

function trackRow(t) {
  const el = document.createElement("div");
  el.className = "track" + (state.current === t.id ? " playing" : "");
  const isCur = state.current === t.id && !$("audio").paused;
  const bits = [];
  if (t.instrumental) bits.push(`<span class="badge inst">Instrumental</span>`);
  bits.push(`<span class="badge">seed ${t.seed}</span>`);
  if (t.render_seconds) {
    const shared = t.group_size > 1 ? ` (batch of ${t.group_size})` : "";
    bits.push(`<span class="badge">${Math.round(t.render_seconds)}s render${shared}</span>`);
  }

  el.innerHTML = `
    <div class="art" style="${artStyle(t.id)}">
      <div class="art-play">${isCur ? ICON.pause : ICON.play}</div>
    </div>
    <div class="tk-main">
      <div class="tk-title"></div>
      <div class="tk-sub"></div>
      <div class="tk-badges">${bits.join("")}</div>
    </div>
    <div class="tk-actions">
      <span class="tk-time">${fmtTime(t.duration)}</span>
      <button class="icon-btn" data-act="redo" title="Use these settings again">${ICON.redo}</button>
      <button class="icon-btn" data-act="dl" title="Download WAV">${ICON.dl}</button>
      <button class="icon-btn danger" data-act="del" title="Delete">${ICON.trash}</button>
    </div>`;

  // textContent so user-authored titles/prompts can never inject markup
  el.querySelector(".tk-title").textContent = t.title;
  el.querySelector(".tk-sub").textContent = t.prompt || t.lyrics.split("\n")[0] || "—";

  el.querySelector(".art").onclick = () => togglePlay(t.id);
  el.querySelector(".tk-main").onclick = () => togglePlay(t.id);
  el.querySelector('[data-act="dl"]').onclick = (e) => { e.stopPropagation(); download(t); };
  el.querySelector('[data-act="del"]').onclick = async (e) => {
    e.stopPropagation();
    if (!confirm(`Delete "${t.title}"?`)) return;
    await api(`/api/track/${t.id}`, { method: "DELETE" }).catch((x) => toast(x.message, true));
    if (state.current === t.id) stopPlayback();
    loadLibrary();
  };
  el.querySelector('[data-act="redo"]').onclick = (e) => { e.stopPropagation(); reuse(t); };
  return el;
}

function jobRow(j) {
  const el = document.createElement("div");
  const failed = j.status === "error";
  el.className = "track pending" + (failed ? " failed" : "");
  const pct = Math.round((j.progress || 0) * 100);
  el.innerHTML = `
    <div class="art skeleton"></div>
    <div class="tk-main">
      <div class="tk-title"></div>
      ${failed
        ? `<div class="tk-err"></div>`
        : `<div class="prog-wrap"><div class="prog-bar" style="width:${pct}%"></div></div>
           <div class="tk-stage"><span class="st"></span><span>${pct}%</span></div>`}
    </div>
    <div class="tk-actions">
      ${j.status === "queued" ? `<button class="icon-btn danger" data-act="cancel" title="Cancel">${ICON.trash}</button>` : ""}
    </div>`;
  el.querySelector(".tk-title").textContent = j.title;
  if (failed) el.querySelector(".tk-err").textContent = j.error || "Generation failed";
  else el.querySelector(".st").textContent = j.stage || "Queued";

  const c = el.querySelector('[data-act="cancel"]');
  if (c) c.onclick = () => api(`/api/cancel/${j.id}`, { method: "POST" })
    .then(() => { state.jobs.delete(j.id); render(); })
    .catch((x) => toast(x.message, true));
  return el;
}

function render() {
  const list = $("trackList");
  list.innerHTML = "";

  const active = [...state.jobs.values()]
    .filter((j) => j.status !== "done")
    .sort((a, b) => b.created - a.created);

  const f = state.filter.toLowerCase();
  const tracks = f
    ? state.tracks.filter((t) =>
        (t.title + " " + t.prompt + " " + t.lyrics).toLowerCase().includes(f))
    : state.tracks;

  active.forEach((j) => list.appendChild(jobRow(j)));
  tracks.forEach((t) => list.appendChild(trackRow(t)));

  state.queue = tracks.map((t) => t.id);
  $("empty").classList.toggle("hidden", active.length > 0 || tracks.length > 0);
}

$("search").addEventListener("input", (e) => { state.filter = e.target.value; render(); });

// The model card's recommended three-part Structured Caption. Labelling each
// section explicitly steers it far better than one undifferentiated blob.
function buildCaption() {
  const parts = [
    ["", $("metaGlobal").value.trim()],
    ["Vocals: ", $("metaVocal").value.trim()],
    ["Arrangement: ", $("metaArrange").value.trim()],
  ];
  return parts
    .filter(([, v]) => v)
    .map(([label, v]) => label + v.replace(/\s*$/, "").replace(/([^.!?])$/, "$1."))
    .join(" ");
}

const EXAMPLE = {
  metaGlobal: "Genre: dreamy synthwave with shoegaze textures. BPM: 100. Key: A minor. Nostalgic and cinematic, building from sparse to euphoric. Wide analog production with tape saturation.",
  metaVocal: "Soft female lead, airy and slightly breathy, doubled an octave up in the chorus, generous plate reverb and a short slapback delay.",
  metaArrange: "Warm analog pads and arpeggiated synth bass throughout; gated reverb drums enter at the first chorus; a soaring lead synth takes the bridge; everything drops to pads and vocal in the outro.",
  lyrics: "[Intro]\n\n[Verse]\nNeon lines on an empty road\nRadio low, the city letting go\n\n[Chorus]\nDriving home at 3am\nI would do it all again\n\n[Bridge]\nHeadlights fading into blue\n\n[Outro]",
  title: "Neon Drive",
};

$("enhanceBtn").onclick = async () => {
  const brief = $("simplePrompt").value.trim();
  if (!brief) return toast("Describe the song first.", true);
  const btn = $("enhanceBtn");
  const label = btn.querySelector("span");
  btn.disabled = true;
  label.textContent = "Writing (the 8B is composing your brief)…";
  try {
    const r = await api("/api/enhance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: brief, instrumental: instBtn.getAttribute("aria-checked") === "true" }),
    });
    // The caption arrives as one paragraph; split it into the three fields at
    // the labels the writer was instructed to use.
    const cap = r.caption || "";
    const vocalIdx = cap.search(/Vocals?:/i);
    const arrIdx = cap.search(/Arrangement:/i);
    const g = vocalIdx > 0 ? cap.slice(0, vocalIdx).trim() : cap;
    const v = vocalIdx > 0 ? cap.slice(vocalIdx, arrIdx > vocalIdx ? arrIdx : undefined).replace(/^Vocals?:\s*/i, "").trim() : "";
    const a = arrIdx > 0 ? cap.slice(arrIdx).replace(/^Arrangement:\s*/i, "").trim() : "";
    $("metaGlobal").value = g;
    $("metaVocal").value = v;
    $("metaArrange").value = a;
    $("lyrics").value = r.lyrics || "";
    $("title").value = r.title || "";
    document.querySelector('.mode-btn[data-mode="custom"]').click();
    updateCounts();
    toast("Song written — review, tweak, then Create.");
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    label.textContent = "Write full song from this";
  }
};

$("loadExample").onclick = (e) => {
  e.preventDefault();
  Object.entries(EXAMPLE).forEach(([k, v]) => { const el = $(k); if (el) el.value = v; });
  updateCounts();
  toast("Example loaded — edit freely.");
};

function reuse(t) {
  document.querySelector('.mode-btn[data-mode="custom"]').click();
  $("metaGlobal").value = t.prompt || "";
  $("metaVocal").value = "";
  $("metaArrange").value = "";
  $("lyrics").value = t.lyrics || "";
  $("title").value = t.title || "";
  instBtn.setAttribute("aria-checked", String(!!t.instrumental));
  $("lyrics").disabled = !!t.instrumental;
  $("lyrics").style.opacity = t.instrumental ? ".45" : "1";
  if (t.duration) $("duration").value = Math.min(300, Math.max(20, Math.round(t.duration)));
  $("durationOut").textContent = fmtTime(+$("duration").value);
  updateCounts();
  $("composer").scrollTop = 0;
  toast("Settings loaded into the composer.");
}

function download(t) {
  const a = document.createElement("a");
  a.href = `/api/audio/${t.id}`;
  a.download = `${t.title}.wav`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ---------------- playback ---------------- */

const audio = $("audio");

function togglePlay(id) {
  if (state.current === id) {
    audio.paused ? audio.play() : audio.pause();
    return;
  }
  play(id);
}

function play(id) {
  const t = state.tracks.find((x) => x.id === id);
  if (!t) return;
  state.current = id;
  audio.src = `/api/audio/${id}`;
  audio.play().catch(() => {});
  $("player").classList.remove("hidden");
  $("npTitle").textContent = t.title;
  $("npSub").textContent = t.prompt || (t.instrumental ? "Instrumental" : "");
  $("npArt").setAttribute("style", artStyle(id));
  $("durTime").textContent = fmtTime(t.duration);
  drawWave(id);
  render();
}

function stopPlayback() {
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  state.current = null;
  $("player").classList.add("hidden");
  render();
}

$("playBtn").onclick = () => {
  if (!state.current) { if (state.queue.length) play(state.queue[0]); return; }
  audio.paused ? audio.play() : audio.pause();
};
$("nextBtn").onclick = () => step(1);
$("prevBtn").onclick = () => {
  if (audio.currentTime > 3) { audio.currentTime = 0; return; }
  step(-1);
};
function step(d) {
  const i = state.queue.indexOf(state.current);
  if (i < 0) return;
  const n = state.queue[(i + d + state.queue.length) % state.queue.length];
  if (n) play(n);
}

audio.addEventListener("play", () => { swapPlayIcon(true); render(); });
audio.addEventListener("pause", () => { swapPlayIcon(false); render(); });
audio.addEventListener("ended", () => step(1));
audio.addEventListener("loadedmetadata", () => { $("durTime").textContent = fmtTime(audio.duration); });
audio.addEventListener("timeupdate", () => {
  $("curTime").textContent = fmtTime(audio.currentTime);
  const p = audio.duration ? audio.currentTime / audio.duration : 0;
  $("waveCursor").style.left = `${p * 100}%`;
  paintWave(p);
});
function swapPlayIcon(playing) {
  $("playIcon").classList.toggle("hidden", playing);
  $("pauseIcon").classList.toggle("hidden", !playing);
}

$("volume").oninput = (e) => { audio.volume = +e.target.value; };
$("dlBtn").onclick = () => {
  const t = state.tracks.find((x) => x.id === state.current);
  if (t) download(t);
};
$("waveWrap").onclick = (e) => {
  if (!audio.duration) return;
  const r = $("waveWrap").getBoundingClientRect();
  audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
};

/* ---------------- waveform ---------------- */

const canvas = $("wave");
const ctx = canvas.getContext("2d");
let peaks = null;
let audioCtx = null;  // one shared context; browsers cap how many you may create

// Decode the WAV once per track and reduce it to per-bar peaks.
async function drawWave(id) {
  peaks = state.waveCache.get(id) || null;
  paintWave(0);
  if (peaks) return;
  try {
    const buf = await fetch(`/api/audio/${id}`).then((r) => r.arrayBuffer());
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!audioCtx) audioCtx = new AC();
    const decoded = await audioCtx.decodeAudioData(buf);
    const ch = decoded.getChannelData(0);
    const N = 240;
    const block = Math.floor(ch.length / N);
    const out = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let peak = 0;
      for (let j = 0; j < block; j += 8) {
        const v = Math.abs(ch[i * block + j]);
        if (v > peak) peak = v;
      }
      out[i] = peak;
    }
    const max = Math.max(...out) || 1;
    for (let i = 0; i < N; i++) out[i] /= max;
    state.waveCache.set(id, out);
    if (state.current === id) { peaks = out; paintWave(audio.duration ? audio.currentTime / audio.duration : 0); }
  } catch {
    /* Decoding is cosmetic — the seek bar still works without it. */
  }
}

function paintWave(progress) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const N = peaks ? peaks.length : 120;
  const bw = w / N;
  for (let i = 0; i < N; i++) {
    const v = peaks ? peaks[i] : 0.18;
    const bh = Math.max(2, v * h * 0.92);
    ctx.fillStyle = i / N <= progress ? "#f5c518" : "#33333f";
    ctx.fillRect(i * bw, (h - bh) / 2, Math.max(1, bw - 1.4), bh);
  }
}
window.addEventListener("resize", () => paintWave(audio.duration ? audio.currentTime / audio.duration : 0));

/* ---------------- data + events ---------------- */

async function loadLibrary() {
  try {
    const r = await api("/api/library");
    state.tracks = r.tracks;
    render();
  } catch (e) { /* server not up yet */ }
}

async function loadStatus() {
  try {
    const s = await api("/api/status");
    $("gpuName").textContent = s.gpu || "No CUDA GPU";
    if (s.vram) $("gpuVram").textContent = `${s.vram.used_gb} / ${s.vram.total_gb} GB used`;
    const dot = $("modelDot");
    dot.className = "dot " + (s.model_state === "ready" ? "ready" : s.model_state === "loading" ? "loading" : s.model_state === "error" ? "error" : "");
    $("gpuChip").title =
      `Model: ${s.model_state}${s.model_error ? " — " + s.model_error : ""}` +
      `\nOffload: ${s.offload}\nTurbo: ${s.turbo ? "on (compiled AR + batched CFG)" : "off"}`;
    $("duration").max = Math.round(s.max_duration || 300);
    // Re-sync any jobs the SSE stream may have missed (e.g. page reload).
    s.active.forEach((j) => state.jobs.set(j.id, j));
    if (s.active.length) render();
  } catch (e) { /* ignore */ }
}

function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => {
    const evt = JSON.parse(m.data);
    if (evt.type === "job") {
      const j = evt.job;
      if (j.status === "done") {
        state.jobs.delete(j.id);
        loadLibrary();
      } else {
        state.jobs.set(j.id, j);
        if (j.status === "error" && j.error !== "Cancelled") toast(`${j.title}: ${j.error}`, true);
        render();
      }
    } else if (evt.type === "library") {
      loadLibrary();
    } else if (evt.type === "model") {
      loadStatus();
      if (evt.state === "loading") $("footNote").textContent = "Loading model into VRAM…";
      if (evt.state === "ready") $("footNote").textContent = "Model ready.";
      if (evt.state === "error") toast("Model failed to load: " + (evt.error || ""), true);
    }
  };
  es.onerror = () => {
    es.close();
    setTimeout(connectEvents, 3000);  // server restarted or dropped; retry
  };
}

loadLibrary();
loadStatus();
connectEvents();
setInterval(loadStatus, 5000);
