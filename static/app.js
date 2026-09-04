const chatEl = document.getElementById("chat");
const emptyState = document.getElementById("empty-state");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");
const attachBtn = document.getElementById("attach-btn");
const fileInput = document.getElementById("file-input");
const attachmentsRow = document.getElementById("attachments-row");
const settingsBtn = document.getElementById("settings-btn");
const settingsPanel = document.getElementById("settings-panel");
const hostInput = document.getElementById("host-input");
const modelInput = document.getElementById("model-input");
const saveSettingsBtn = document.getElementById("save-settings-btn");
const testConnBtn = document.getElementById("test-conn-btn");
const statusPill = document.getElementById("status-pill");
const dropOverlay = document.getElementById("drop-overlay");

const DEFAULT_HOST = "https://awkward-scion-passover.ngrok-free.dev";
const DEFAULT_MODEL = "qwen3.8:27b";

const FILE_ICONS = {
  pdf: "📄", docx: "📝", doc: "📝", xlsx: "📊", xlsm: "📊", xls: "📊",
  pptx: "📈", ppt: "📈", csv: "🧾", txt: "🗒️", md: "🗒️", json: "🧩", log: "🗒️",
};

let pendingAttachments = []; // [{file_id, filename, kind, ext, size, data_url?, char_count?}]
let history = [];

function ext(name) {
  return (name.includes(".") ? name.split(".").pop() : "").toLowerCase();
}

function fmtSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function getSettings() {
  return {
    host: localStorage.getItem("de_host") || DEFAULT_HOST,
    model: localStorage.getItem("de_model") || DEFAULT_MODEL,
  };
}

function initSettingsForm() {
  const s = getSettings();
  hostInput.value = s.host;
  modelInput.value = s.model;
}
initSettingsForm();

settingsBtn.addEventListener("click", () => settingsPanel.classList.toggle("hidden"));

saveSettingsBtn.addEventListener("click", () => {
  localStorage.setItem("de_host", hostInput.value.trim() || DEFAULT_HOST);
  localStorage.setItem("de_model", modelInput.value.trim() || DEFAULT_MODEL);
  settingsPanel.classList.add("hidden");
  checkHealth();
});

async function checkHealth() {
  statusPill.innerHTML = `<span class="dot-pulse"></span>checking…`;
  statusPill.className = "status-pill status-unknown";
  const s = getSettings();
  try {
    const res = await fetch("/api/health", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(s),
    });
    const data = await res.json();
    if (data.ok) {
      statusPill.innerHTML = `<span class="dot-pulse"></span>online · ${s.model}`;
      statusPill.className = "status-pill status-ok";
    } else {
      statusPill.innerHTML = `<span class="dot-pulse"></span>offline`;
      statusPill.className = "status-pill status-bad";
    }
  } catch (e) {
    statusPill.innerHTML = `<span class="dot-pulse"></span>offline`;
    statusPill.className = "status-pill status-bad";
  }
}
testConnBtn.addEventListener("click", checkHealth);
checkHealth();

// ---- file attach: picker + drag/drop ----

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  uploadFiles([...fileInput.files]);
  fileInput.value = "";
});

let dragCounter = 0;
window.addEventListener("dragenter", (e) => {
  if (!e.dataTransfer.types.includes("Files")) return;
  dragCounter++;
  dropOverlay.classList.remove("hidden");
});
window.addEventListener("dragleave", () => {
  dragCounter = Math.max(0, dragCounter - 1);
  if (dragCounter === 0) dropOverlay.classList.add("hidden");
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropOverlay.classList.add("hidden");
  if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);
});

function renderAttachments() {
  attachmentsRow.innerHTML = "";
  if (!pendingAttachments.length) {
    attachmentsRow.classList.add("hidden");
    return;
  }
  attachmentsRow.classList.remove("hidden");
  pendingAttachments.forEach((att) => {
    const chip = document.createElement("div");
    chip.className = "attach-chip";
    if (att.kind === "image" && att.data_url) {
      chip.innerHTML = `<img class="thumb" src="${att.data_url}" />`;
    } else {
      chip.innerHTML = `<div class="file-icon-box">${FILE_ICONS[att.ext] || "📁"}</div>`;
    }
    const meta = document.createElement("div");
    meta.className = "attach-meta";
    meta.innerHTML = `<span class="attach-name">${escapeHtml(att.filename)}</span><span class="attach-size">${att.pending ? "uploading…" : fmtSize(att.size || 0)}</span>`;
    chip.appendChild(meta);
    const rm = document.createElement("button");
    rm.className = "remove-chip";
    rm.textContent = "✕";
    rm.addEventListener("click", () => {
      pendingAttachments = pendingAttachments.filter((a) => a !== att);
      renderAttachments();
    });
    chip.appendChild(rm);
    attachmentsRow.appendChild(chip);
  });
}

async function uploadFiles(files) {
  for (const file of files) {
    const placeholder = { filename: file.name, ext: ext(file.name), size: file.size, pending: true };
    pendingAttachments.push(placeholder);
    renderAttachments();

    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      const idx = pendingAttachments.indexOf(placeholder);
      if (!data.ok) {
        if (idx !== -1) pendingAttachments.splice(idx, 1);
        renderAttachments();
        alert(`Could not attach ${file.name}: ${data.error || "unknown error"}`);
        continue;
      }
      const resolved = {
        file_id: data.file_id,
        filename: data.filename,
        kind: data.kind,
        ext: ext(data.filename),
        size: data.size,
        data_url: data.data_url,
        char_count: data.char_count,
        pending: false,
      };
      if (idx !== -1) pendingAttachments[idx] = resolved;
      renderAttachments();
    } catch (e) {
      const idx = pendingAttachments.indexOf(placeholder);
      if (idx !== -1) pendingAttachments.splice(idx, 1);
      renderAttachments();
      alert(`Upload failed for ${file.name}: ${e.message}`);
    }
  }
}

// ---- composer ----

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
textInput.addEventListener("input", () => {
  textInput.style.height = "auto";
  textInput.style.height = Math.min(textInput.scrollHeight, 160) + "px";
});
sendBtn.addEventListener("click", sendMessage);

function clearEmptyState() {
  if (emptyState) emptyState.remove();
}

function attachmentChipHtml(att) {
  if (att.kind === "image" && att.data_url) {
    return `<img class="msg-image" src="${att.data_url}" />`;
  }
  return `<div class="file-chip"><span class="file-icon">${FILE_ICONS[att.ext] || "📁"}</span><div><div class="file-name">${escapeHtml(att.filename)}</div><div class="file-meta">${att.char_count ? att.char_count + " chars" : fmtSize(att.size || 0)}</div></div></div>`;
}

function addUserMessage(text, attachments) {
  clearEmptyState();
  const row = document.createElement("div");
  row.className = "msg-row user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (attachments.length) {
    const wrap = document.createElement("div");
    wrap.className = "msg-attachments";
    wrap.innerHTML = attachments.map(attachmentChipHtml).join("");
    bubble.appendChild(wrap);
  }
  if (text) {
    const p = document.createElement("div");
    p.textContent = text;
    bubble.appendChild(p);
  }
  row.appendChild(bubble);
  const avatar = document.createElement("div");
  avatar.className = "avatar user-avatar";
  avatar.textContent = "🧑";
  row.appendChild(avatar);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addThinkingBubble() {
  clearEmptyState();
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  const avatar = document.createElement("div");
  avatar.className = "avatar assistant-avatar";
  avatar.textContent = "✨";
  row.appendChild(avatar);
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<div class="thinking"><div class="spinner"></div>Analyzing… <span class="timer">0.0s</span></div>`;
  row.appendChild(bubble);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;

  const startedAt = performance.now();
  const timerEl = bubble.querySelector(".timer");
  const interval = setInterval(() => {
    timerEl.textContent = `${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
  }, 100);

  return { row, stopTimer: () => clearInterval(interval) };
}

function syntaxHighlight(json) {
  const str = JSON.stringify(json, null, 2)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return str.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false)\b|\bnull\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
    (match) => {
      let cls = "tok-number";
      if (/^"/.test(match)) {
        cls = /:$/.test(match) ? "tok-key" : "tok-string";
      } else if (/true|false/.test(match)) {
        cls = "tok-bool";
      } else if (/null/.test(match)) {
        cls = "tok-null";
      }
      return `<span class="${cls}">${match}</span>`;
    }
  );
}

function renderResult(row, data) {
  const bubble = row.querySelector(".bubble");
  bubble.innerHTML = "";

  if (!data.ok) {
    bubble.innerHTML = `<div class="error-text">Error: ${escapeHtml(data.error || "unknown error")}</div>`;
    return;
  }

  const card = document.createElement("div");
  card.className = "result-card";

  const header = document.createElement("div");
  header.className = "result-header";

  if (data.parsed && data.parsed.image_type) {
    const badge = document.createElement("span");
    badge.className = "type-badge";
    badge.textContent = data.parsed.image_type.replace(/_/g, " ");
    header.appendChild(badge);
  }
  if (data.parsed && typeof data.parsed.confidence === "number") {
    const conf = document.createElement("span");
    conf.className = "confidence-badge";
    conf.textContent = `${(data.parsed.confidence * 100).toFixed(0)}% confidence`;
    header.appendChild(conf);
  }
  if (typeof data.elapsed_seconds === "number") {
    const timing = document.createElement("span");
    timing.className = "timing-badge";
    timing.textContent = `⏱ ${data.elapsed_seconds}s`;
    header.appendChild(timing);
  }
  if (data.tools_used && data.tools_used.length) {
    const tool = document.createElement("span");
    tool.className = "tool-badge";
    const counts = {};
    data.tools_used.forEach((t) => (counts[t.name] = (counts[t.name] || 0) + 1));
    tool.textContent = `🔧 ${Object.entries(counts).map(([n, c]) => `${n} ×${c}`).join(", ")}`;
    header.appendChild(tool);
  }
  card.appendChild(header);

  if (data.parsed) {
    if (data.parsed.summary) {
      const summary = document.createElement("div");
      summary.className = "summary-line";
      summary.textContent = data.parsed.summary;
      card.appendChild(summary);
    }

    const toolbar = document.createElement("div");
    toolbar.className = "json-toolbar";
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "Copy JSON";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(JSON.stringify(data.parsed, null, 2));
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "Copy JSON"), 1200);
    });
    toolbar.appendChild(copyBtn);
    card.appendChild(toolbar);

    const pre = document.createElement("pre");
    pre.className = "json-view";
    pre.innerHTML = syntaxHighlight(data.parsed);
    card.appendChild(pre);
  } else {
    const warn = document.createElement("div");
    warn.className = "error-text";
    warn.textContent = data.parse_error
      ? `Model reply wasn't valid JSON (${data.parse_error}). Raw reply below:`
      : "No structured data returned. Raw reply below:";
    card.appendChild(warn);
    const pre = document.createElement("pre");
    pre.className = "json-view";
    pre.textContent = data.raw || "(empty response)";
    card.appendChild(pre);
  }

  bubble.appendChild(card);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

async function sendMessage() {
  const text = textInput.value.trim();
  const attachments = pendingAttachments.filter((a) => !a.pending);
  if (!text && !attachments.length) return;
  if (pendingAttachments.some((a) => a.pending)) return; // wait for uploads

  addUserMessage(text, attachments);
  textInput.value = "";
  textInput.style.height = "auto";
  pendingAttachments = [];
  renderAttachments();
  sendBtn.disabled = true;

  const { row: thinkingRow, stopTimer } = addThinkingBubble();
  const s = getSettings();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text || "Analyze the attached content.",
        attachments: attachments.map((a) => ({ file_id: a.file_id, filename: a.filename })),
        history,
        host: s.host,
        model: s.model,
      }),
    });
    const data = await res.json();
    stopTimer();
    renderResult(thinkingRow, data);

    history.push({ role: "user", content: text || "Analyze the attached content." });
    if (data.ok) {
      history.push({ role: "assistant", content: data.raw || JSON.stringify(data.parsed) });
    }
  } catch (e) {
    stopTimer();
    renderResult(thinkingRow, { ok: false, error: e.message });
  } finally {
    sendBtn.disabled = false;
  }
}
