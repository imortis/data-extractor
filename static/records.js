const cardsEl = document.getElementById("cards");
const emptyEl = document.getElementById("empty");
const searchInput = document.getElementById("search-input");
const categoryFilter = document.getElementById("category-filter");
const tabsEl = document.getElementById("group-tabs");
const resultsLine = document.getElementById("results-line");
const detailEl = document.getElementById("detail");
const detailBody = document.getElementById("detail-body");
const importInput = document.getElementById("import-input");

const GROUP_LABELS = { all: "All records", student: "Students", faculty: "Faculty", other: "Other" };
const STATUS_LABELS = { verified: "verified", unchecked: "unchecked", conflict: "conflict", source_inconsistent: "source inconsistent" };

const params = new URLSearchParams(location.search);
const state = { q: params.get("q") || "", group: params.get("group") || "all", category: params.get("category") || "" };
searchInput.value = state.q;

let requestNumber = 0;
let lastFocused = null;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

/** Appends text with the search terms highlighted, using text nodes so nothing is parsed as HTML. */
function appendHighlighted(parent, text, terms) {
  const usable = terms.filter(Boolean).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!usable.length) {
    parent.textContent = text;
    return;
  }
  const pattern = new RegExp(`(${usable.join("|")})`, "gi");
  String(text).split(pattern).forEach((part, i) => {
    parent.appendChild(i % 2 ? el("mark", "", part) : document.createTextNode(part));
  });
}

function formatDate(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function pluralize(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

function fileLink(record, kind, label) {
  const link = el("a", "file-link", label);
  link.href = `/api/records/${record.id}/file/${kind}`;
  link.setAttribute("aria-label", `Download ${label} for ${record.title}`);
  return link;
}

function renderTrust(trust) {
  const wrap = el("div");
  if (!trust || !trust.total) {
    wrap.appendChild(el("div", "trust-note", "Not verified"));
    return wrap;
  }
  const bar = el("div", "trust-bar");
  Object.keys(STATUS_LABELS).forEach((status) => {
    if (!trust[status]) return;
    const part = el("span", `bar-${status}`);
    part.style.flexGrow = trust[status];
    part.title = `${trust[status]} ${STATUS_LABELS[status]}`;
    bar.appendChild(part);
  });
  wrap.appendChild(bar);
  const flagged = (trust.conflict || 0) + (trust.source_inconsistent || 0);
  const note = `${trust.verified || 0} of ${trust.total} values verified` + (flagged ? ` · ${flagged} to review` : "");
  const noteEl = el("div", "trust-note", note);
  noteEl.style.marginTop = "5px";
  wrap.appendChild(noteEl);
  return wrap;
}

function renderCard(record, terms) {
  const card = el("article", `record-card group-${record.group}`);

  const top = el("div", "card-top");
  top.appendChild(el("div", "card-avatar", (record.title.trim()[0] || "?").toUpperCase()));
  const heading = el("div", "card-heading");
  const title = el("button", "card-title");
  title.type = "button";
  appendHighlighted(title, record.title, terms);
  title.addEventListener("click", () => openDetail(record.id, title));
  heading.appendChild(title);
  if (record.topic) {
    const topic = el("div", "card-topic");
    appendHighlighted(topic, record.topic, terms);
    heading.appendChild(topic);
  }
  top.appendChild(heading);
  card.appendChild(top);

  const chips = el("div", "card-chips");
  chips.appendChild(el("span", "chip", GROUP_LABELS[record.group] || record.group));
  chips.appendChild(el("span", "chip", record.category.replace(/_/g, " ")));
  if (record.table_count) {
    chips.appendChild(el("span", "chip", `${pluralize(record.table_count, "table")}, ${pluralize(record.row_count, "row")}`));
  }
  if (record.warnings) chips.appendChild(el("span", "chip chip-warning", pluralize(record.warnings, "warning")));
  card.appendChild(chips);

  if (record.snippets.length) {
    const snippets = el("div", "card-snippets");
    record.snippets.forEach((snippet) => {
      const box = el("div", "snippet");
      box.appendChild(el("span", "snippet-where", snippet.where));
      const body = el("span");
      appendHighlighted(body, snippet.text, terms);
      box.appendChild(body);
      snippets.appendChild(box);
    });
    card.appendChild(snippets);
  } else if (record.summary) {
    card.appendChild(el("p", "card-summary", record.summary));
  }

  card.appendChild(renderTrust(record.trust));

  const foot = el("div", "card-foot");
  foot.appendChild(el("span", "card-date", formatDate(record.saved_at)));
  const files = el("div", "card-files");
  if (record.has_excel) files.appendChild(fileLink(record, "xlsx", "Excel"));
  files.appendChild(fileLink(record, "json", "JSON"));
  if (record.sources.length) files.appendChild(fileLink(record, "source1", "Original"));
  foot.appendChild(files);
  card.appendChild(foot);
  return card;
}

function renderTabs(counts) {
  tabsEl.innerHTML = "";
  Object.keys(GROUP_LABELS).forEach((group) => {
    const tab = el("button", "group-tab", GROUP_LABELS[group]);
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(state.group === group));
    tab.appendChild(el("span", "count", counts[group] || 0));
    tab.addEventListener("click", () => {
      state.group = group;
      load();
    });
    tabsEl.appendChild(tab);
  });
}

function renderCategories(categories) {
  const current = state.category;
  categoryFilter.innerHTML = "";
  const any = el("option", "", "All document types");
  any.value = "";
  categoryFilter.appendChild(any);
  categories.forEach((category) => {
    const option = el("option", "", category.replace(/_/g, " "));
    option.value = category;
    categoryFilter.appendChild(option);
  });
  categoryFilter.value = categories.includes(current) ? current : "";
}

function renderEmpty(data) {
  emptyEl.innerHTML = "";
  emptyEl.classList.remove("hidden");
  if (!data.counts.all) {
    emptyEl.appendChild(el("strong", "", "No records yet"));
    emptyEl.appendChild(document.createTextNode("Everything you extract is saved here automatically. You can also import workbooks you already have."));
  } else {
    emptyEl.appendChild(el("strong", "", "Nothing matches"));
    emptyEl.appendChild(document.createTextNode("Try fewer words, or clear the filters."));
  }
}

async function load() {
  const thisRequest = ++requestNumber;
  const query = new URLSearchParams();
  if (state.q) query.set("q", state.q);
  if (state.group !== "all") query.set("group", state.group);
  if (state.category) query.set("category", state.category);
  history.replaceState(null, "", query.toString() ? `/records?${query}` : "/records");

  let data;
  try {
    const response = await fetch(`/api/records?${query}`);
    data = await response.json();
  } catch (error) {
    resultsLine.textContent = `Couldn't load records: ${error.message}`;
    return;
  }
  if (thisRequest !== requestNumber) return; // a newer search has already started

  const terms = state.q.trim().split(/\s+/).filter(Boolean);
  renderTabs(data.counts);
  renderCategories(data.categories);
  cardsEl.innerHTML = "";
  emptyEl.classList.add("hidden");
  data.records.forEach((record) => cardsEl.appendChild(renderCard(record, terms)));
  if (!data.records.length) renderEmpty(data);
  resultsLine.textContent = state.q ? `${pluralize(data.records.length, "record")} matching “${state.q}”` : pluralize(data.records.length, "record");
}

// ---- detail view ----

function renderKeyValues(entries, fields) {
  const wrap = el("div", "detail-table-wrap");
  const table = el("table", "extracted-table");
  const body = el("tbody");
  entries.forEach(([key, value, path]) => {
    const row = el("tr");
    row.appendChild(el("td", "", key.replace(/_/g, " ")));
    const cell = el("td", "", value === null || value === undefined ? "" : String(value));
    const check = fields && fields[path];
    if (check) {
      cell.classList.add(`trust-${check.status.replace(/_/g, "-")}`);
      if (check.evidence && check.evidence.length) cell.title = check.evidence.join("\n");
    }
    row.appendChild(cell);
    body.appendChild(row);
  });
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}

function renderTable(tableData, check) {
  const wrap = el("div", "detail-table-wrap");
  const table = el("table", "extracted-table");
  const head = el("thead");
  const headRow = el("tr");
  (tableData.headers || []).forEach((header, column) => {
    const th = el("th", "", header);
    const headerCheck = check && check.headers && check.headers[column];
    if (headerCheck && headerCheck.status === "conflict") {
      th.classList.add("trust-header-conflict");
      th.title = headerCheck.evidence.join("\n");
    }
    headRow.appendChild(th);
  });
  head.appendChild(headRow);
  table.appendChild(head);
  const body = el("tbody");
  (tableData.rows || []).forEach((rowData, rowIndex) => {
    const row = el("tr");
    (Array.isArray(rowData) ? rowData : [rowData]).forEach((value, column) => {
      const cell = el("td", "", value === null || value === undefined ? "" : String(value));
      const cellCheck = check && check.cells && check.cells[rowIndex] && check.cells[rowIndex][column];
      if (cellCheck) {
        cell.classList.add(`trust-${cellCheck.status.replace(/_/g, "-")}`);
        if (cellCheck.evidence && cellCheck.evidence.length) cell.title = cellCheck.evidence.join("\n");
      }
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}

function section(title, content) {
  const box = el("section", "detail-section");
  box.appendChild(el("h3", "", title));
  box.appendChild(content);
  return box;
}

async function openDetail(id, trigger) {
  lastFocused = trigger || null;
  detailBody.innerHTML = "";
  detailBody.appendChild(el("p", "", "Loading…"));
  detailEl.classList.remove("hidden");

  let data;
  try {
    data = await (await fetch(`/api/records/${id}`)).json();
  } catch (error) {
    data = { ok: false, error: error.message };
  }
  detailBody.innerHTML = "";
  if (!data.ok) {
    detailBody.appendChild(el("p", "error-text", data.error || "Couldn't open this record."));
    return;
  }

  const { record, parsed } = data;
  const verification = parsed.verification || {};
  document.getElementById("detail-title").textContent = record.title;
  document.getElementById("detail-sub").textContent =
    [record.topic, record.category.replace(/_/g, " "), formatDate(record.saved_at)].filter(Boolean).join(" · ");

  const actions = el("div", "detail-actions");
  if (record.has_excel) actions.appendChild(fileLink(record, "xlsx", "Excel"));
  actions.appendChild(fileLink(record, "json", "JSON"));
  record.sources.forEach((source, i) => {
    actions.appendChild(fileLink(record, `source${i + 1}`, record.sources.length > 1 ? `Original ${i + 1}` : "Original"));
  });
  const groupLabel = el("label", "", "Group");
  groupLabel.htmlFor = "detail-group";
  actions.appendChild(groupLabel);
  const groupSelect = el("select");
  groupSelect.id = "detail-group";
  ["student", "faculty", "other"].forEach((group) => {
    const option = el("option", "", GROUP_LABELS[group]);
    option.value = group;
    groupSelect.appendChild(option);
  });
  groupSelect.value = record.group;
  groupSelect.addEventListener("change", async () => {
    await fetch(`/api/records/${id}/group`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group: groupSelect.value }),
    });
    load();
  });
  actions.appendChild(groupSelect);
  detailBody.appendChild(actions);

  if (Array.isArray(parsed.extraction_warnings) && parsed.extraction_warnings.length) {
    const list = el("ul", "detail-warnings");
    parsed.extraction_warnings.forEach((warning) => list.appendChild(el("li", "", warning)));
    detailBody.appendChild(list);
  }
  if (parsed.summary) detailBody.appendChild(section("Summary", el("p", "", parsed.summary)));

  const fields = verification.fields || {};
  const info = parsed.key_info && typeof parsed.key_info === "object" ? Object.entries(parsed.key_info) : [];
  if (info.length) {
    detailBody.appendChild(section("Key details", renderKeyValues(info.map(([k, v]) => [k, v, `key_info.${k}`]), fields)));
  }
  (parsed.tables || []).forEach((table, i) => {
    detailBody.appendChild(section(table.title || `Table ${i + 1}`, renderTable(table, (verification.tables || [])[i])));
  });
  (parsed.sections || []).forEach((sectionData) => {
    const items = (sectionData.items || []).map((item) => (item && typeof item === "object" ? item : { value: item }));
    const headers = [...new Set(items.flatMap((item) => Object.keys(item)))];
    if (!items.length) return;
    const table = {
      headers: headers.map((h) => h.replace(/_/g, " ")),
      rows: items.map((item) => headers.map((h) => item[h] ?? "")),
    };
    detailBody.appendChild(section(sectionData.title || "Section", renderTable(table)));
  });
  document.getElementById("detail-close").focus();
}

function closeDetail() {
  detailEl.classList.add("hidden");
  if (lastFocused) lastFocused.focus();
}

// ---- controls ----

let searchTimer = null;
searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.q = searchInput.value.trim();
    load();
  }, 160);
});
categoryFilter.addEventListener("change", () => {
  state.category = categoryFilter.value;
  load();
});
document.getElementById("detail-close").addEventListener("click", closeDetail);
detailEl.addEventListener("click", (event) => {
  if (event.target === detailEl) closeDetail();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !detailEl.classList.contains("hidden")) closeDetail();
  if (event.key === "/" && document.activeElement !== searchInput && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    event.preventDefault();
    searchInput.focus();
  }
});

document.getElementById("import-btn").addEventListener("click", () => importInput.click());
importInput.addEventListener("change", async () => {
  const files = [...importInput.files];
  importInput.value = "";
  const failures = [];
  for (const [index, file] of files.entries()) {
    resultsLine.textContent = `Importing ${index + 1} of ${files.length}…`;
    const form = new FormData();
    form.append("file", file);
    try {
      const result = await (await fetch("/api/records/import", { method: "POST", body: form })).json();
      if (!result.ok) failures.push(result.error);
    } catch (error) {
      failures.push(`${file.name}: ${error.message}`);
    }
  }
  await load();
  if (failures.length) resultsLine.textContent = `Import problems: ${failures.join("; ")}`;
});

load();
const openId = params.get("open");
if (openId) openDetail(openId);
