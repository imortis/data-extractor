"""
Backend for the image / document data-extraction chat UI.

- Accepts images (vision, via /api/chat "images") and documents (PDF, DOCX,
  XLSX, PPTX, CSV, TXT, MD, JSON) which are text-extracted server-side.
- Extracted content is inlined into the prompt AND made available to the
  model as a real tool ("read_file") it can call for the full/partial text
  of any attachment (useful when content was truncated for size).
- PDFs are ALSO rendered to page images (via PyMuPDF) and sent to the vision
  model alongside any extracted text — this is what makes scanned PDFs (no
  text layer at all) work, the same way ChatGPT/Gemini handle them: by
  actually looking at the page, not just reading a text layer that may not
  exist. A second tool ("get_pdf_page_image") lets the model request any
  additional page beyond the ones auto-attached.
- DOCX/XLSX/PPTX are OOXML zip packages, so any embedded pictures (photos,
  screenshots, scanned inserts, slide backgrounds) are pulled straight out
  of the zip's media folder and sent to the vision model too, not just the
  text layer.
- Times the full round trip (including any tool-call loop) and returns it.
"""

import base64
import csv
import datetime
import io
import json
import os
import re
import time
import uuid
import zipfile

import requests
from flask import Flask, jsonify, request, send_from_directory

# --- MongoDB Atlas setup ---
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed, rely on real env vars

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    _MONGO_URI = os.environ.get("MONGO_URI", "")
    _MONGO_DB  = os.environ.get("MONGO_DB", "data_extractor")
    if _MONGO_URI:
        _mongo_client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=5000)
        _mongo_db = _mongo_client[_MONGO_DB]
        print(f"[MongoDB] Connected to db '{_MONGO_DB}' on Atlas")
    else:
        _mongo_client = None
        _mongo_db = None
        print("[MongoDB] No MONGO_URI set — Atlas saving disabled")
except Exception as _mongo_err:
    _mongo_client = None
    _mongo_db = None
    print(f"[MongoDB] Init error: {_mongo_err}")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30MB per upload

DEFAULT_HOST = "https://awkward-scion-passover.ngrok-free.dev"
DEFAULT_MODEL = "qwen3.8:27b"
REQUEST_TIMEOUT = 300
INLINE_CHAR_LIMIT = 12000  # per-file chars inlined directly into the prompt
TOOL_CHUNK_CHARS = 8000  # per-call chunk size the read_file tool returns
MAX_TOOL_ITERATIONS = 6  # a bit higher now: get_pdf_page_image calls eat iterations too
MAX_AUTO_RENDER_PDF_PAGES = 6  # pages auto-rendered to images on upload, no tool call needed
PDF_RENDER_DPI = 150
MAX_EMBEDDED_IMAGES = 6  # per docx/xlsx/pptx file

IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
OOXML_EXTS = {"docx", "xlsx", "xlsm", "pptx"}
DOC_EXTRACTORS = {}  # populated below

# In-memory store for this session's uploaded files: file_id -> dict
FILES = {}

SYSTEM_PROMPT = """You are an expert data-extraction assistant. Each turn you may be given an image and/or one or more attached documents (PDF, Word, Excel, PowerPoint, CSV, text). Some attachments include real images for you to look at, not just extracted text: scanned PDFs (little or no text layer) are rendered to page images, and pictures embedded inside Word/Excel/PowerPoint files are attached directly. Always read any attached images visually — they may contain the ONLY copy of the information (e.g. a scanned page has no text layer at all) or extra detail the text layer missed (stamps, signatures, charts, photos, handwriting). Your job:

1. Classify the overall content into EXACTLY ONE PRECISE category from this list:
   certificate, circular, offer_letter, resume, id_card, receipt_or_invoice, contract, form, report, letter, announcement, spreadsheet, presentation, table, chart_or_graph, code, handwritten_note, photo, screenshot, business_card, other_document, other

   Classification Guidelines:
   - "certificate": Diplomas, course completion certificates, awards, licenses, degrees, achievement or participation certificates.
   - "circular": Official government/university/company circulars, notices, academic memos, public directives, administrative bulletins.
   - "offer_letter": Job offers, appointment letters, internship offers, wage/salary letters, promotion letters.
   - "resume": CVs, resumes, bio-data, professional portfolios.
   - "id_card": Government IDs (Passport, Driver License, Aadhaar, PAN, Voter ID), student/employee ID badges.
   - "receipt_or_invoice": Bills, payment receipts, tax invoices, purchase orders, order confirmations.
   - "contract": NDAs, service agreements, lease/rental agreements, terms of service.
   - DO NOT classify as generic "other_document" if it matches specific categories like certificate, circular, offer_letter, resume, id_card, or invoice.

2. Extract ONLY the key, important information a human would actually want at a glance — not a transcription of the whole thing. Shape "key_info" to fit the content, e.g.:
   - certificate → recipient_name, issuing_organization, certificate_title/course_name, issue_date, credential_id/grade/distinction
   - circular → circular_number/id, issuing_authority/department, title/subject, issue_date, effective_date, target_audience, key_directive/action_required
   - offer_letter → candidate_name, company_name, job_title/designation, joining_date, compensation/salary, work_location
   - resume → name, role/title, top_skills, experience_years, latest_organization/education
   - receipt/invoice → vendor, invoice_number, date, total_amount, tax, payment_status
   - id_card → name, id_type, id_number, dob, expiry_date, address

3. If a document's content appears truncated ("...[truncated, N more characters]"), you may call the read_file tool with the file's id to fetch more of it before answering. If a PDF note says specific additional pages are NOT yet attached, you may call get_pdf_page_image(file_id, page_number) to see one of those specific pages. Never call a tool to re-fetch content already given to you in this same message.

4. Reply with ONLY a single valid JSON object — no markdown fences, no commentary, no <think> tags, nothing before or after the JSON. Use exactly this shape:

{
  "image_type": "<one precise category from the list above>",
  "confidence": <float 0-1>,
  "summary": "<one sentence describing the content>",
  "key_info": { <at most 8 of the most important fields, snake_case keys, short values> }
}

Rules:
- Be concise. Skip boilerplate: letterhead/address blocks, repeated institution names, cc/distribution lists, page footers, blank/unfilled fields, routine sign-offs — include them only if that boilerplate IS the actual point of the request.
- Never invent data that isn't visible in the image or present in the document text.
- If a field is unreadable or absent, omit it or set it to null — never guess.
- key_info must have at most 8 fields. Prioritize the most important ones; drop the rest rather than padding.
- Values should be short (a word, number, date, or short phrase) — not full paragraphs or verbatim quotes.
- Output must be strictly valid JSON, parseable by a strict parser. /no_think"""

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Fetch text content of a previously attached document by its file_id. "
            "Use this when a file's inlined content was marked truncated and you need "
            "more of it. Supports pagination via offset/length."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "The attachment's file_id."},
                "offset": {"type": "integer", "description": "Character offset to start from. Default 0."},
                "length": {"type": "integer", "description": f"Max characters to return. Default {TOOL_CHUNK_CHARS}."},
            },
            "required": ["file_id"],
        },
    },
}

GET_PDF_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_pdf_page_image",
        "description": (
            "Render a specific page of an attached PDF as an image so you can visually read it "
            "(useful for pages beyond the ones already attached, or if the text layer looks "
            "garbled, empty, or missing tables/charts/stamps/signatures the text can't capture). "
            "The rendered image is provided in the message right after your call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "The PDF attachment's file_id."},
                "page_number": {"type": "integer", "description": "1-indexed page number."},
            },
            "required": ["file_id", "page_number"],
        },
    },
}

THINK_TAG_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    stripped = THINK_TAG_RE.sub("", text or "", count=1)
    return stripped if stripped else (text or "")


def extract_json(text: str):
    text = strip_think(text).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end != -1 else text
    return json.loads(candidate)


# ---------------------------------------------------------------------------
# Document text extraction
# ---------------------------------------------------------------------------

def extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        parts.append(f"--- page {i + 1} ---\n{text}")
    return "\n\n".join(parts)


def render_pdf_page_b64(data: bytes, page_number: int, dpi: int = PDF_RENDER_DPI) -> str | None:
    """Render one 1-indexed PDF page to a base64 PNG. Returns None if out of range."""
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if page_number < 1 or page_number > doc.page_count:
            return None
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))
        return base64.b64encode(pix.tobytes("png")).decode("ascii")


def pdf_page_count(data: bytes) -> int:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return doc.page_count


def extract_embedded_images(data: bytes, max_images: int = MAX_EMBEDDED_IMAGES) -> list[str]:
    """DOCX/XLSX/PPTX are OOXML zip packages — embedded pictures always live under a
    .../media/ folder inside the zip regardless of which of the three it is."""
    images = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            media_names = sorted(
                n for n in z.namelist()
                if "/media/" in n and n.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"))
            )
            for name in media_names[:max_images]:
                images.append(base64.b64encode(z.read(name)).decode("ascii"))
    except (zipfile.BadZipFile, KeyError):
        pass
    return images


def extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def extract_xlsx(data: bytes) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    parts = []
    for sheet in wb.worksheets:
        parts.append(f"--- sheet: {sheet.title} ---")
        row_count = 0
        for row in sheet.iter_rows(values_only=True):
            if row_count >= 500:
                parts.append(f"... [sheet truncated at 500 rows, more rows exist]")
                break
            cells = ["" if c is None else str(c) for c in row]
            if any(c.strip() for c in cells):
                parts.append(" | ".join(cells))
            row_count += 1
    return "\n".join(parts)


def extract_pptx(data: bytes) -> str:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    parts = []
    for i, slide in enumerate(prs.slides):
        parts.append(f"--- slide {i + 1} ---")
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)
    return "\n".join(parts)


def extract_csv(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    parts = []
    for i, row in enumerate(reader):
        if i >= 1000:
            parts.append("... [truncated at 1000 rows]")
            break
        parts.append(" | ".join(row))
    return "\n".join(parts)


def extract_plain(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


DOC_EXTRACTORS = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "xlsx": extract_xlsx,
    "xlsm": extract_xlsx,
    "pptx": extract_pptx,
    "csv": extract_csv,
    "txt": extract_plain,
    "md": extract_plain,
    "json": extract_plain,
    "log": extract_plain,
}


def get_ext(filename: str) -> str:
    return (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "No file provided"}), 400

    filename = f.filename
    ext = get_ext(filename)
    data = f.read()
    file_id = uuid.uuid4().hex[:12]

    if ext in IMAGE_EXTS:
        b64 = base64.b64encode(data).decode("ascii")
        FILES[file_id] = {
            "kind": "image",
            "filename": filename,
            "ext": ext,
            "base64": b64,
            "size": len(data),
        }
        return jsonify(
            {
                "ok": True,
                "file_id": file_id,
                "filename": filename,
                "kind": "image",
                "size": len(data),
                "data_url": f"data:image/{ext};base64,{b64}",
            }
        )

    extractor = DOC_EXTRACTORS.get(ext)
    if not extractor:
        return jsonify({"ok": False, "error": f"Unsupported file type: .{ext}"}), 400

    try:
        text = extractor(data)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Could not read {filename}: {exc}"}), 400

    record = {
        "kind": "document",
        "filename": filename,
        "ext": ext,
        "text": text,
        "char_count": len(text),
        "size": len(data),
    }

    page_images_count = 0
    pdf_pages = None
    if ext == "pdf":
        try:
            total_pages = pdf_page_count(data)
            render_count = min(total_pages, MAX_AUTO_RENDER_PDF_PAGES)
            page_images = [render_pdf_page_b64(data, p) for p in range(1, render_count + 1)]
            record["raw_bytes"] = data  # kept so get_pdf_page_image can render more pages later
            record["pdf_total_pages"] = total_pages
            record["pdf_rendered_pages"] = render_count
            record["pdf_page_images"] = page_images
            page_images_count = len(page_images)
            pdf_pages = total_pages
        except Exception as exc:  # noqa: BLE001
            record["pdf_render_error"] = str(exc)

    embedded_images_count = 0
    if ext in OOXML_EXTS:
        embedded = extract_embedded_images(data)
        record["embedded_images"] = embedded
        embedded_images_count = len(embedded)

    FILES[file_id] = record

    return jsonify(
        {
            "ok": True,
            "file_id": file_id,
            "filename": filename,
            "kind": "document",
            "size": len(data),
            "char_count": len(text),
            "preview": text[:300],
            "pdf_total_pages": pdf_pages,
            "image_count": page_images_count + embedded_images_count,
        }
    )


@app.route("/api/health", methods=["POST"])
def health():
    host, model = get_config()
    try:
        resp = requests.get(
            f"{host}/api/tags",
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=15,
        )
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        names = [m.get("name") for m in tags]
        return jsonify(
            {
                "ok": True,
                "host": host,
                "model": model,
                "model_available": any(model in n for n in names) if names else None,
                "models": names,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "host": host, "error": str(exc)}), 200


def get_config():
    body = request.get_json(silent=True) or {}
    host = (body.get("host") or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
    model = body.get("model") or os.environ.get("REPOPILOT_LLM_MODEL") or DEFAULT_MODEL
    return host, model


def build_attachment_content(attachments):
    """Returns (text_block, image_b64_list, missing_filenames) for the given attachment refs."""
    text_blocks = []
    images = []
    missing = []
    for att in attachments or []:
        file_id = att.get("file_id")
        meta = FILES.get(file_id)
        if not meta:
            missing.append(att.get("filename") or file_id)
            continue
        if meta["kind"] == "image":
            images.append(meta["base64"])
        else:
            text = meta["text"]
            truncated = len(text) > INLINE_CHAR_LIMIT
            snippet = text[:INLINE_CHAR_LIMIT]
            note = (
                f"\n...[truncated, {len(text) - INLINE_CHAR_LIMIT} more characters — "
                f"call read_file(file_id=\"{file_id}\") for more]"
                if truncated
                else ""
            )

            extra_notes = []
            if meta.get("pdf_page_images"):
                images.extend(meta["pdf_page_images"])
                rendered, total = meta["pdf_rendered_pages"], meta["pdf_total_pages"]
                if not text.strip():
                    extra_notes.append(
                        f"(No selectable text layer — this looks like a scanned PDF. "
                        f"Pages 1-{rendered} of {total} are ALREADY attached as images below for you to read visually "
                        f"— do not call get_pdf_page_image for these, you already have them.)"
                    )
                else:
                    extra_notes.append(
                        f"(Pages 1-{rendered} of {total} are ALREADY attached as images below "
                        f"— do not re-fetch these via get_pdf_page_image.)"
                    )
                if total > rendered:
                    extra_notes.append(
                        f"Only pages {rendered + 1}-{total} are NOT yet attached — call "
                        f"get_pdf_page_image(file_id=\"{file_id}\", page_number=N) for one of those specific pages "
                        f"only if you actually need it."
                    )
            elif meta.get("pdf_render_error"):
                extra_notes.append(f"(Could not render page images: {meta['pdf_render_error']})")

            if meta.get("embedded_images"):
                images.extend(meta["embedded_images"])
                extra_notes.append(f"({len(meta['embedded_images'])} embedded picture(s) from this file also attached below.)")

            extra = ("\n" + "\n".join(extra_notes)) if extra_notes else ""
            text_blocks.append(
                f"### Attached file: {meta['filename']} (file_id: {file_id}, {meta['char_count']} chars)\n{snippet}{note}{extra}"
            )
    return "\n\n".join(text_blocks), images, missing


def run_read_file_tool(args):
    file_id = args.get("file_id", "")
    offset = int(args.get("offset") or 0)
    length = int(args.get("length") or TOOL_CHUNK_CHARS)
    meta = FILES.get(file_id)
    if not meta or meta["kind"] != "document":
        return {"error": f"Unknown file_id: {file_id}"}
    text = meta["text"]
    chunk = text[offset : offset + length]
    return {
        "file_id": file_id,
        "filename": meta["filename"],
        "offset": offset,
        "returned_chars": len(chunk),
        "total_chars": len(text),
        "has_more": offset + length < len(text),
        "content": chunk,
    }


def run_get_pdf_page_tool(args):
    """Returns (ack_dict, image_b64_or_None). Ollama tool-result messages are text-only,
    so the image (if any) has to be delivered as a separate user-role message right after —
    there's no way to attach media to a role:"tool" message itself."""
    file_id = args.get("file_id", "")
    try:
        page_number = int(args.get("page_number"))
    except (TypeError, ValueError):
        return {"error": "page_number must be an integer"}, None

    meta = FILES.get(file_id)
    if not meta or meta.get("ext") != "pdf" or "raw_bytes" not in meta:
        return {"error": f"Unknown PDF file_id: {file_id}"}, None

    image_b64 = render_pdf_page_b64(meta["raw_bytes"], page_number)
    if image_b64 is None:
        return {"error": f"Page {page_number} out of range (file has {meta.get('pdf_total_pages')} pages)"}, None

    return (
        {
            "file_id": file_id,
            "filename": meta["filename"],
            "page_number": page_number,
            "status": "Image attached in the next message.",
        },
        image_b64,
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    start_time = time.monotonic()
    body = request.get_json(force=True)
    host, model = get_config()
    message = body.get("message", "") or "Analyze the attached content."
    history = body.get("history", [])
    attachments = body.get("attachments", [])

    attach_text, images, missing = build_attachment_content(attachments)
    if missing:
        return jsonify(
            {
                "ok": False,
                "error": (
                    f"Attachment(s) not found on the server: {', '.join(missing)}. "
                    "The server may have restarted since you uploaded — please re-attach and resend."
                ),
            }
        ), 400

    user_content = message
    if attach_text:
        user_content = f"{message}\n\n{attach_text}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    user_msg = {"role": "user", "content": user_content}
    if images:
        user_msg["images"] = images
    messages.append(user_msg)

    tools_used = []
    # Confirmed live against this Kaggle instance: qwen3-vl:8b breaks when "tools" is present
    # on a request that also carries an image — content comes back empty, thinking runs to
    # ~10k chars, and it sometimes hallucinates a tool call that was never asked for. Tool
    # calling and vision don't mix on this model build, so tools are disabled for any turn
    # that has an image in it (including one just injected by get_pdf_page_image below).
    has_images_this_turn = bool(images)
    try:
        for iteration in range(MAX_TOOL_ITERATIONS):
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
            }
            has_docs = any(FILES.get(a.get("file_id"), {}).get("kind") == "document" for a in attachments)
            has_pdf = any(FILES.get(a.get("file_id"), {}).get("ext") == "pdf" for a in attachments)
            tools = []
            if not has_images_this_turn:
                if has_docs:
                    tools.append(READ_FILE_TOOL)
                if has_pdf:
                    tools.append(GET_PDF_PAGE_TOOL)
            if tools:
                payload["tools"] = tools

            resp = requests.post(
                f"{host}/api/chat",
                json=payload,
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            resp_msg = resp.json().get("message", {})
            tool_calls = resp_msg.get("tool_calls") or []

            if not tool_calls:
                raw_content = resp_msg.get("content", "")
                break

            messages.append(resp_msg)
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if name == "read_file":
                    result = run_read_file_tool(args)
                    tools_used.append({"name": name, "args": args})
                    messages.append({"role": "tool", "content": json.dumps(result)})
                elif name == "get_pdf_page_image":
                    result, image_b64 = run_get_pdf_page_tool(args)
                    tools_used.append({"name": name, "args": args})
                    messages.append({"role": "tool", "content": json.dumps(result)})
                    if image_b64:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"(page {args.get('page_number')} of {result.get('filename', 'the PDF')} you requested)",
                                "images": [image_b64],
                            }
                        )
                        has_images_this_turn = True  # disable tools on the next iteration too
                else:
                    result = {"error": f"Unknown tool: {name}"}
                    tools_used.append({"name": name, "args": args})
                    messages.append({"role": "tool", "content": json.dumps(result)})
        else:
            raw_content = resp_msg.get("content", "")
    except requests.exceptions.RequestException as exc:
        return jsonify({"ok": False, "error": f"Could not reach {host}: {exc}"}), 502

    elapsed = round(time.monotonic() - start_time, 2)
    visible = strip_think(raw_content)

    parsed = None
    parse_error = None
    category = None
    original_filename = "document.json"

    try:
        parsed = extract_json(raw_content)
        if isinstance(parsed, dict):
            # Extract category identified by the model
            category = str(parsed.get("image_type") or parsed.get("category") or "other").strip().lower()
            
            # Determine source filename prefix if available
            orig_name = "document"
            if attachments and len(attachments) > 0:
                first_att = attachments[0]
                orig_name = first_att.get("filename") or FILES.get(first_att.get("file_id"), {}).get("filename", "document")
            name_base = os.path.splitext(os.path.basename(orig_name))[0]
            clean_name = re.sub(r"[^\w\-_]", "_", name_base).strip("_") or "extracted"
            original_filename = f"{clean_name}.json"

    except json.JSONDecodeError as exc:
        parse_error = str(exc)

    return jsonify(
        {
            "ok": True,
            "raw": visible,
            "parsed": parsed,
            "parse_error": parse_error,
            "elapsed_seconds": elapsed,
            "tools_used": tools_used,
            "category": category,
            "original_filename": original_filename,
        }
    )



@app.route("/api/save-to-db", methods=["POST"])
def save_to_db():
    """Save extracted JSON into the appropriate MongoDB Atlas collection by category."""
    if _mongo_db is None:
        return jsonify({"ok": False, "error": "MongoDB is not configured. Check your MONGO_URI in .env"}), 503

    body = request.get_json(force=True) or {}
    parsed = body.get("parsed")
    category = body.get("category")
    filename = body.get("filename", "unknown")

    if not parsed:
        return jsonify({"ok": False, "error": "No parsed JSON data provided"}), 400
    if not isinstance(parsed, dict):
        return jsonify({"ok": False, "error": "Parsed data must be a JSON object"}), 400

    # Derive collection name from category (fall back to image_type inside the doc)
    if not category:
        category = str(parsed.get("image_type") or parsed.get("category") or "other").strip().lower()
    collection_name = re.sub(r"[^\w]", "_", category).strip("_") or "other"

    # Build the record to insert
    record = {
        "source_filename": filename,
        "category": category,
        "saved_at": datetime.datetime.utcnow(),
        "extracted_data": parsed,
    }

    try:
        collection = _mongo_db[collection_name]
        result = collection.insert_one(record)
        inserted_id = str(result.inserted_id)
        print(f"[MongoDB] Saved '{filename}' -> db={_MONGO_DB}, collection={collection_name}, id={inserted_id}")
        return jsonify({
            "ok": True,
            "collection": collection_name,
            "inserted_id": inserted_id,
            "db": _MONGO_DB,
        })
    except (ConnectionFailure, OperationFailure) as exc:
        return jsonify({"ok": False, "error": f"MongoDB error: {exc}"}), 500
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/save-local", methods=["POST"])
def save_local():
    """Save extracted JSON to a local folder based on category."""
    body = request.get_json(force=True) or {}
    parsed = body.get("parsed")
    category = body.get("category")
    filename = body.get("filename", "document.json")

    if not parsed or not isinstance(parsed, dict):
        return jsonify({"ok": False, "error": "Invalid parsed JSON data"}), 400

    if not category:
        category = str(parsed.get("image_type") or parsed.get("category") or "other").strip().lower()
    
    category_slug = re.sub(r"[^\w\-_]", "_", category).replace("__", "_").strip("_") or "other"
    
    name_base = os.path.splitext(os.path.basename(filename))[0]
    clean_name = re.sub(r"[^\w\-_]", "_", name_base).strip("_") or "extracted"
    
    category_dir = os.path.join(os.path.dirname(__file__), "exports", category_slug)
    os.makedirs(category_dir, exist_ok=True)
    
    save_filename = f"{clean_name}.json"
    full_save_path = os.path.join(category_dir, save_filename)
    
    try:
        with open(full_save_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        saved_path = os.path.relpath(full_save_path, os.path.dirname(__file__)).replace("\\", "/")
        print(f"[Export] Saved locally: {saved_path}")
        return jsonify({"ok": True, "saved_path": saved_path})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/open-folder", methods=["POST"])
def open_folder():
    """Open the local exports folder in Windows Explorer."""
    data = request.get_json(force=True) or {}
    rel_path = data.get("path")
    if not rel_path:
        target_dir = os.path.join(os.path.dirname(__file__), "exports")
    else:
        full_path = os.path.join(os.path.dirname(__file__), rel_path)
        if os.path.isfile(full_path):
            target_dir = os.path.dirname(full_path)
        else:
            target_dir = full_path

    os.makedirs(target_dir, exist_ok=True)
    try:
        os.startfile(target_dir)
        return jsonify({"ok": True, "opened": target_dir})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
