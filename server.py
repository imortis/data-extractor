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
from flask import Flask, jsonify, request, send_from_directory, send_file

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
REQUEST_TIMEOUT = 600
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
   - "table" or "spreadsheet": Statements, admission lists, financial reports, schedules, matrices, marksheets, rosters, or spreadsheets containing tabular rows and columns.
   - DO NOT classify as generic "other_document" if it matches specific categories.

2. Extract key summary information:
   - primary_name: Main person, institution, company, vendor, or issuing authority (e.g. 'Acme Corp', 'Reserve Bank', 'John Doe', 'Tech University').
   - concise_topic: Strictly 2 to 3 words describing the core subject or role (e.g. 'Financial Balance Sheet', 'Quarterly Sales Report', 'Software Engineer Resume', 'Itemized Product Invoice').
   - summary: One sentence summarizing the document.
   - key_info: At most 8 of the most important high-level scalar fields (e.g. document_date, total_amount, report_period, total_count).

3. Extract ALL Tables and Tabular Data into "tables":
   If the image or document contains ANY table, schedule, statement, ledger, balance sheet, price list, inventory, or spreadsheet grid:
   - Extract EVERY table dynamically into the "tables" array.
   - Works for ANY table domain (e.g. invoices, academic schedules, admissions, payroll, balance sheets, scientific results, sports stats, rosters).
   - Schema per table:
     {
       "title": "<Document / Table Title, e.g. 'Quarterly Financial Statement' or 'Product Inventory' or 'Sanctioned Intake'>",
       "headers": ["<Col 1>", "<Col 2>", "<Col 3>", ...],
       "rows": [
         ["<Row 1 Col 1>", "<Row 1 Col 2>", ...],
         ["<Row 2 Col 1>", "<Row 2 Col 2>", ...]
       ]
     }
   - CRITICAL UNIVERSAL RULES FOR HIGH-ACCURACY TABLE EXTRACTION:
     a. Flatten Grouped / Multi-Tier Headers: When ANY table has super-headers grouping multiple sub-columns (e.g. 'Q1' with sub-columns 'Actual', 'Budget', or 'Category A' with 'Intake', 'Admissions', or 'Taxes' with 'CGST', 'SGST'), flatten them into composite, fully-qualified header names: e.g. 'Q1 - Actual', 'Q1 - Budget', 'Taxes - CGST', 'Taxes - SGST'. This ensures every Excel column has an accurate, unambiguous header regardless of how complex the table header hierarchy is.
     b. Strict Column-Count Invariant: Every row in `rows` MUST have the EXACT same number of elements as the `headers` list. Never skip or collapse columns. If a cell is blank or has no data, output "" or "0" (if it is a numeric count/amount column). Never let columns shift horizontally.
     c. Complete Row Coverage: Extract every row from top to bottom. Do not skip rows. Include all serial numbers, line items, item descriptions, subtotal rows, and grand total rows.
     d. Exact Data Preservation: Maintain exact numbers, codes, currency values, percentages, abbreviations, and text verbatim. Do not truncate, round, or approximate values.
     e. If no tables exist in the content, set "tables": [].

4. If a document's content appears truncated ("...[truncated, N more characters]"), you may call the read_file tool with the file's id to fetch more of it before answering. If a PDF note says specific additional pages are NOT yet attached, you may call get_pdf_page_image(file_id, page_number) to see one of those specific pages. Never call a tool to re-fetch content already given to you in this same message.

5. Reply with ONLY a single valid JSON object — no markdown fences, no commentary, no <think> tags, nothing before or after the JSON. Use exactly this shape:

{
  "image_type": "<one precise category from the list above>",
  "confidence": <float 0-1>,
  "summary": "<one sentence describing the content>",
  "primary_name": "<primary person, student, candidate, recipient, company, or issuing authority name>",
  "concise_topic": "<strictly 2 to 3 words describing the specific topic or document type>",
  "key_info": { <at most 8 of the most important fields, snake_case keys, short values> },
  "tables": [
    {
      "title": "<table title>",
      "headers": ["<header1>", "<header2>", ...],
      "rows": [
        ["<val1>", "<val2>", ...],
        ...
      ]
    }
  ]
}

Rules:
- Be concise. Skip boilerplate: letterhead/address blocks, repeated institution names, cc/distribution lists, page footers, blank/unfilled fields, routine sign-offs — include them only if that boilerplate IS the actual point of the request.
- Never invent data that isn't visible in the image or present in the document text.
- If a field is unreadable or absent, omit it or set it to null — never guess.
- primary_name should be the main person or organization's name (e.g. 'Sahitya Chadda', 'Google', 'NatWest').
- concise_topic MUST be strictly 2 to 3 words describing the core subject, role, or event (e.g. 'Software Engineer Resume', 'Coding Event Participation', 'Cloud Services Invoice', 'Academic Examination Circular').
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

def optimize_image_bytes(data: bytes, max_dim: int = 1800) -> tuple[str, str]:
    """Optimize image to safe max dimension for vision models, returning (base64_str, ext)."""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            im = im.resize(new_size, Image.Resampling.LANCZOS)

        out_format = "PNG" if im.mode in ("RGBA", "P") else "JPEG"
        out_io = io.BytesIO()
        if out_format == "JPEG":
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out_io, format="JPEG", quality=90, optimize=True)
        else:
            im.save(out_io, format="PNG", optimize=True)

        opt_data = out_io.getvalue()
        return base64.b64encode(opt_data).decode("ascii"), out_format.lower()
    except Exception:
        return base64.b64encode(data).decode("ascii"), "png"


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
        b64, out_ext = optimize_image_bytes(data)
        FILES[file_id] = {
            "kind": "image",
            "filename": filename,
            "ext": out_ext,
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
                "data_url": f"data:image/{out_ext};base64,{b64}",
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
    suggested_filename = "document_data.json"

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
            
            # Generate dynamic filename: name_concisetopicin2-3words.json
            suggested_filename = generate_suggested_filename(parsed, orig_name)

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
            "suggested_filename": suggested_filename,
            "original_filename": suggested_filename,
        }
    )


def sanitize_token(text: str) -> str:
    """Keep alphanumeric characters and underscores."""
    text = re.sub(r"[^\w\s-]", "", str(text or ""))
    tokens = [t for t in re.split(r"[\s_-]+", text) if t]
    return "_".join(tokens)


def extract_primary_name(parsed: dict, fallback: str = "") -> str:
    if not isinstance(parsed, dict):
        return fallback or "document"
    # Check explicit field from model prompt
    for field in ("primary_name", "entity_name", "candidate_name", "recipient_name", "name", "full_name"):
        val = parsed.get(field)
        if val and isinstance(val, str) and val.strip():
            return val.strip()

    # Check inside key_info
    key_info = parsed.get("key_info") or {}
    if isinstance(key_info, dict):
        for key in (
            "recipient_name", "candidate_name", "name", "full_name", "student_name",
            "employee_name", "vendor", "issuing_organization", "issuing_authority",
            "company_name", "company", "organization", "author"
        ):
            val = key_info.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()

    # Fallback to base of uploaded filename if provided
    if fallback:
        base = os.path.splitext(os.path.basename(fallback))[0]
        clean = re.sub(r"[^\w\s-]", "", base).strip()
        if clean and clean.lower() not in ("document", "extracted", "file", "upload", "unknown", "image"):
            return clean

    return "document"


def extract_concise_topic(parsed: dict, fallback_category: str = "document") -> str:
    if not isinstance(parsed, dict):
        return "extracted_data"

    # 1. Check explicit field from model prompt
    raw_topic = parsed.get("concise_topic") or parsed.get("short_topic") or parsed.get("topic")

    # 2. Check key_info for titles/events/roles
    if not raw_topic:
        key_info = parsed.get("key_info") or {}
        if isinstance(key_info, dict):
            for key in (
                "certificate_title", "event_name", "course_name", "job_title",
                "role", "designation", "title", "subject", "id_type"
            ):
                val = key_info.get(key)
                if val and isinstance(val, str) and val.strip():
                    raw_topic = val.strip()
                    break

    # 3. Check summary if available
    if not raw_topic and parsed.get("summary"):
        raw_topic = parsed.get("summary")

    # 4. Fall back to category
    if not raw_topic:
        raw_topic = parsed.get("image_type") or parsed.get("category") or fallback_category

    words = [w for w in re.findall(r"[A-Za-z0-9]+", str(raw_topic)) if len(w) > 1 or w.isalnum()]
    stopwords = {"a", "an", "the", "and", "or", "of", "in", "for", "with", "at", "by", "from", "on", "to", "is", "as"}
    filtered_words = [w for w in words if w.lower() not in stopwords]
    candidate_words = filtered_words if len(filtered_words) >= 2 else words

    if len(candidate_words) >= 3:
        selected_words = candidate_words[:3]
    elif len(candidate_words) == 2:
        selected_words = candidate_words
    elif len(candidate_words) == 1:
        cat = parsed.get("image_type") or fallback_category or "document"
        cat_words = [w for w in re.findall(r"[A-Za-z0-9]+", str(cat)) if w.lower() != candidate_words[0].lower()]
        selected_words = candidate_words + (cat_words[:1] if cat_words else ["Doc"])
    else:
        selected_words = ["Data", "Doc"]

    return "_".join(w.capitalize() for w in selected_words)


def generate_suggested_filename(parsed: dict | None, orig_name: str = "") -> str:
    """Generate filename adhering to: name_concisetopicin2-3words.json"""
    if not parsed or not isinstance(parsed, dict):
        base = os.path.splitext(os.path.basename(orig_name))[0] if orig_name else "document"
        clean = sanitize_token(base) or "document"
        return f"{clean}.json"

    primary_name = extract_primary_name(parsed, orig_name)
    clean_name = sanitize_token(primary_name) or "document"
    topic = extract_concise_topic(parsed)
    clean_topic = sanitize_token(topic) or "Data"

    return f"{clean_name}_{clean_topic}.json"


@app.route("/api/save-to-db", methods=["POST"])
def save_to_db():
    """Save extracted JSON into the appropriate MongoDB Atlas collection by category."""
    if _mongo_db is None:
        return jsonify({"ok": False, "error": "MongoDB is not configured. Check your MONGO_URI in .env"}), 503

    body = request.get_json(force=True) or {}
    parsed = body.get("parsed")
    category = body.get("category")
    filename = body.get("filename", "")

    if not parsed:
        return jsonify({"ok": False, "error": "No parsed JSON data provided"}), 400
    if not isinstance(parsed, dict):
        return jsonify({"ok": False, "error": "Parsed data must be a JSON object"}), 400

    # Derive collection name from category (fall back to image_type inside the doc)
    if not category:
        category = str(parsed.get("image_type") or parsed.get("category") or "other").strip().lower()
    collection_name = re.sub(r"[^\w]", "_", category).strip("_") or "other"

    # Standardize filename format
    if not filename or filename in ("unknown", "document.json", f"{category}.json", "extracted.json"):
        filename = generate_suggested_filename(parsed)
    elif not filename.lower().endswith(".json"):
        filename = f"{filename}.json"

    # Build the record to insert
    record = {
        "source_filename": filename,
        "saved_filename": filename,
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
            "filename": filename,
        })
    except (ConnectionFailure, OperationFailure) as exc:
        return jsonify({"ok": False, "error": f"MongoDB error: {exc}"}), 500
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def generate_excel_bytes(parsed: dict, filename: str = "") -> bytes:
    """Build an in-memory .xlsx workbook with professional formatting from parsed table data."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    tables = parsed.get("tables") if isinstance(parsed, dict) else None

    # Fallback if no tables extracted
    if not tables or not isinstance(tables, list):
        headers = ["Field", "Extracted Value"]
        key_info = parsed.get("key_info", {}) if isinstance(parsed, dict) else {}
        rows = [[k.replace("_", " ").title(), str(v)] for k, v in key_info.items()]
        if not rows:
            rows = [["Summary", str(parsed.get("summary", ""))] if isinstance(parsed, dict) else ["Data", str(parsed)]]
        tables = [{"title": parsed.get("summary", "Extracted Data") if isinstance(parsed, dict) else "Data", "headers": headers, "rows": rows}]

    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )
    double_bottom_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='94A3B8'),
        bottom=Side(style='double', color='1E293B')
    )

    header_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")

    total_font = Font(name="Calibri", size=10, bold=True, color="0F172A")
    total_fill = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")

    even_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    odd_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

    for t_idx, tbl in enumerate(tables):
        if not isinstance(tbl, dict):
            continue
        title = tbl.get("title") or f"Table {t_idx + 1}"
        safe_title = re.sub(r"[\\/*?:\[\]]", "_", title)[:30].strip() or f"Sheet{t_idx+1}"
        ws = wb.create_sheet(title=safe_title)
        ws.views.sheetView[0].showGridLines = True

        headers = tbl.get("headers") or []
        rows = tbl.get("rows") or []

        curr_row = 1

        # Title Row Banner
        if title:
            max_cols = max(len(headers), 1)
            ws.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=max_cols)
            title_cell = ws.cell(row=curr_row, column=1, value=title)
            title_cell.font = Font(name="Calibri", size=11, bold=True, color="0F172A")
            title_cell.alignment = Alignment(horizontal="left", vertical="center")
            ws.row_dimensions[curr_row].height = 24
            curr_row += 1

        # Header Row
        header_row_num = curr_row
        ws.row_dimensions[header_row_num].height = 32
        for col_idx, h in enumerate(headers, 1):
            cell = ws.cell(row=header_row_num, column=col_idx, value=str(h))
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
        curr_row += 1

        # Data Rows
        for r_idx, row in enumerate(rows):
            is_total_row = False
            for check_val in row[:3]:
                if any(k in str(check_val).lower() for k in ("total", "subtotal", "sub total")):
                    is_total_row = True
                    break

            ws.row_dimensions[curr_row].height = 20
            row_fill = total_fill if is_total_row else (even_fill if r_idx % 2 == 0 else odd_fill)
            row_font = total_font if is_total_row else Font(name="Calibri", size=10, color="1E293B")
            row_border = double_bottom_border if is_total_row else thin_border

            for col_idx in range(1, len(headers) + 1):
                raw_val = row[col_idx - 1] if col_idx - 1 < len(row) else ""
                val = raw_val

                # Attempt numeric conversion
                val_str = str(raw_val).strip() if raw_val is not None else ""
                val_clean = val_str.replace(",", "")
                is_num = False
                align_horiz = "left"

                if val_clean.lstrip("-").isdigit():
                    val = int(val_clean)
                    is_num = True
                    align_horiz = "right"
                elif re.match(r"^-?\d+\.\d+$", val_clean):
                    try:
                        val = float(val_clean)
                        is_num = True
                        align_horiz = "right"
                    except ValueError:
                        pass

                if col_idx == 1 and (is_num or len(val_str) <= 4):
                    align_horiz = "center"

                cell = ws.cell(row=curr_row, column=col_idx, value=val)
                cell.font = row_font
                cell.fill = row_fill
                cell.alignment = Alignment(horizontal=align_horiz, vertical="center")
                cell.border = row_border

            curr_row += 1

        # Auto-adjust column widths
        for col_idx in range(1, len(headers) + 1):
            col_letter = get_column_letter(col_idx)
            max_len = 0
            for r in range(header_row_num, curr_row):
                c_val = ws.cell(row=r, column=col_idx).value
                if c_val is not None:
                    max_len = max(max_len, len(str(c_val)))
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 11), 60)

    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio.getvalue()


@app.route("/api/export-excel", methods=["POST"])
def export_excel():
    """Convert extracted table data into a clean, styled .xlsx spreadsheet for download."""
    data = request.get_json(force=True) or {}
    parsed = data.get("parsed") or {}
    filename = data.get("filename") or "extracted_data.xlsx"

    try:
        excel_bytes = generate_excel_bytes(parsed, filename)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        clean_name = re.sub(r"[^\w\-_]", "_", base_name).strip("_") or "extracted_table"
        download_name = f"{clean_name}.xlsx"

        return send_file(
            io.BytesIO(excel_bytes),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=download_name,
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/save-local", methods=["POST"])
def save_local():
    """Save extracted JSON and Excel to a local folder based on category, named name_concisetopicin2-3words.json/.xlsx."""
    body = request.get_json(force=True) or {}
    parsed = body.get("parsed")
    category = body.get("category")
    filename = body.get("filename", "")

    if not parsed or not isinstance(parsed, dict):
        return jsonify({"ok": False, "error": "Invalid parsed JSON data"}), 400

    if not category:
        category = str(parsed.get("image_type") or parsed.get("category") or "other").strip().lower()

    category_slug = re.sub(r"[^\w\-_]", "_", category).replace("__", "_").strip("_") or "other"

    # Ensure standardized filename: name_concisetopicin2-3words.json
    if not filename or filename in ("document.json", f"{category}.json", "extracted.json"):
        save_filename = generate_suggested_filename(parsed)
    else:
        name_base = os.path.splitext(os.path.basename(filename))[0]
        clean_name = re.sub(r"[^\w\-_]", "_", name_base).strip("_") or "extracted"
        save_filename = f"{clean_name}.json"

    category_dir = os.path.join(os.path.dirname(__file__), "exports", category_slug)
    os.makedirs(category_dir, exist_ok=True)

    full_save_path = os.path.join(category_dir, save_filename)

    try:
        with open(full_save_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        saved_path = os.path.relpath(full_save_path, os.path.dirname(__file__)).replace("\\", "/")
        print(f"[Export] Saved locally: {saved_path}")

        # Also generate and save Excel file if tables are present
        excel_saved_path = None
        if parsed.get("tables"):
            try:
                base_clean = os.path.splitext(save_filename)[0]
                excel_filename = f"{base_clean}.xlsx"
                excel_full_path = os.path.join(category_dir, excel_filename)
                excel_bytes = generate_excel_bytes(parsed, excel_filename)
                with open(excel_full_path, "wb") as f_excel:
                    f_excel.write(excel_bytes)
                excel_saved_path = os.path.relpath(excel_full_path, os.path.dirname(__file__)).replace("\\", "/")
                print(f"[Export] Saved Excel locally: {excel_saved_path}")
            except Exception as e_err:
                print(f"[Export] Could not write local Excel: {e_err}")

        return jsonify({
            "ok": True,
            "saved_path": saved_path,
            "filename": save_filename,
            "excel_saved_path": excel_saved_path
        })
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
