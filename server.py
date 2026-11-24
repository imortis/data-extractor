"""
Backend for the image / document data-extraction chat UI.

- Accepts images (vision, via /api/chat "images") and documents (PDF, DOCX,
  XLSX, PPTX, CSV, TXT, MD, JSON) which are text-extracted server-side.
- PDFs are ALSO rendered to page images (via PyMuPDF) and sent to the vision
  model alongside any extracted text — this is what makes scanned PDFs (no
  text layer at all) work, the same way ChatGPT/Gemini handle them: by
  actually looking at the page, not just reading a text layer that may not
  exist.
- Large attachments are split into parts, sent to the model one part at a
  time, and merged into a single result. Anything that could not be processed
  is reported in "warnings" instead of being dropped silently.
- DOCX/XLSX/PPTX are OOXML zip packages, so any embedded pictures (photos,
  screenshots, scanned inserts, slide backgrounds) are pulled straight out
  of the zip's media folder and sent to the vision model too, not just the
  text layer.
- Every extracted value is checked against evidence the model didn't produce
  (see verification.py), and the result is returned under "verification".
- Times the full round trip (across all parts) and returns it.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
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

import digit_repair
import file_tables
import records
import repair
from verification import (
    UNCHECKED,
    VERIFIED,
    SourceEvidence,
    verify_record,
)

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
INLINE_CHAR_LIMIT = 12000  # max document chars sent to the model in one part
MAX_IMAGES_PER_PART = 6  # max images sent to the model in one part (GPU memory)
MAX_PARTS_PER_REQUEST = 12  # anything beyond this is reported as not processed
MAX_EXCEL_DIGITS = 15  # Excel silently corrupts numbers with more significant digits
MAX_OCR_IMAGES = 12  # OCR takes tens of seconds per image on CPU
# The model's default temperature is 1: identical requests gave different misreads.
MODEL_OPTIONS = {"temperature": 0, "seed": 42}
MODEL_ALTERNATIVES = 5  # alternatives per token, for repairing misread digits
MODEL_KEEP_ALIVE = "30m"  # reloading the model costs about 9 seconds
# An image costs about one token per 32x32 pixels, so small scans leave under a token per digit.
# Doubling the size cut misread cells from 7 to 1 of 720 on the sample scans.
MODEL_IMAGE_SCALE = 2
MODEL_IMAGE_MAX_DIM = 3200
MODEL_IMAGE_MAX_PIXELS = 6_000_000
MAX_IMAGES_TO_ENLARGE = 2  # several enlarged images in one request risk running out of GPU memory
MIN_TEXT_LAYER_CHARS = 20  # fewer than this on a PDF page means it's effectively a scan
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
   - record_owner: "student" if the content is about students (ID cards, grade reports, admissions), "faculty" if it is about teaching staff or employees (publications, appointments), otherwise "other".
   - key_info: Every important high-level scalar field that is printed in the content (e.g. document_date, total_amount, report_period, usn). Copy values as printed.
   - sections: Any repeating non-table content, as one section per kind of entry. For example a resume's work experience, education and projects; a certificate's list of courses; a letter's list of conditions. Every entry is an object of short snake_case fields, for example {"title": "Experience", "items": [{"role": "...", "organization": "...", "start_date": "...", "end_date": "..."}]}. Use [] if there is nothing like this. Do not repeat table content here.

3. Extract ALL Tables and Tabular Data into "tables":
   If the image or document contains ANY table, schedule, statement, ledger, balance sheet, price list, inventory, or spreadsheet grid:
   - Extract EVERY table dynamically into the "tables" array.
   - Keep EVERY column shown in the table, exactly as it appears, even if all of its cells are blank. Never drop a column because it is empty.
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
     b. Strict Column-Count Invariant: Every row in `rows` MUST have the EXACT same number of elements as the `headers` list. Never skip or collapse columns. If a cell is truly blank or unreadable, output "". If a cell has a printed value, output that value, including a printed "0". Never write a value that is not printed. Never let columns shift horizontally.
     c. Complete Row Coverage: Extract every row from top to bottom. Do not skip rows. Include all serial numbers, line items, item descriptions, subtotal rows, and grand total rows, including subtotal rows that have no label.
     d. Exact Data Preservation: Maintain exact numbers, codes, currency values, percentages, abbreviations, and text verbatim. Do not truncate, round, or approximate values. Do not expand abbreviations, change capitalization, or correct spelling: a printed 'ENGG.' or 'SIENCE' stays exactly as printed.
     e. If no tables exist in the content, set "tables": [].

4. Large attachments may arrive in several parts, one part per message. If told you are seeing one part, extract only what is in that part and do not guess at the rest. If told how the previous part's last table ended and this part continues that table, reuse its exact title and headers, and do not repeat its header row as data.

5. Reply with ONLY a single valid JSON object — no markdown fences, no commentary, no <think> tags, nothing before or after the JSON. Use exactly this shape:

{
  "image_type": "<one precise category from the list above>",
  "confidence": <float 0-1>,
  "summary": "<one sentence describing the content>",
  "primary_name": "<primary person, student, candidate, recipient, company, or issuing authority name>",
  "concise_topic": "<strictly 2 to 3 words describing the specific topic or document type>",
  "record_owner": "<student | faculty | other>",
  "key_info": { <the important fields, snake_case keys, short values> },
  "sections": [ { "title": "<section name>", "items": [ { <short snake_case fields> } ] } ],
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
- Be concise in "summary" and "key_info". There, skip boilerplate: letterhead/address blocks, repeated institution names, cc/distribution lists, page footers, blank/unfilled fields, routine sign-offs — include them only if that boilerplate IS the actual point of the request. This does NOT apply to "tables": always extract tables in full, including blank cells and blank columns.
- Never invent data that isn't visible in the image or present in the document text.
- If a "key_info" field is unreadable or absent, omit it or set it to null — never guess. In "tables", use "" for those cells so the columns stay aligned.
- primary_name should be the main person or organization's name (e.g. 'Sahitya Chadda', 'Google', 'NatWest'), copied as printed. If no such name is printed, set it to null. Never infer it from a course list, a code, or general knowledge.
- concise_topic MUST be strictly 2 to 3 words describing the core subject, role, or event (e.g. 'Software Engineer Resume', 'Coding Event Participation', 'Cloud Services Invoice', 'Academic Examination Circular').
- "key_info" values should be short (a word, number, date, or short phrase) — not full paragraphs or verbatim quotes. Table cells stay verbatim.
- Output must be strictly valid JSON, parseable by a strict parser. /no_think"""

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

def extract_pdf_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [
        f"--- page {i + 1} ---\n{(page.extract_text() or '').strip()}"
        for i, page in enumerate(reader.pages)
    ]


def extract_pdf(data: bytes) -> str:
    return "\n\n".join(extract_pdf_pages(data))


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
            # A cell's line breaks would split one row into several lines of evidence.
            parts.append(" | ".join(" ".join(cell.text.split()) for cell in row.cells))
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


def enlarge_for_model(image_b64: str) -> str:
    """Gives the model enough pixels per digit. OCR and previews keep the original."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        scale = min(
            MODEL_IMAGE_SCALE,
            MODEL_IMAGE_MAX_DIM / max(image.size),
            (MODEL_IMAGE_MAX_PIXELS / (image.width * image.height)) ** 0.5,
        )
        if scale <= 1:
            return image_b64
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=95)
        return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return image_b64


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
            "original": data,
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

    page_texts = None
    try:
        if ext == "pdf":
            page_texts = extract_pdf_pages(data)
            text = "\n\n".join(page_texts)
        else:
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
        "original": data,
    }

    try:
        if ext == "pdf":
            record["file_tables"], record["pdf_page_tables"] = file_tables.from_pdf(data)
        elif ext in file_tables.EXTRACTORS:
            record["file_tables"], record["model_text"] = file_tables.EXTRACTORS[ext](data)
    except Exception as exc:  # noqa: BLE001
        # The model reads the tables instead, as it always did.
        for key in ("file_tables", "model_text", "pdf_page_tables"):
            record.pop(key, None)
        print(f"[file_tables] {filename}: tables not read directly: {exc}")

    pdf_pages = None
    if ext == "pdf":
        try:
            pdf_pages = pdf_page_count(data)
            record["raw_bytes"] = data  # pages are rendered when they are sent to the model
            record["pdf_total_pages"] = pdf_pages
            record["pdf_page_texts"] = page_texts
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
            "image_count": (pdf_pages or 0) + embedded_images_count,
            "file_table_count": len(record.get("file_tables") or []),
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


def split_text(text: str, limit: int = INLINE_CHAR_LIMIT) -> list[str]:
    """Split text into chunks of at most `limit` chars, on line breaks where possible."""
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


def make_unit(source, text, image_count=0, load_images=None, page=None, source_text=""):
    """The smallest piece of an attachment that can go to the model on its own."""
    return {
        "source": source,
        "text": text,
        "image_count": image_count,
        "load_images": load_images,  # called only when the unit is actually sent
        "page": page,
        "source_text": source_text,  # the document's own text, excluding labels added for the model
    }


def pdf_units(meta):
    name, total = meta["filename"], meta["pdf_total_pages"]
    page_texts = meta.get("pdf_page_texts") or []
    page_tables = meta.get("pdf_page_tables") or {}
    units = []
    for page in range(1, total + 1):
        page_text = page_texts[page - 1] if page <= len(page_texts) else ""
        note = file_tables.pdf_note(meta["file_tables"], page_tables[page]) if page in page_tables else ""
        for i, chunk in enumerate(split_text(page_text) or [""]):
            text = f"### Attached file: {name} (page {page} of {total}, page image attached)\n{chunk}"
            if note and i == 0:
                text += "\n" + note
            own_text = re.sub(r"^--- page \d+ ---\n?", "", chunk)
            if i == 0:
                units.append(make_unit(
                    name, text, image_count=1, page=page, source_text=own_text,
                    load_images=lambda p=page: [render_pdf_page_b64(meta["raw_bytes"], p)],
                ))
            else:
                units.append(make_unit(name, text, page=page, source_text=own_text))
    return units


def document_units(meta):
    name = meta["filename"]
    if "model_text" in meta:
        # Tables read from the file reach the model only as short placeholders, but the
        # full text is still the evidence for whatever the model does extract.
        chunks = split_text(meta["model_text"]) or ["(no text outside the tables)"]
        sources = [meta["text"]] + [""] * (len(chunks) - 1)
    else:
        chunks = split_text(meta["text"]) or ["(no extractable text)"]
        sources = chunks
    units = [
        make_unit(name, f"### Attached file: {name} (section {i} of {len(chunks)})\n{chunk}", source_text=source)
        for i, (chunk, source) in enumerate(zip(chunks, sources), 1)
    ]
    for picture in meta.get("embedded_images") or []:
        units.append(make_unit(
            name, f"### Picture embedded in {name}",
            image_count=1, load_images=lambda b64=picture: [b64],
        ))
    return units


def build_units(attachments):
    """Returns (units, missing_filenames, warnings) for the given attachment refs."""
    units, missing, warnings = [], [], []
    for att in attachments or []:
        file_id = att.get("file_id")
        meta = FILES.get(file_id)
        if not meta:
            missing.append(att.get("filename") or file_id)
        elif meta["kind"] == "image":
            units.append(make_unit(
                meta["filename"], f"### Attached image: {meta['filename']}",
                image_count=1, load_images=lambda b64=meta["base64"]: [b64],
            ))
        elif meta.get("pdf_total_pages"):
            units.extend(pdf_units(meta))
        else:
            if meta.get("pdf_render_error"):
                warnings.append(
                    f"{meta['filename']}: page images could not be rendered, so only its text layer "
                    f"was read ({meta['pdf_render_error']})."
                )
            units.extend(document_units(meta))
    return units, missing, warnings


def pack_units(units):
    """Group units, in order, into parts that fit the per-request image and text limits."""
    parts, current, images, chars = [], [], 0, 0
    for unit in units:
        too_many_images = images + unit["image_count"] > MAX_IMAGES_PER_PART
        too_much_text = chars + len(unit["text"]) > INLINE_CHAR_LIMIT
        if current and (too_many_images or too_much_text):
            parts.append(current)
            current, images, chars = [], 0, 0
        current.append(unit)
        images += unit["image_count"]
        chars += len(unit["text"])
    if current:
        parts.append(current)
    return parts


def describe_units(units):
    """Human-readable coverage, e.g. 'report.pdf pages 7-12, notes.docx'."""
    grouped = {}
    for unit in units:
        grouped.setdefault(unit["source"], []).append(unit["page"])
    labels = []
    for source, pages in grouped.items():
        pages = [p for p in pages if p is not None]
        if not pages:
            labels.append(source)
        elif min(pages) == max(pages):
            labels.append(f"{source} page {pages[0]}")
        else:
            labels.append(f"{source} pages {min(pages)}-{max(pages)}")
    return ", ".join(labels)


def table_label(table, index):
    return (isinstance(table, dict) and table.get("title")) or f"Table {index + 1}"


def normalize_tables(parsed: dict) -> list[str]:
    """Give every row its table's column count without discarding any cell.

    Mutates `parsed` and returns a warning for each table that needed repair. Safe to
    run repeatedly: an already-rectangular table comes through unchanged.
    """
    tables = parsed.get("tables")
    if tables is None:
        return []
    if not isinstance(tables, list):
        parsed["tables"] = []
        return ['Model returned "tables" in an unexpected format, so it could not be read.']

    warnings, clean = [], []
    for index, table in enumerate(tables):
        if not isinstance(table, dict):
            warnings.append(f"Table {index + 1} was in an unexpected format and was skipped.")
            continue

        headers = table.get("headers")
        headers = list(headers) if isinstance(headers, list) else []
        rows = []
        for row in table.get("rows") if isinstance(table.get("rows"), list) else []:
            if isinstance(row, dict):
                row = list(row.values())
            rows.append(list(row) if isinstance(row, list) else [row])

        label = table_label(table, index)
        width = max([len(headers)] + [len(row) for row in rows])
        long_rows = sum(len(row) > len(headers) for row in rows)
        short_rows = sum(len(row) < len(headers) for row in rows)
        if long_rows:
            warnings.append(
                f'"{label}": {long_rows} row(s) had more cells than headers. The extra cells were '
                f'kept under "Extra Column" headers — check that those rows line up.'
            )
        if short_rows:
            warnings.append(
                f'"{label}": {short_rows} row(s) had fewer cells than headers. Blanks were added at '
                f"the end of each — check that those rows line up."
            )

        headers += [f"Extra Column {n}" for n in range(1, width - len(headers) + 1)]
        clean.append({
            **table,
            "headers": headers,
            "rows": [row + [""] * (width - len(row)) for row in rows],
        })

    parsed["tables"] = clean
    return warnings


def same_cells(a, b) -> bool:
    return [str(c).strip().lower() for c in a] == [str(c).strip().lower() for c in b]


def merge_part_results(results: list[dict]) -> dict:
    """Combine per-part replies, joining a table that continues across a part boundary."""
    merged = dict(results[0])
    merged["key_info"] = {}
    merged["tables"] = []
    merged["sections"] = []
    confidences = []

    for result in results:
        for field in ("image_type", "summary", "primary_name", "concise_topic"):
            if not merged.get(field) and result.get(field):
                merged[field] = result[field]
        if isinstance(result.get("confidence"), (int, float)):
            confidences.append(result["confidence"])

        key_info = result.get("key_info")
        if isinstance(key_info, dict):
            for key, value in key_info.items():
                if key not in merged["key_info"]:
                    merged["key_info"][key] = value

        sections = result.get("sections")
        for section in sections if isinstance(sections, list) else []:
            if not isinstance(section, dict):
                continue
            existing = next((s for s in merged["sections"] if s.get("title") == section.get("title")), None)
            items = section.get("items") if isinstance(section.get("items"), list) else []
            if existing:
                existing["items"].extend(items)
            else:
                merged["sections"].append({**section, "items": list(items)})

        for position, table in enumerate(result.get("tables") or []):
            previous = merged["tables"][-1] if merged["tables"] else None
            continues = (
                position == 0
                and previous is not None
                and str(previous.get("title")).strip().lower() == str(table.get("title")).strip().lower()
                and same_cells(previous["headers"], table["headers"])
            )
            if continues:
                previous["rows"].extend(r for r in table["rows"] if not same_cells(r, table["headers"]))
            else:
                merged["tables"].append({**table, "rows": list(table["rows"])})

    if confidences:
        merged["confidence"] = min(confidences)  # the result is only as sure as its weakest part
    return merged


def add_file_tables(parsed, attachments):
    """Puts tables read from the files first, dropping any the model typed out anyway."""
    exact = [t for a in attachments or [] for t in FILES.get(a.get("file_id"), {}).get("file_tables") or []]
    if not exact:
        return
    exact = json.loads(json.dumps(exact))  # the upload keeps its own copy
    model_tables = [
        t for t in parsed.get("tables") or []
        if not any(file_tables.same_table(t, e) for e in exact)
    ]
    parsed["tables"] = exact + model_tables


def build_part_text(message, part, index, count, last_table):
    sections = [message]
    if count > 1:
        sections.append(
            f"(This is part {index} of {count} of the attached content, covering "
            f"{describe_units(part)}. Extract only what is in this part.)"
        )
    if last_table:
        sections.append(
            "(The previous part's last table was titled "
            f"{json.dumps(last_table.get('title'), ensure_ascii=False)} with headers "
            f"{json.dumps(last_table.get('headers'), ensure_ascii=False)}. If this part continues "
            "that table, reuse that exact title and those exact headers.)"
        )
    sections.extend(unit["text"] for unit in part)
    return "\n\n".join(sections)


def load_part_images(part, warnings):
    images = []
    for unit in part:
        if not unit["load_images"]:
            continue
        try:
            images.extend(image for image in unit["load_images"]() if image)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not render {describe_units([unit])}, so only its text was read: {exc}")
    return images


def collect_evidence(units):
    """Read the document independently of the vision model. Never raises: verification is
    an addition to extraction, so a failure here costs evidence, not the result."""
    evidence = SourceEvidence()
    ocr_used = 0
    ocr_skipped = []
    for unit in units:
        own_text = unit["source_text"]
        if len(re.sub(r"\W", "", own_text)) >= MIN_TEXT_LAYER_CHARS or (own_text.strip() and not unit["load_images"]):
            evidence.add_text(own_text, unit["source"])
            continue
        if not unit["load_images"]:
            continue
        if ocr_used >= MAX_OCR_IMAGES:
            ocr_skipped.append(unit)
            continue
        ocr_used += 1
        try:
            import ocr_engine

            for image in unit["load_images"]():
                evidence.add_ocr(ocr_engine.read_image_b64(image), describe_units([unit]), image=image)
        except Exception as exc:  # noqa: BLE001
            evidence.notes.append(f"OCR failed for {describe_units([unit])}, so its values weren't source-checked: {exc}")
    if ocr_skipped:
        evidence.notes.append(f"OCR limit reached, so these weren't source-checked: {describe_units(ocr_skipped)}.")
    return evidence


def verify_and_repair(parsed, evidence_future, warnings, host, model, row_alternatives=None):
    """Verify, fix digits from the model's second choices, re-read what's still flagged, verify again.

    Returns (verification or None, repair_seconds, repair_calls).
    """
    try:
        evidence = evidence_future.result() if evidence_future else SourceEvidence()
        warnings.extend(evidence.notes)
        verification = verify_record(parsed, evidence)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Verification couldn't run, so nothing in this result has been checked: {exc}")
        return None, 0, 0

    def ask_model(system, prompt, image_b64):
        reply = call_model(host, model, [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt, "images": [image_b64]},
        ])
        return extract_json(reply)

    started = time.monotonic()
    repairs, calls = [], 0
    try:
        repairs = digit_repair.repair_tables(parsed, verification, row_alternatives)
        if repairs:
            verification = verify_record(parsed, evidence)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Checking the model's second choices failed, so that step was skipped: {exc}")
    try:
        reread, notes, calls = repair.repair_tables(parsed, verification, evidence, ask_model)
        repairs = repairs + reread
        warnings.extend(notes)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Re-reading flagged rows failed, so they were left as they were: {exc}")
    repair_seconds = round(time.monotonic() - started, 2)

    if repairs:
        verification = verify_record(parsed, evidence)
    verification["repairs"] = repairs
    verification["summary"]["repaired"] = len(repairs)
    for item in repairs:
        table_check = verification["tables"][item["table"]]
        column = parsed["tables"][item["table"]]["headers"].index(item["column"]) if item["column"] in parsed["tables"][item["table"]]["headers"] else None
        if column is None:
            continue
        cell = table_check["cells"][item["row"]][column]
        if item.get("method") == "second_choice_digit":
            # The sum this value was chosen to satisfy can't also be what verifies it.
            if cell["status"] == VERIFIED and not any(note.startswith("matches ") for note in cell["evidence"]):
                cell["status"] = UNCHECKED
                verification["summary"][VERIFIED] -= 1
                verification["summary"][UNCHECKED] += 1
            cell["evidence"].insert(
                0, f'corrected from "{item["before"]}": the second choice for one digit, and the only such change that makes the sums hold'
            )
        else:
            cell["evidence"].insert(0, f'corrected from "{item["before"]}" by a zoomed re-read')
    for table in verification["tables"]:
        # Positions in this run's OCR mean nothing once saved.
        table.pop("row_sources", None)
        table.pop("column_centres", None)
        for line in table["suspected_missing_rows"]:
            warnings.append(f'"{table["title"]}": a row in the document may be missing from the table: {line}')
    return verification, repair_seconds, calls


def request_model(host, model, messages, alternatives=False):
    extra = {"logprobs": True, "top_logprobs": MODEL_ALTERNATIVES} if alternatives else {}
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": MODEL_OPTIONS,
            "keep_alive": MODEL_KEEP_ALIVE,
            **extra,
        },
        headers={"ngrok-skip-browser-warning": "true"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def call_model(host, model, messages):
    return request_model(host, model, messages).get("message", {}).get("content", "")


@app.route("/api/chat", methods=["POST"])
def chat():
    start_time = time.monotonic()
    body = request.get_json(force=True)
    host, model = get_config()
    message = body.get("message", "") or "Analyze the attached content."
    history = body.get("history", [])
    attachments = body.get("attachments", [])

    units, missing, warnings = build_units(attachments)
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

    parts = pack_units(units) or [[]]
    part_count = len(parts)
    if part_count > MAX_PARTS_PER_REQUEST:
        skipped = [unit for part in parts[MAX_PARTS_PER_REQUEST:] for unit in part]
        warnings.append(
            f"This attachment is too large to process in one request, so this result is INCOMPLETE. "
            f"Not processed: {describe_units(skipped)}."
        )
        parts = parts[:MAX_PARTS_PER_REQUEST]
        part_count = len(parts)

    base_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            base_messages.append({"role": role, "content": content})

    # OCR runs alongside the model, which is remote, so the two don't add up to the sum of both.
    processed_units = [unit for part in parts for unit in part]
    executor = ThreadPoolExecutor(max_workers=1)
    evidence_future = executor.submit(collect_evidence, processed_units) if processed_units else None
    executor.shutdown(wait=False)

    results = []
    row_alternatives = {}
    raw_replies = []
    parse_error = None
    request_error = None

    for index, part in enumerate(parts, 1):
        last_tables = results[-1].get("tables") if results else None
        user_msg = {
            "role": "user",
            "content": build_part_text(message, part, index, part_count, last_tables[-1] if last_tables else None),
        }
        images = load_part_images(part, warnings)
        if images:
            if len(images) <= MAX_IMAGES_TO_ENLARGE:
                images = [enlarge_for_model(image) for image in images]
            user_msg["images"] = images

        try:
            # Second choices only help with misread scans and pictures, and add megabytes of reply.
            reply = request_model(host, model, base_messages + [user_msg], alternatives=bool(images))
            raw_content = reply.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as exc:
            # The model is unreachable, so later parts would only wait out the same timeout.
            request_error = exc
            remaining = [unit for later in parts[index - 1:] for unit in later]
            if remaining:
                warnings.append(f"Model request failed, so this result is INCOMPLETE. Not processed: {describe_units(remaining)}.")
            break

        raw_replies.append(strip_think(raw_content))
        try:
            parsed_part = extract_json(raw_content)
            if not isinstance(parsed_part, dict):
                raise ValueError("Expected a JSON object")
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            parse_error = parse_error or str(exc)
            if part_count > 1:
                warnings.append(f"Model reply wasn't valid JSON, so this result is INCOMPLETE. Not processed: {describe_units(part)}.")
            continue

        warnings.extend(normalize_tables(parsed_part))
        results.append(parsed_part)
        row_alternatives.update(digit_repair.by_row(digit_repair.digit_alternatives(raw_content, reply.get("logprobs")), parsed_part))

    if request_error is not None and not results:
        return jsonify({"ok": False, "error": f"Could not reach {host}: {request_error}"}), 502

    parsed = None
    category = None
    suggested_filename = "document_data.json"

    if results:
        parsed = merge_part_results(results)
        parse_error = None
        add_file_tables(parsed, attachments)

        # Extract category identified by the model
        category = str(parsed.get("image_type") or parsed.get("category") or "other").strip().lower()

        # Determine source filename prefix if available
        orig_name = "document"
        if attachments and len(attachments) > 0:
            first_att = attachments[0]
            orig_name = first_att.get("filename") or FILES.get(first_att.get("file_id"), {}).get("filename", "document")

        # Generate dynamic filename: name_concisetopicin2-3words.json
        suggested_filename = generate_suggested_filename(parsed, orig_name)

    extraction_seconds = round(time.monotonic() - start_time, 2)
    verification = None
    repair_seconds = repair_calls = 0
    if parsed and processed_units:
        verification, repair_seconds, repair_calls = verify_and_repair(parsed, evidence_future, warnings, host, model, row_alternatives)

    # History holds the data itself: a re-read may have changed it, and checks would only add noise.
    if parsed and (part_count > 1 or (verification and verification.get("repairs"))):
        visible = json.dumps(parsed, ensure_ascii=False)
    else:
        visible = "\n\n".join(raw_replies)

    if verification:
        parsed["verification"] = verification
    if parsed and warnings:
        parsed["extraction_warnings"] = warnings

    record = None
    if parsed and processed_units:
        try:
            sources = [
                (FILES[a["file_id"]]["filename"], FILES[a["file_id"]]["original"])
                for a in attachments
                if FILES.get(a.get("file_id"), {}).get("original")
            ]
            record = records.save_record(parsed, category, suggested_filename, generate_excel_bytes, sources=sources)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"This result couldn't be saved to Records: {exc}")

    elapsed = round(time.monotonic() - start_time, 2)

    return jsonify(
        {
            "ok": True,
            "raw": visible,
            "parsed": parsed,
            "parse_error": parse_error,
            "elapsed_seconds": elapsed,
            "parts_processed": len(results),
            "warnings": warnings,
            "timing": {
                "extraction_seconds": extraction_seconds,
                "repair_seconds": repair_seconds,
                "repair_calls": repair_calls,
            },
            "record": record,
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


NUMERIC_CELL_RE = re.compile(r"^(-?)(0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{2,3})*,\d{3})(?:\.(\d+))?$")


def clean_excel_text(value) -> str:
    """openpyxl refuses to write control characters, which would fail the whole export."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    return ILLEGAL_CHARACTERS_RE.sub("", str(value))


def excel_cell_value(raw):
    """Returns (value, number_format). Only values that survive as numbers become numbers.

    Everything else stays text exactly as extracted: codes with leading zeros ('00123'),
    ambiguous separators ('12,50'), and anything too long for Excel's 15-digit precision.
    Number formats keep the source's decimal places, so '12.50' still displays as 12.50.
    """
    if raw is None:
        return "", None
    if isinstance(raw, (list, dict)):
        raw = json.dumps(raw, ensure_ascii=False)
    text = clean_excel_text(raw)

    stripped = text.strip()
    match = NUMERIC_CELL_RE.match(stripped)
    if not match:
        return text, None

    sign, whole, decimals = match.groups()
    digits = whole.replace(",", "") + (decimals or "")
    if len(digits) > MAX_EXCEL_DIGITS or (sign and not digits.strip("0")):
        return text, None

    whole_format = "#,##0" if "," in whole else "0"
    plain = stripped.replace(",", "")
    if decimals:
        return float(plain), f"{whole_format}.{'0' * len(decimals)}"
    return int(plain), whole_format


def header_levels(headers) -> list[list[str]]:
    """Recover grouped headers from flat names like 'CET - Intake'.

    A prefix only counts as a group when a neighbouring column shares it, so a lone
    header that happens to contain ' - ' stays whole.
    """
    parts = [[p.strip() for p in str(h).split(" - ")] for h in headers]
    levels = []
    for i, part in enumerate(parts):
        shared = len(part) > 1 and any(
            0 <= j < len(parts) and len(parts[j]) > 1 and parts[j][0] == part[0]
            for j in (i - 1, i + 1)
        )
        levels.append(part if shared else [str(headers[i])])
    return levels


def section_as_table(section: dict, section_index: int) -> dict:
    """Shape a section's entries as a table."""
    items = [i if isinstance(i, dict) else {"value": i} for i in section.get("items") or []]
    keys = list(dict.fromkeys(key for item in items for key in item))
    return {
        "title": section.get("title") or f"Section {section_index + 1}",
        "headers": [key.replace("_", " ").title() for key in keys],
        "rows": [[item.get(key, "") for key in keys] for item in items],
    }


def generate_excel_bytes(parsed: dict, filename: str = "") -> bytes:
    """Build a workbook holding only the document's data, in the document's layout.

    Verification results stay in the record's JSON and the app, never in the workbook.
    """
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    if not isinstance(parsed, dict):
        parsed = {"summary": str(parsed)}

    # The client sends `parsed` back for export, so it can't be assumed to be normalized.
    normalize_tables(parsed)

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

    def write_table(tbl, index):
        """Writes one table to its own sheet."""
        title = clean_excel_text(tbl.get("title") or f"Table {index + 1}")
        safe_title = re.sub(r"[\\/*?:\[\]]", "_", title)[:30].strip() or f"Sheet{index + 1}"
        ws = wb.create_sheet(title=safe_title)
        ws.views.sheetView[0].showGridLines = True

        headers = tbl.get("headers") or []
        rows = tbl.get("rows") or []
        curr_row = 1

        # Title Row Banner
        max_cols = max(len(headers), 1)
        title_cell = ws.cell(row=curr_row, column=1, value=title)
        title_cell.data_type = "s"
        title_cell.font = Font(name="Calibri", size=11, bold=True, color="0F172A")
        title_cell.alignment = Alignment(horizontal="left", vertical="center")
        if max_cols > 1:
            ws.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=max_cols)
        ws.row_dimensions[curr_row].height = 24
        curr_row += 1

        # Grouped Header Rows, as in the source document
        levels = header_levels(headers)
        depth = max((len(level) for level in levels), default=1)
        header_top = curr_row
        for level_index in range(depth):
            ws.row_dimensions[header_top + level_index].height = 32 if level_index == depth - 1 else 22
            for col_idx in range(1, len(headers) + 1):
                level = levels[col_idx - 1]
                text = level[level_index] if level_index < len(level) else None
                cell = ws.cell(row=header_top + level_index, column=col_idx)
                if text is not None:
                    cell.value = clean_excel_text(text)
                    cell.data_type = "s"
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = thin_border

        for col_idx, level in enumerate(levels, 1):
            if len(level) < depth:
                ws.merge_cells(
                    start_row=header_top + len(level) - 1, start_column=col_idx,
                    end_row=header_top + depth - 1, end_column=col_idx,
                )
        for level_index in range(depth - 1):
            col_idx = 0
            while col_idx < len(levels):
                level = levels[col_idx]
                end = col_idx
                if len(level) > level_index + 1:
                    while (
                        end + 1 < len(levels)
                        and len(levels[end + 1]) > level_index + 1
                        and levels[end + 1][:level_index + 1] == level[:level_index + 1]
                    ):
                        end += 1
                    if end > col_idx:
                        ws.merge_cells(
                            start_row=header_top + level_index, start_column=col_idx + 1,
                            end_row=header_top + level_index, end_column=end + 1,
                        )
                col_idx = end + 1
        curr_row = header_top + depth
        first_data_row = curr_row
        ws.freeze_panes = ws.cell(row=first_data_row, column=1)

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
                val, number_format = excel_cell_value(row[col_idx - 1])
                is_num = number_format is not None
                align_horiz = "right" if is_num else "left"

                if col_idx == 1 and (is_num or len(str(val).strip()) <= 4):
                    align_horiz = "center"

                cell = ws.cell(row=curr_row, column=col_idx, value=val)
                if is_num:
                    cell.number_format = number_format
                else:
                    cell.data_type = "s"  # extracted text like "=1+1" must not run as a formula
                cell.font = row_font
                cell.fill = row_fill
                cell.alignment = Alignment(horizontal=align_horiz, vertical="center")
                cell.border = row_border

            curr_row += 1

        # Auto-adjust column widths
        for col_idx in range(1, len(headers) + 1):
            col_letter = get_column_letter(col_idx)
            max_len = len(levels[col_idx - 1][-1])
            for r in range(first_data_row, curr_row):
                c_val = ws.cell(row=r, column=col_idx).value
                if c_val is not None:
                    max_len = max(max_len, len(str(c_val)))
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 11), 60)

    def field_rows():
        """Every non-table value printed in the document."""
        rows = []
        key_info = parsed.get("key_info")
        for key, value in (key_info.items() if isinstance(key_info, dict) else []):
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            rows.append([key.replace("_", " ").title(), "" if value is None else value])
        name = parsed.get("primary_name")
        if name and not any(str(value).strip() == str(name).strip() for _, value in rows):
            rows.insert(0, ["Name", name])
        return rows

    tables = parsed.get("tables") if isinstance(parsed.get("tables"), list) else []
    for t_idx, tbl in enumerate(tables):
        write_table(tbl, t_idx)

    info_rows = field_rows()
    if info_rows:
        write_table({"title": "Document Info", "headers": ["Field", "Value"], "rows": info_rows}, len(wb.sheetnames))

    sections = parsed.get("sections") if isinstance(parsed.get("sections"), list) else []
    for s_idx, section in enumerate(sections):
        if isinstance(section, dict) and section.get("items"):
            write_table(section_as_table(section, s_idx), len(wb.sheetnames))

    # A workbook needs one sheet, so a document with no fields or tables keeps its description.
    if not wb.sheetnames:
        write_table(
            {"title": "Document Info", "headers": ["Field", "Value"], "rows": [["Summary", parsed.get("summary") or ""]]},
            0,
        )

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
    """Save extracted JSON and Excel to the record library."""
    body = request.get_json(force=True) or {}
    parsed = body.get("parsed")
    if not parsed or not isinstance(parsed, dict):
        return jsonify({"ok": False, "error": "Invalid parsed JSON data"}), 400

    category = body.get("category") or str(parsed.get("image_type") or parsed.get("category") or "other").strip().lower()
    try:
        saved = records.save_record(parsed, category, body.get("filename") or generate_suggested_filename(parsed), generate_excel_bytes)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, **saved})


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

app.register_blueprint(records.create_blueprint(generate_excel_bytes))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
