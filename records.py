"""The record library: everything extracted, stored on disk and searchable.

Each record is a set of files in <records dir>/<category>/:
    <name>.json          the extracted data
    <name>.xlsx          its workbook
    <name>.meta.json     group, saved time, and the source files
    <name>.source1.pdf   the original upload
Files are written to a temporary name and then moved into place, so a crash can't leave a
half-written record. Saving identical data twice returns the existing record.
"""

import datetime
import hashlib
import io
import json
import os
import re
from urllib.parse import urlparse

from flask import Blueprint, jsonify, request, send_file, send_from_directory

GROUPS = ("student", "faculty", "other")
MAX_SNIPPETS = 3
SOURCE_NAME_RE = re.compile(r"\.source\d+\.[^.]+$")

STUDENT_WORDS = (
    "student", "usn", "sgpa", "cgpa", "marksheet", "marks card", "grade report", "provisional grade",
    "admission", "hostel", "enrol", "roll no", "semester", "attendance", "scholarship",
)
FACULTY_WORDS = (
    "faculty", "professor", "lecturer", "teaching staff", "staff", "employee", "publication",
    "name of the author", "appointment", "salary", "payroll", "designation", "research paper",
)


def records_dir():
    return os.environ.get("RECORDS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")


def slug(text, fallback="record"):
    return re.sub(r"[^\w\-]+", "_", str(text or "")).strip("_") or fallback


def content_hash(parsed):
    return hashlib.sha256(json.dumps(parsed, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def write_atomic(path, data):
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as f:
        f.write(data)
    os.replace(temporary, path)


def table_list(parsed):
    tables = parsed.get("tables")
    return [t for t in tables if isinstance(t, dict)] if isinstance(tables, list) else []


def classify_group(parsed):
    """Prefers the model's own answer, then counts telling words in the record's labels."""
    owner = str(parsed.get("record_owner") or "").strip().lower()
    if owner in GROUPS:
        return owner
    labels = [
        parsed.get("image_type"), parsed.get("concise_topic"), parsed.get("summary"),
        parsed.get("primary_name"),
    ]
    key_info = parsed.get("key_info")
    if isinstance(key_info, dict):
        labels.extend(key_info.keys())
    for table in table_list(parsed):
        labels.append(table.get("title"))
        labels.extend(table.get("headers") or [])
    text = " ".join(str(label) for label in labels if label).lower().replace("_", " ")
    student = sum(word in text for word in STUDENT_WORDS)
    faculty = sum(word in text for word in FACULTY_WORDS)
    if student > faculty:
        return "student"
    if faculty > student:
        return "faculty"
    return "other"


def find_existing(category_dir, digest):
    if not os.path.isdir(category_dir):
        return None
    for name in os.listdir(category_dir):
        if not name.endswith(".meta.json"):
            continue
        try:
            with open(os.path.join(category_dir, name), encoding="utf-8") as f:
                if json.load(f).get("content_hash") == digest:
                    return name[: -len(".meta.json")]
        except (OSError, ValueError):
            continue
    return None


def unique_base(category_dir, base):
    candidate, number = base, 2
    while os.path.exists(os.path.join(category_dir, f"{candidate}.json")):
        candidate = f"{base}_{number}"
        number += 1
    return candidate


def save_record(parsed, category, filename, generate_excel, sources=None, excel_bytes=None):
    """Stores one record and returns its details. `sources` is [(filename, bytes)]."""
    category_slug = slug(category or parsed.get("image_type") or "other", "other").lower()
    category_dir = os.path.join(records_dir(), category_slug)
    os.makedirs(category_dir, exist_ok=True)

    digest = content_hash(parsed)
    existing = find_existing(category_dir, digest)
    base = existing or unique_base(category_dir, slug(os.path.splitext(os.path.basename(filename or ""))[0]))
    prefix = os.path.join(category_dir, base)

    if not existing:
        write_atomic(f"{prefix}.json", json.dumps(parsed, indent=2, ensure_ascii=False).encode("utf-8"))
        write_atomic(f"{prefix}.xlsx", excel_bytes if excel_bytes is not None else generate_excel(parsed))
        saved_sources = []
        for number, (source_name, source_bytes) in enumerate(sources or [], 1):
            extension = slug(os.path.splitext(source_name)[1].lstrip("."), "bin").lower()
            stored = f"{base}.source{number}.{extension}"
            write_atomic(os.path.join(category_dir, stored), source_bytes)
            saved_sources.append({"stored": stored, "original_name": source_name})
        meta = {
            "group": classify_group(parsed),
            "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "content_hash": digest,
            "sources": saved_sources,
        }
        write_atomic(f"{prefix}.meta.json", json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))

    relative = os.path.relpath(prefix, os.path.dirname(records_dir())).replace("\\", "/")
    return {
        "id": record_id(category_slug, base),
        "filename": f"{base}.json",
        "saved_path": f"{relative}.json",
        "excel_saved_path": f"{relative}.xlsx",
        "already_saved": bool(existing),
    }


def record_id(category_slug, base):
    return hashlib.sha1(f"{category_slug}/{base}".encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Reading the library
# ---------------------------------------------------------------------------

_cache = {}  # json path -> (signature, record)


def load_record(category_slug, base):
    prefix = os.path.join(records_dir(), category_slug, base)
    json_path = f"{prefix}.json"
    meta_path = f"{prefix}.meta.json"
    signature = (
        os.path.getmtime(json_path),
        os.path.getmtime(meta_path) if os.path.exists(meta_path) else 0,
    )
    cached = _cache.get(json_path)
    if cached and cached[0] == signature:
        return cached[1]

    with open(json_path, encoding="utf-8") as f:
        parsed = json.load(f)
    if not isinstance(parsed, dict):
        return None
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    tables = table_list(parsed)
    title = parsed.get("primary_name") or base.replace("_", " ")
    verification = parsed.get("verification") if isinstance(parsed.get("verification"), dict) else {}

    # Everything searchable, labelled by where it came from.
    segments = [("Name", str(title), 5), ("File", base.replace("_", " "), 4)]
    if parsed.get("concise_topic"):
        segments.append(("Topic", str(parsed["concise_topic"]), 4))
    segments.append(("Type", str(parsed.get("image_type") or category_slug).replace("_", " "), 3))
    if parsed.get("summary"):
        segments.append(("Summary", str(parsed["summary"]), 2))
    key_info = parsed.get("key_info")
    for key, value in (key_info.items() if isinstance(key_info, dict) else []):
        segments.append((key.replace("_", " ").title(), str(value), 3))
    sections = parsed.get("sections")
    for section in sections if isinstance(sections, list) else []:
        if not isinstance(section, dict):
            continue
        for item in section.get("items") or []:
            text = " | ".join(str(v) for v in item.values()) if isinstance(item, dict) else str(item)
            segments.append((str(section.get("title") or "Section"), text, 2))
    for table in tables:
        table_title = str(table.get("title") or "Table")
        segments.append((table_title, " | ".join(str(h) for h in table.get("headers") or []), 1))
        for row in table.get("rows") or []:
            cells = row if isinstance(row, list) else [row]
            segments.append((table_title, " | ".join(str(c) for c in cells if str(c).strip()), 1))

    record = {
        "id": record_id(category_slug, base),
        "base": base,
        "category": category_slug,
        "group": meta.get("group") if meta.get("group") in GROUPS else classify_group(parsed),
        "title": str(title),
        "topic": parsed.get("concise_topic"),
        "summary": parsed.get("summary"),
        "saved_at": meta.get("saved_at") or datetime.datetime.fromtimestamp(signature[0]).astimezone().isoformat(timespec="seconds"),
        "table_count": len(tables),
        "row_count": sum(len(t.get("rows") or []) for t in tables),
        "trust": verification.get("summary"),
        "has_excel": os.path.exists(f"{prefix}.xlsx"),
        "sources": meta.get("sources") or [],
        "warnings": len(parsed.get("extraction_warnings") or []),
        "_segments": [(where, text, text.casefold(), weight) for where, text, weight in segments],
    }
    _cache[json_path] = (signature, record)
    return record


def all_records():
    root = records_dir()
    if not os.path.isdir(root):
        return []
    records = []
    for category_slug in sorted(os.listdir(root)):
        category_dir = os.path.join(root, category_slug)
        if not os.path.isdir(category_dir):
            continue
        for name in os.listdir(category_dir):
            if not name.endswith(".json") or name.endswith(".meta.json") or SOURCE_NAME_RE.search(name):
                continue
            try:
                record = load_record(category_slug, name[: -len(".json")])
            except (OSError, ValueError):
                continue  # a damaged file must not take the whole library down with it
            if record:
                records.append(record)
    return records


def find_record(identifier):
    return next((r for r in all_records() if r["id"] == identifier), None)


def public(record, snippets=None):
    data = {k: v for k, v in record.items() if not k.startswith("_")}
    data["snippets"] = snippets or []
    return data


def search_records(query, group=None, category=None):
    terms = str(query or "").casefold().replace(",", "").split()
    results = []
    for record in all_records():
        if group and record["group"] != group:
            continue
        if category and record["category"] != category:
            continue
        if not terms:
            results.append((0, record, []))
            continue

        def term_in(term, text):
            return term in text or term in text.replace(",", "")

        haystack = " ".join(segment[2] for segment in record["_segments"])
        if not all(term_in(term, haystack) for term in terms):
            continue

        score = 0
        hits = []
        for where, text, folded, weight in record["_segments"]:
            matched = sum(term_in(term, folded) for term in terms)
            if matched:
                score += matched * weight
                hits.append((matched, weight, where, text))
        hits.sort(key=lambda hit: (-hit[0], -hit[1]))
        # The card's heading already shows these, so a snippet would only repeat them.
        inner_hits = [hit for hit in hits if hit[2] not in ("Name", "File", "Topic", "Type")]
        snippets = [{"where": where, "text": text[:280]} for _, _, where, text in inner_hits[:MAX_SNIPPETS]]
        results.append((score, record, snippets))

    results.sort(key=lambda item: (-item[0], item[1]["saved_at"]), reverse=False)
    if not terms:
        results.sort(key=lambda item: item[1]["saved_at"], reverse=True)
    return [public(record, snippets) for _, record, snippets in results]


# ---------------------------------------------------------------------------
# Importing workbooks and JSON that were extracted earlier
# ---------------------------------------------------------------------------

def parsed_from_workbook(data, filename):
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), data_only=True)
    tables = []
    for sheet in workbook.worksheets:
        if sheet.title in ("Checks", "Extraction Warnings", "Document Info"):
            continue
        rows = [["" if c is None else c for c in row] for row in sheet.iter_rows(values_only=True)]
        rows = [row for row in rows if any(str(c).strip() for c in row)]
        if not rows:
            continue
        # The header is the fullest of the first few rows; a title banner has one filled cell.
        header_index = max(range(min(5, len(rows))), key=lambda i: sum(bool(str(c).strip()) for c in rows[i]))
        filled = [str(c).strip() for row in rows[:header_index] for c in row if str(c).strip()]
        tables.append({
            "title": filled[0] if filled else sheet.title,
            "headers": [str(c) for c in rows[header_index]],
            "rows": [[str(c) for c in row] for row in rows[header_index + 1:]],
        })
    stem = os.path.splitext(os.path.basename(filename))[0].replace("_", " ")
    return {
        "image_type": "spreadsheet",
        "primary_name": stem,
        "concise_topic": "Imported Workbook",
        "summary": f"Imported from {os.path.basename(filename)}.",
        "key_info": {},
        "sections": [],
        "tables": tables,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def same_origin():
    """Blocks other websites from changing this local library from the user's browser."""
    origin = request.headers.get("Origin")
    return not origin or urlparse(origin).netloc == request.host


def create_blueprint(generate_excel):
    blueprint = Blueprint("records", __name__)

    @blueprint.route("/records")
    def records_page():
        return send_from_directory("templates", "records.html")

    @blueprint.route("/api/records")
    def list_records():
        everything = all_records()
        counts = {group: sum(r["group"] == group for r in everything) for group in GROUPS}
        counts["all"] = len(everything)
        return jsonify({
            "ok": True,
            "records": search_records(
                request.args.get("q", ""),
                request.args.get("group") or None,
                request.args.get("category") or None,
            ),
            "counts": counts,
            "categories": sorted({r["category"] for r in everything}),
        })

    @blueprint.route("/api/records/<identifier>")
    def get_record(identifier):
        record = find_record(identifier)
        if not record:
            return jsonify({"ok": False, "error": "Record not found"}), 404
        path = os.path.join(records_dir(), record["category"], f"{record['base']}.json")
        with open(path, encoding="utf-8") as f:
            parsed = json.load(f)
        return jsonify({"ok": True, "record": public(record), "parsed": parsed})

    @blueprint.route("/api/records/<identifier>/file/<kind>")
    def get_file(identifier, kind):
        record = find_record(identifier)
        if not record:
            return jsonify({"ok": False, "error": "Record not found"}), 404
        category_dir = os.path.abspath(os.path.join(records_dir(), record["category"]))
        if kind in ("json", "xlsx"):
            stored, download = f"{record['base']}.{kind}", f"{record['base']}.{kind}"
        else:
            match = re.fullmatch(r"source(\d+)", kind)
            sources = record["sources"]
            if not match or not 1 <= int(match.group(1)) <= len(sources):
                return jsonify({"ok": False, "error": "File not found"}), 404
            source = sources[int(match.group(1)) - 1]
            stored, download = source["stored"], source["original_name"]
        path = os.path.join(category_dir, stored)
        if not os.path.isfile(path):
            return jsonify({"ok": False, "error": "File not found"}), 404
        return send_file(path, as_attachment=True, download_name=os.path.basename(download))

    @blueprint.route("/api/records/<identifier>/group", methods=["POST"])
    def set_group(identifier):
        if not same_origin():
            return jsonify({"ok": False, "error": "Request from another site was refused"}), 403
        record = find_record(identifier)
        body = request.get_json(silent=True) or {}
        group = body.get("group")
        if not record:
            return jsonify({"ok": False, "error": "Record not found"}), 404
        if group not in GROUPS:
            return jsonify({"ok": False, "error": "Unknown group"}), 400
        meta_path = os.path.join(records_dir(), record["category"], f"{record['base']}.meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        meta["group"] = group
        meta.setdefault("saved_at", record["saved_at"])
        write_atomic(meta_path, json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))
        return jsonify({"ok": True, "group": group})

    @blueprint.route("/api/records/import", methods=["POST"])
    def import_record():
        if not same_origin():
            return jsonify({"ok": False, "error": "Request from another site was refused"}), 403
        upload = request.files.get("file")
        if not upload or not upload.filename:
            return jsonify({"ok": False, "error": "No file provided"}), 400
        data = upload.read()
        extension = os.path.splitext(upload.filename)[1].lower()
        try:
            if extension in (".xlsx", ".xlsm"):
                parsed = parsed_from_workbook(data, upload.filename)
                saved = save_record(
                    parsed, "imported", upload.filename, generate_excel,
                    excel_bytes=data if extension == ".xlsx" else None,
                )
            elif extension == ".json":
                parsed = json.loads(data.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("The JSON file must hold one object")
                saved = save_record(parsed, parsed.get("image_type"), upload.filename, generate_excel)
            else:
                return jsonify({"ok": False, "error": "Only .xlsx, .xlsm and .json can be imported"}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"Could not import {upload.filename}: {exc}"}), 400
        return jsonify({"ok": True, **saved})

    return blueprint
