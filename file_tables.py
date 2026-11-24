"""Tables read straight from files that store them as text: Word, Excel, CSV, PowerPoint and typed PDFs.

These are exact, take well under a second, and spare the model from typing them back out,
which is where nearly all of its time goes. The model still reads the rest of the document,
and still reads any table this module can't find, such as one printed in a scan or a picture.
"""

import csv
import datetime
import io
import re

TABLE_ORIGIN = "file"  # marks a table in the result as read from the file, not by the model
MAX_TITLE_CHARS = 200
SAMPLE_ROWS = 5  # rows the model sees from each table, so it can still pick out key details
MAX_SHEET_ROWS = 50_000
NUMBERISH_RE = re.compile(r"^[-+]?[\d,]*\.?\d+%?$")


def clean(value) -> str:
    """A cell's text on one line, as printed."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def make_table(title, headers, rows):
    width = max([len(headers)] + [len(r) for r in rows]) if (headers or rows) else 0
    headers = list(headers) + [""] * (width - len(headers))
    headers = [h or f"Column {i + 1}" for i, h in enumerate(headers)]
    return {
        "title": clean(title)[:MAX_TITLE_CHARS] or "Table",
        "headers": headers,
        "rows": [list(r) + [""] * (width - len(r)) for r in rows],
        "origin": TABLE_ORIGIN,
    }


def drop_empty(rows):
    return [r for r in rows if any(c for c in r)]


def flatten_headers(top, bottom):
    """'Q1' over 'Actual' becomes 'Q1 - Actual'."""
    out = []
    for a, b in zip(top, bottom):
        out.append(f"{a} - {b}" if a and b and a != b else a or b)
    return out


def placeholder(index, table):
    """What the model sees in place of a table it doesn't need to type out."""
    lines = [
        f'[TABLE {index}: "{table["title"]}" has already been extracted exactly from the file '
        f'({len(table["rows"])} rows x {len(table["headers"])} columns). Do NOT include it in "tables". '
        "Headers: " + " | ".join(table["headers"])
    ]
    sample = table["rows"][:SAMPLE_ROWS]
    if sample:
        lines.append("First rows, for context only:" if len(table["rows"]) > SAMPLE_ROWS else "Rows, for context only:")
        lines.extend(" | ".join(row) for row in sample)
    return "\n".join(lines) + "]"


def is_texty(row):
    filled = [c for c in row if c]
    return bool(filled) and not any(NUMBERISH_RE.match(c.replace(" ", "")) for c in filled)


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------

def docx_grid(table):
    """The printed grid. A cell merged across columns gives its text once, with blanks after it;
    a cell merged downwards repeats its text in each row it covers.

    Returns (rows, spans) where spans[r] says whether row r has a cell merged across columns.
    """
    from docx.table import _Cell

    rows, spans = [], []
    for tr in table._tbl.tr_lst:
        row = [""] * (tr.grid_before or 0)
        spanned = False
        for tc in tr.tc_lst:
            span = tc.grid_span or 1
            spanned = spanned or span > 1
            column = len(row)
            if tc.vMerge == "continue":
                # A cell merged downwards applies to every row it covers, as with one author
                # listed once beside several of their papers.
                text = rows[-1][column] if rows and column < len(rows[-1]) else ""
            else:
                text = clean(_Cell(tc, table).text)
            row.append(text)
            row.extend([""] * (span - 1))
        rows.append(row)
        spans.append(spanned)
    return rows, spans


def docx_spread(table, row_index):
    """The same header row with a merged heading repeated over each column it covers."""
    from docx.table import _Cell

    tr = table._tbl.tr_lst[row_index]
    row = [""] * (tr.grid_before or 0)
    for tc in tr.tc_lst:
        text = "" if tc.vMerge == "continue" else clean(_Cell(tc, table).text)
        row.extend([text] * (tc.grid_span or 1))
    return row


def from_docx(data: bytes):
    """Returns (tables, model_text). Short or single-column tables stay in the text."""
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    tables, text = [], []
    last_paragraph = ""
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = clean(Paragraph(child, doc).text)
            if paragraph:
                text.append(paragraph)
                last_paragraph = paragraph
            continue
        if child.tag != qn("w:tbl"):
            continue
        table = Table(child, doc)
        grid, spans = docx_grid(table)
        width = max((len(r) for r in grid), default=0)
        data_rows = drop_empty(grid)
        if width < 2 or len(data_rows) < 2:
            text.extend(" | ".join(c for c in r if c) for r in data_rows)
            continue
        header_rows = 1
        headers = grid[0]
        # A heading merged across columns usually sits over a second header row.
        if spans[0] and len(grid) > 2 and is_texty(grid[1]):
            headers = flatten_headers(docx_spread(table, 0), docx_spread(table, 1))
            header_rows = 2
        rows = drop_empty(grid[header_rows:])
        title = last_paragraph if len(last_paragraph) <= MAX_TITLE_CHARS else ""
        made = make_table(title or f"Table {len(tables) + 1}", headers, rows)
        tables.append(made)
        text.append(placeholder(len(tables), made))
        last_paragraph = ""
    return tables, "\n".join(text)


# ---------------------------------------------------------------------------
# Excel and CSV
# ---------------------------------------------------------------------------

def cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.15g}"
    if isinstance(value, datetime.datetime):
        return value.date().isoformat() if value.time() == datetime.time() else value.isoformat(sep=" ")
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    return clean(value)


def grid_tables(grid, name):
    """Splits a sheet into tables at blank rows, with title lines kept as titles.

    Returns (tables, leftover text lines for the model).
    """
    blocks, current = [], []
    for row in grid:
        if any(row):
            current.append(row)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    # A blank row inside a table: the next block starts with data, not with headers.
    merged = []
    for block in blocks:
        if merged and not is_texty(block[0]) and len(merged[-1]) > 1:
            merged[-1].extend(block)
        else:
            merged.append(block)

    tables, leftover, titles = [], [], []
    for block in merged:
        filled = [[i for i, c in enumerate(r) if c] for r in block]
        lo = min(f[0] for f in filled)
        hi = max(f[-1] for f in filled)
        block = [r[lo:hi + 1] for r in block]
        # Lines with a single entry above a wider table are its title.
        while len(block) > 1 and hi - lo >= 2 and sum(1 for c in block[0] if c) == 1:
            titles.append(next(c for c in block[0] if c))
            block = block[1:]
        if len(block) < 2 or hi - lo < 1:
            lines = [" | ".join(c for c in r if c) for r in block]
            titles.extend(lines)
            continue
        title = " - ".join(titles[-2:]) if titles else (name if not tables else f"{name} ({len(tables) + 1})")
        leftover.extend(titles)
        titles = []
        tables.append(make_table(title, block[0], block[1:]))
    leftover.extend(titles)
    return tables, leftover


def from_xlsx(data: bytes):
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), data_only=True, read_only=len(data) > 20_000_000)
    tables, text = [], []
    for sheet in wb.worksheets:
        grid = []
        for row in sheet.iter_rows(values_only=True):
            if len(grid) >= MAX_SHEET_ROWS:
                break
            grid.append([cell_text(v) for v in row])
        found, leftover = grid_tables(grid, sheet.title)
        text.append(f"--- sheet: {sheet.title} ---")
        text.extend(leftover)
        for table in found:
            tables.append(table)
            text.append(placeholder(len(tables), table))
    return tables, "\n".join(text)


def from_csv(data: bytes):
    rows = [[clean(c) for c in r] for r in csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace")))]
    rows = drop_empty(rows[:MAX_SHEET_ROWS])
    if len(rows) < 2:
        return [], "\n".join(" | ".join(r) for r in rows)
    table = make_table("Table", rows[0], rows[1:])
    return [table], placeholder(1, table)


# ---------------------------------------------------------------------------
# PowerPoint
# ---------------------------------------------------------------------------

def from_pptx(data: bytes):
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    tables, text = [], []
    for number, slide in enumerate(prs.slides, 1):
        text.append(f"--- slide {number} ---")
        title_shape = slide.shapes.title
        slide_title = clean(title_shape.text) if title_shape is not None and title_shape.has_text_frame else ""
        for shape in slide.shapes:
            if shape.has_text_frame and clean(shape.text_frame.text):
                text.append(shape.text_frame.text.strip())
            if not getattr(shape, "has_table", False) or not shape.has_table:
                continue
            grid = [[("" if cell.is_spanned else clean(cell.text)) for cell in row.cells] for row in shape.table.rows]
            grid = drop_empty(grid)
            if len(grid) < 2 or max(len(r) for r in grid) < 2:
                text.extend(" | ".join(c for c in r if c) for r in grid)
                continue
            table = make_table(slide_title or f"Slide {number} table", grid[0], grid[1:])
            tables.append(table)
            text.append(placeholder(len(tables), table))
    return tables, "\n".join(text)


# ---------------------------------------------------------------------------
# Typed PDFs
# ---------------------------------------------------------------------------

def from_pdf(data: bytes):
    """Ruled tables in a PDF's text layer. Returns (tables, {page: [table index]}).

    Only tables drawn with lines are taken, because those are found reliably. Anything else
    is left to the model, which sees the page image.
    """
    import pymupdf

    tables, by_page = [], {}
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, 1):
            if len(page.get_text().strip()) < 20:
                continue  # a scan: the model reads it
            try:
                found = page.find_tables(strategy="lines").tables
            except Exception:  # noqa: BLE001
                continue
            for position, tab in enumerate(found):
                grid = [[clean(c) for c in row] for row in tab.extract()]
                headers = [clean(h) for h in tab.header.names]
                rows = grid if tab.header.external else grid[1:]
                rows = drop_empty(rows)
                if len(headers) < 2 or not rows:
                    continue
                previous = tables[-1] if tables else None
                # A table that runs onto the next page repeats its headers there.
                if (
                    position == 0 and previous is not None and previous["page_end"] == page_number - 1
                    and [h.lower() for h in previous["headers"]] == [h.lower() for h in headers]
                ):
                    previous["rows"].extend(rows)
                    previous["page_end"] = page_number
                    by_page.setdefault(page_number, []).append(len(tables) - 1)
                    continue
                table = make_table(f"Table on page {page_number}", headers, rows)
                table["page_end"] = page_number
                tables.append(table)
                by_page.setdefault(page_number, []).append(len(tables) - 1)
    for table in tables:
        table.pop("page_end")
    return tables, by_page


def pdf_note(tables, indices):
    """Tells the model which tables on a page it doesn't need to type out."""
    lines = [
        f'(Already extracted exactly from the file, so do NOT include in "tables": '
        f'"{tables[i]["title"]}" with headers ' + " | ".join(tables[i]["headers"]) + ")"
        for i in indices
    ]
    return "\n".join(lines)


EXTRACTORS = {"docx": from_docx, "xlsx": from_xlsx, "xlsm": from_xlsx, "csv": from_csv, "pptx": from_pptx}


def same_table(model_table, file_table) -> bool:
    """A table the model typed out anyway, when it was told not to."""
    def norm(row):
        return [clean(c).lower() for c in row]

    if norm(model_table.get("headers") or []) == norm(file_table["headers"]):
        return True
    rows = [norm(r) for r in model_table.get("rows") or []]
    known = {tuple(norm(r)) for r in file_table["rows"]}
    return bool(rows) and sum(tuple(r) in known for r in rows) >= 0.5 * len(rows)
