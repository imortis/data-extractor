"""Targeted re-reading: rerun only the table cells that verification flagged.

Each flagged cell is cut out from where it is printed, enlarged, outlined and numbered on one
contact sheet, and the model replies with just those values. That costs a few tokens per cell
rather than a whole table, and it works even when the model slid a row sideways, because the
crop comes from the column's printed position and not from the model's first answer.

A new value is only kept when it agrees with the OCR reading at that position and doesn't
break any of the table's arithmetic. On a badly degraded scan, OCR and the model can misread a
digit the same way, so their agreement alone isn't enough to overwrite a value. Anything that
fails these tests stays flagged for a person to review.
"""

import base64
import io

from PIL import Image, ImageDraw, ImageFont

from verification import CONFLICT, same_word, to_number, verify_table, words_of

MAX_CELLS_PER_SHEET = 24
MAX_REPAIR_CALLS = 3
TILE_SCALE = 4
TILE_COLUMNS = 4
TILE_MARGIN = 6
CROP_PADDING_RATIO = 0.5

REPAIR_SYSTEM_PROMPT = (
    "The image is a sheet of numbered tiles, each cut from a scanned table. Read only what is printed "
    "inside the red box of each tile, exactly as printed, ignoring anything outside the box. Keep zeros, "
    'abbreviations, capitals and spelling errors. If the red box is empty, use "". Reply with ONE JSON '
    'object mapping each tile number to its value and nothing else, for example {"1": "27", "2": ""}. /no_think'
)


def _font():
    try:
        return ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        return ImageFont.load_default()


def locate_cell(source, column_centre, neighbour_gap):
    """Finds the printed box for a column in an OCR row. Returns (box, ocr_text or None)."""
    centres = {}
    for word, raw, box in zip(source["words"], source["raws"], source["boxes"]):
        centres.setdefault(tuple(box), (raw, (box[0] + box[2]) / 2))
    nearest = min(centres, key=lambda box: abs(centres[box][1] - column_centre), default=None)
    if nearest is not None and abs(centres[nearest][1] - column_centre) <= neighbour_gap / 2:
        return list(nearest), centres[nearest][0]
    half_width = neighbour_gap * 0.4
    return [column_centre - half_width, source["top"], column_centre + half_width, source["bottom"]], None


def build_sheet(cells):
    """Lays numbered, enlarged tiles out on one image. `cells` need "image" and "box"."""
    pages = {}
    tiles = []
    for number, cell in enumerate(cells, 1):
        page = pages.setdefault(id(cell["image"]), Image.open(io.BytesIO(base64.b64decode(cell["image"]))).convert("RGB"))
        x1, y1, x2, y2 = cell["box"]
        pad_x, pad_y = max((x2 - x1) * CROP_PADDING_RATIO, 8), max((y2 - y1) * CROP_PADDING_RATIO, 6)
        left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        crop = page.crop((left, top, min(page.width, int(x2 + pad_x)), min(page.height, int(y2 + pad_y))))
        crop = crop.resize((crop.width * TILE_SCALE, crop.height * TILE_SCALE), Image.Resampling.LANCZOS)
        ImageDraw.Draw(crop).rectangle(
            [(x1 - left) * TILE_SCALE - 3, (y1 - top) * TILE_SCALE - 3, (x2 - left) * TILE_SCALE + 3, (y2 - top) * TILE_SCALE + 3],
            outline="red", width=3,
        )
        tiles.append((number, crop))

    label_height = 28
    rows = [tiles[i:i + TILE_COLUMNS] for i in range(0, len(tiles), TILE_COLUMNS)]
    width = max(sum(t.width + TILE_MARGIN for _, t in row) for row in rows) + TILE_MARGIN
    height = sum(max(t.height for _, t in row) + label_height + TILE_MARGIN for row in rows) + TILE_MARGIN
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = TILE_MARGIN
    for row in rows:
        x = TILE_MARGIN
        for number, tile in row:
            draw.text((x, y), str(number), fill="black", font=_font())
            sheet.paste(tile, (x, y + label_height))
            x += tile.width + TILE_MARGIN
        y += max(t.height for _, t in row) + label_height + TILE_MARGIN
    out = io.BytesIO()
    sheet.save(out, format="JPEG", quality=95)
    return base64.b64encode(out.getvalue()).decode("ascii")


def flagged_cells(table, check, evidence):
    """The flagged cells that have a printed position to crop from."""
    row_sources = check.get("row_sources") or []
    centres = check.get("column_centres") or []
    known = sorted(c for c in centres if c is not None)
    gaps = [b - a for a, b in zip(known, known[1:]) if b > a]
    typical_gap = sorted(gaps)[len(gaps) // 2] if gaps else 60
    found = []
    for r, row_checks in enumerate(check["cells"]):
        source_index = row_sources[r] if r < len(row_sources) else None
        if source_index is None or "boxes" not in evidence.rows[source_index]:
            continue
        source = evidence.rows[source_index]
        for c, cell_check in enumerate(row_checks):
            if cell_check["status"] != CONFLICT or c >= len(centres) or centres[c] is None:
                continue
            box, ocr_text = locate_cell(source, centres[c], typical_gap)
            found.append({"row": r, "column": c, "box": box, "ocr": ocr_text, "image": source["image"]})
    return found


def agrees_with_ocr(reading, ocr_text):
    read_words, ocr_words = words_of(reading), words_of(ocr_text or "")
    return len(read_words) == len(ocr_words) and all(same_word(a, b) for a, b in zip(read_words, ocr_words))


def arithmetic_failures(table, check):
    """How many of the table's discovered rules and totals its current values break."""
    rows = table["rows"]
    failures = 0
    total_rows = {item["row"] for item in check.get("total_checks") or []}
    for rule in check.get("rules") or []:
        columns = [rule["target"], *rule["terms"]]
        for index, row in enumerate(rows):
            if index in total_rows:
                continue
            values = [to_number(row[c]) if c < len(row) else float("nan") for c in columns]
            if not any(value != value for value in values) and abs(values[0] - sum(values[1:])) >= 1e-9:
                failures += 1
    for item in check.get("total_checks") or []:
        for column in item["columns"]:
            total = to_number(rows[item["row"]][column])
            parts = [to_number(rows[r][column]) for r in item["members"]]
            if total == total and abs(total - sum(p for p in parts if p == p)) >= 1e-9:
                failures += 1
    return failures


def clear_slid_extra_cells(table):
    """Once a slid row is back under its headers, the cell it overflowed into has to be empty."""
    extra = [i for i, h in enumerate(table["headers"]) if str(h).startswith("Extra Column")]
    if not extra:
        return
    real = [i for i in range(len(table["headers"])) if i not in extra]
    for row in table["rows"]:
        for i in extra:
            if str(row[i]).strip() and str(row[real[-1]]).strip() == str(row[i]).strip():
                row[i] = ""
    keep = [i for i in range(len(table["headers"])) if i not in extra or any(str(r[i]).strip() for r in table["rows"])]
    table["headers"] = [table["headers"][i] for i in keep]
    table["rows"] = [[row[i] for i in keep] for row in table["rows"]]


def repair_tables(parsed, verification, evidence, ask_model):
    """Re-reads flagged cells in place. Returns (repairs, notes, model_calls)."""
    repairs, notes = [], []
    calls = 0
    for table_index, table in enumerate(parsed.get("tables") or []):
        if table.get("origin") == "file":
            continue  # read straight from the file: a re-read can only make it worse
        check = verification["tables"][table_index]
        cells = flagged_cells(table, check, evidence)
        if not cells:
            continue

        for start in range(0, len(cells), MAX_CELLS_PER_SHEET):
            if calls >= MAX_REPAIR_CALLS:
                notes.append(f'"{table.get("title")}": re-read limit reached, so some flagged cells were left for review.')
                break
            batch = cells[start:start + MAX_CELLS_PER_SHEET]
            calls += 1
            try:
                reply = ask_model(
                    REPAIR_SYSTEM_PROMPT,
                    f"Read tiles 1 to {len(batch)}. Reply with the JSON object only.",
                    build_sheet(batch),
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(f'"{table.get("title")}": a re-read failed, so those cells were left for review: {exc}')
                continue
            if not isinstance(reply, dict):
                continue

            # A row's corrections stand or fall together: while a slid row is half moved,
            # its arithmetic is meaningless.
            by_row = {}
            for number, cell in enumerate(batch, 1):
                reading = reply.get(str(number))
                if reading is None:
                    continue
                reading = str(reading).strip()
                old = table["rows"][cell["row"]][cell["column"]]
                if reading != str(old).strip() and agrees_with_ocr(reading, cell["ocr"]):
                    by_row.setdefault(cell["row"], []).append((cell["column"], old, reading))

            for row_index, changes in by_row.items():
                before = arithmetic_failures(table, check)
                for column, _, reading in changes:
                    table["rows"][row_index][column] = reading
                if arithmetic_failures(table, check) > before:
                    for column, old, _ in changes:
                        table["rows"][row_index][column] = old
                    continue
                repairs.extend(
                    {
                        "table": table_index,
                        "row": row_index,
                        "column": table["headers"][column],
                        "before": old,
                        "after": reading,
                    }
                    for column, old, reading in changes
                )

        if repairs:
            clear_slid_extra_cells(table)
            verification["tables"][table_index] = verify_table(table, evidence)
    return repairs, notes, calls
