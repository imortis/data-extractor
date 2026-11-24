"""Checks every extracted value against evidence the vision model didn't produce.

Three independent signals are combined per value:
- Source match: the document's own text layer, or an OCR read of its image.
- Arithmetic: rules discovered from each table's own numbers, with no template or schema.
- Format and logic: impossible dates, out-of-range values, invoices that don't add up.

Values are never changed here. A check can only report what supports or contradicts them.
"""

import collections
import datetime
import difflib
import functools
import itertools
import math
import re

import numpy as np

VERIFIED = "verified"
UNCHECKED = "unchecked"
CONFLICT = "conflict"
SOURCE_INCONSISTENT = "source_inconsistent"  # read correctly, but the document disagrees with itself
STATUSES = (VERIFIED, UNCHECKED, CONFLICT, SOURCE_INCONSISTENT)
FILE_ORIGIN = "file"  # a table read straight from the file, not by the model

MAX_RULE_TERMS = 5
MIN_RULE_SUPPORT = 0.85
MIN_RULE_ROWS = 5
MAX_MINING_ROWS = 200
MAX_COMBINATIONS_PER_TARGET = 200_000
MAX_RULES_PER_TARGET = 3
MIN_OCR_SCORE = 0.5
NEAR_WORD_RATIO = 0.6
MAX_ALIGNMENT_WINDOW = 30
MIN_COLUMN_POSITION_SAMPLES = 3
ROW_SHIFT_OFFSETS = (-3, -2, -1, 1, 2, 3)
MIN_SHIFT_ADVANTAGE = 2
MIN_IMPLICIT_SUBTOTAL_AGREEMENT = 0.8
MAX_CONFIRMED_RULE_FAILURES = 2  # this many failures on source-confirmed rows disproves a rule

BLANK = "\0blank"
WORD_RE = re.compile(r"\d+(?:\.\d+)+(?![^\W_])|[^\W_]+")
NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
TOTAL_LABEL_RE = re.compile(r"\b(?:sub\s*-?\s*)?totals?\b|\bgrand\s+total\b", re.IGNORECASE)
NOT_ADDITIVE_HEADER_RE = re.compile(
    r"%|percent|\brate\b|\bavg\b|average|ratio|\bmean\b|gpa|\bs\.?\s*no\b|\bsl\.?\s*no\b|serial",
    re.IGNORECASE,
)
# What OCR commonly reads in place of a digit.
DIGIT_CONFUSIONS = str.maketrans({"o": "0", "l": "1", "i": "1"})


# ---------------------------------------------------------------------------
# Comparing text
# ---------------------------------------------------------------------------

def words_of(text) -> list[str]:
    """Casefolded words, so 'M.S.P.' == 'msp' and '1,234' == '1234' but '12.50' != '1250'."""
    text = str(text).casefold()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"(?<=[^\W\d_])\.(?=[^\W\d_])", "", text)
    return WORD_RE.findall(text)


def is_number(word: str) -> bool:
    return bool(NUMBER_RE.match(word))


def has_digit(word: str) -> bool:
    return any(ch.isdigit() for ch in word)


def same_word(model_word: str, source_word: str) -> bool:
    if model_word == source_word:
        return True
    return is_number(model_word) and source_word.translate(DIGIT_CONFUSIONS) == model_word


def word_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def to_number(cell):
    words = words_of(cell)
    text = str(cell).strip()
    if len(words) != 1 or not is_number(words[0]) or not re.fullmatch(r"-?[\d,]+(?:\.\d+)?", text):
        return math.nan
    return float(text.replace(",", ""))


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class SourceEvidence:
    """What independent sources say the document contains."""

    def __init__(self):
        self.rows = []
        self.vocabulary = set()
        self.word_stream = " "
        self.sources = []
        self.notes = []

    @property
    def available(self) -> bool:
        return bool(self.vocabulary)

    def _add_row(self, cells, source, place=None):
        words, raws, scores, positions, boxes = [], [], [], [], []
        for text, score, x, *box in cells:
            for word in words_of(text):
                words.append(word)
                raws.append(text)
                scores.append(score)
                positions.append(x)
                boxes.append(box[0] if box else None)
        if words:
            row = {
                "words": words, "raws": raws, "scores": scores, "positions": positions,
                "source": source, **(place or {}),
            }
            if place:
                row["boxes"] = boxes
            self.rows.append(row)
            self.vocabulary.update(words)
            self.word_stream += " ".join(words) + " "

    def add_text(self, text: str, label: str):
        source = f"text layer of {label}"
        if source not in self.sources:
            self.sources.append(source)
        for line in str(text).splitlines():
            self._add_row([(cell, 1.0, None) for cell in line.split(" | ")], source)

    def add_ocr(self, tokens: list[dict], label: str, image=None):
        source = f"OCR of {label}"
        if source not in self.sources:
            self.sources.append(source)
        for row in group_tokens_into_rows(tokens):
            self._add_row(
                [(token["text"], token["score"], (token["box"][0] + token["box"][2]) / 2, token["box"]) for token in row],
                source,
                {
                    "top": min(token["box"][1] for token in row),
                    "bottom": max(token["box"][3] for token in row),
                    "image": image,
                },
            )


def group_tokens_into_rows(tokens: list[dict]) -> list[list[dict]]:
    """Follow each printed line left to right, which tolerates a slightly skewed scan."""
    if not tokens:
        return []
    heights = sorted(t["box"][3] - t["box"][1] for t in tokens)
    tolerance = 0.6 * max(heights[len(heights) // 2], 1)

    rows = []  # each: {"tokens": [...], "y": y centre of its last token}
    for token in sorted(tokens, key=lambda t: t["box"][0]):
        y = (token["box"][1] + token["box"][3]) / 2
        nearest = min(rows, key=lambda r: abs(r["y"] - y), default=None)
        if nearest is not None and abs(nearest["y"] - y) <= tolerance:
            nearest["tokens"].append(token)
            nearest["y"] = y
        else:
            rows.append({"tokens": [token], "y": y})

    rows.sort(key=lambda r: sum((t["box"][1] + t["box"][3]) / 2 for t in r["tokens"]) / len(r["tokens"]))
    return [r["tokens"] for r in rows]


# ---------------------------------------------------------------------------
# Aligning a table with the source, cell by cell
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=500_000)
def substitution_score(model_word: str, source_word: str) -> float:
    if same_word(model_word, source_word):
        return 3
    if model_word == BLANK:
        return -0.5 if has_digit(source_word) else -3
    if has_digit(model_word) and has_digit(source_word):
        return -1
    # The ratio can't exceed this, so most pairs need no comparison at all.
    if 2 * min(len(model_word), len(source_word)) < NEAR_WORD_RATIO * (len(model_word) + len(source_word)):
        return -3
    return 1 if word_ratio(model_word, source_word) >= NEAR_WORD_RATIO else -3


def match_ceiling(model_words, source_counts) -> int:
    """The most words that could agree, whatever the alignment."""
    return sum(min(count, source_counts[word]) for word, count in collections.Counter(model_words).items())


def source_word_counts(source_words):
    """Each source word, and the number it reads as with a letter mistaken for a digit."""
    counts = collections.Counter(source_words)
    for word in source_words:
        translated = word.translate(DIGIT_CONFUSIONS)
        if translated != word:
            counts[translated] += 1
    return counts


def align_words(model_words, source_words):
    """Global alignment. Returns [(model_index, source_index or None)]."""
    gap = -3  # dearer than a mismatch, or a blank cell lets values slide sideways and hide a misread
    n, m = len(model_words), len(source_words)
    scores = np.zeros((n + 1, m + 1))
    scores[:, 0] = np.arange(n + 1) * gap
    scores[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores[i, j] = max(
                scores[i - 1, j - 1] + substitution_score(model_words[i - 1], source_words[j - 1]),
                scores[i - 1, j] + gap,
                scores[i, j - 1] + gap,
            )
    pairs = []
    i, j = n, m
    while i > 0:
        if j > 0 and scores[i, j] == scores[i - 1, j - 1] + substitution_score(model_words[i - 1], source_words[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif scores[i, j] == scores[i - 1, j] + gap:
            pairs.append((i - 1, None))
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def flatten_row(row):
    words, owners = [], []
    for column, cell in enumerate(row):
        cell_words = words_of(cell) if cell is not None else []
        for word in cell_words or [BLANK]:
            words.append(word)
            owners.append(column)
    return words, owners


def match_rows(table_rows, source_rows):
    """Assign table rows to source rows in reading order, maximising words that agree."""
    if not table_rows or not source_rows:
        return {}
    flat = [flatten_row(row)[0] for row in table_rows]
    source_sets = [set(row["words"]) for row in source_rows]
    source_counts = [source_word_counts(row["words"]) for row in source_rows]
    n, m = len(flat), len(source_rows)
    weights = np.zeros((n, m))
    for i, words in enumerate(flat):
        real = [w for w in words if w != BLANK]
        needed = max(2, 0.3 * len(real))
        expected = int(i * m / n)
        for j in range(max(0, expected - MAX_ALIGNMENT_WINDOW), min(m, expected + MAX_ALIGNMENT_WINDOW + 1)):
            if len(set(real) & source_sets[j]) < min(needed, 2) or match_ceiling(real, source_counts[j]) < needed:
                continue
            pairs = align_words(words, source_rows[j]["words"])
            agree = sum(
                1 for a, b in pairs
                if b is not None and words[a] != BLANK and same_word(words[a], source_rows[j]["words"][b])
            )
            if agree >= needed:
                weights[i, j] = agree

    best = np.zeros((n + 1, m + 1))
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            take = best[i + 1, j + 1] + weights[i, j] if weights[i, j] else -1
            best[i, j] = max(take, best[i + 1, j], best[i, j + 1])
    assignment = {}
    i = j = 0
    while i < n and j < m:
        if weights[i, j] and best[i, j] == best[i + 1, j + 1] + weights[i, j]:
            assignment[i] = j
            i, j = i + 1, j + 1
        elif best[i, j] == best[i + 1, j]:
            i += 1
        else:
            j += 1
    return assignment


def check_cell_against_row(cell_pairs, evidence):
    """Returns (matched, contradiction text or None) for one cell's word alignments."""
    contradiction = None
    all_matched = bool(cell_pairs)
    unaligned = []
    for model_word, source_word, raw, score in cell_pairs:
        if source_word is None:
            all_matched = False
            if model_word != BLANK:
                unaligned.append(model_word)
        elif same_word(model_word, source_word):
            continue
        else:
            all_matched = False
            comparable = (
                (model_word == BLANK and has_digit(source_word))
                or (has_digit(model_word) and has_digit(source_word))
                or (model_word != BLANK and word_ratio(model_word, source_word) >= NEAR_WORD_RATIO)
            )
            if comparable and score >= MIN_OCR_SCORE and contradiction is None:
                contradiction = raw

    # Row labels often wrap onto a neighbouring line, so a long word may be found elsewhere.
    if unaligned and contradiction is None and all(len(w) >= 3 and w in evidence.vocabulary for w in unaligned):
        all_matched = all(pair[1] is not None and same_word(pair[0], pair[1]) or pair[0] in unaligned for pair in cell_pairs)
    return all_matched, contradiction


def source_match_table(headers, rows, evidence):
    """Returns ({(row, col): (matched, contradiction, source, x)}, suspected_missing_rows)."""
    results = {}
    assignment = match_rows(rows, evidence.rows)
    for row_index, source_index in assignment.items():
        source = evidence.rows[source_index]
        words, owners = flatten_row(rows[row_index])
        by_cell = {}
        pairs = align_words(words, source["words"])
        alignments = [source_word_index for _, source_word_index in pairs]
        owners_by_model = [owners[model_index] for model_index, _ in pairs]
        for model_index, source_word_index in pairs:
            column = owners[model_index]
            if source_word_index is None:
                by_cell.setdefault(column, []).append((words[model_index], None, None, 0))
            else:
                by_cell.setdefault(column, []).append((
                    words[model_index],
                    source["words"][source_word_index],
                    source["raws"][source_word_index],
                    source["scores"][source_word_index],
                ))
        for column, cell_pairs in by_cell.items():
            matched, contradiction = check_cell_against_row(cell_pairs, evidence)
            aligned = [i for m, c in zip(alignments, owners_by_model) if c == column for i in [m] if i is not None]
            x = source["positions"][aligned[0]] if aligned else None
            results[(row_index, column)] = (matched, contradiction, source["source"], x)

    # Rows with no aligned line can still have long words confirmed from the wider document.
    for row_index, row in enumerate(rows):
        for column, cell in enumerate(row):
            if (row_index, column) in results:
                continue
            cell_words = words_of(cell) if cell is not None else []
            if cell_words and all(len(w) >= 3 and w in evidence.vocabulary for w in cell_words):
                results[(row_index, column)] = (True, None, "document text", None)

    # A numeric line sitting inside the table that the table has no row for.
    missing = []
    if assignment:
        low, high = min(assignment.values()), max(assignment.values())
        used = set(assignment.values())
        numeric_columns = sum(
            1 for c in range(len(headers))
            if sum(not math.isnan(to_number(r[c])) for r in rows) >= 0.5 * len(rows)
        )
        words_at = {j: set(flatten_row(rows[i])[0]) for i, j in assignment.items()}
        for j in range(low + 1, high):
            if j in used:
                continue
            line_words = evidence.rows[j]["words"]
            numbers = [w for w in line_words if is_number(w)]
            if len(numbers) < max(3, 0.5 * numeric_columns):
                continue
            # A cell's text that wrapped onto its own line belongs to the row around it.
            nearby = words_at[max(k for k in used if k < j)] | words_at[min(k for k in used if k > j)]
            if sum(w in nearby for w in line_words) >= 0.8 * len(line_words):
                continue
            # Each word carries its whole cell's text, so a cell would repeat once per word.
            missing.append(" ".join(raw for raw, _ in itertools.groupby(evidence.rows[j]["raws"])))
    return results, missing, assignment


# ---------------------------------------------------------------------------
# Arithmetic discovered from the table itself
# ---------------------------------------------------------------------------

def find_total_rows(headers, rows, matrix):
    """Rows labelled as totals that really do sum the rows above them, in most columns."""
    checks = []
    additive = [c for c in range(len(headers)) if not NOT_ADDITIVE_HEADER_RE.search(str(headers[c]))]
    labelled = [
        r for r, row in enumerate(rows)
        if any(TOTAL_LABEL_RE.search(str(cell)) for cell in row[:3] if not is_number(str(cell)))
    ]
    # A shaded subtotal row may have no label at all, so also go by what it adds up to.
    for row_index in range(2, len(rows)):
        if row_index in labelled:
            continue
        has_text = any(str(cell).strip() and math.isnan(to_number(cell)) for cell in rows[row_index])
        columns = [c for c in additive if not math.isnan(matrix[row_index, c])]
        members = [r for r in range(row_index) if r not in labelled]
        if has_text or len(columns) < 3:
            continue
        agree = sum(np.nansum(matrix[members, c]) == matrix[row_index, c] for c in columns)
        if agree >= MIN_IMPLICIT_SUBTOTAL_AGREEMENT * len(columns):
            labelled.append(row_index)
    labelled.sort()

    previous_total = -1
    for total_row in labelled:
        cumulative = [r for r in range(total_row) if r not in labelled]
        since_last = [r for r in range(previous_total + 1, total_row) if r not in labelled]
        best = None
        for members in (cumulative, since_last):
            if not members:
                continue
            columns = [c for c in additive if not math.isnan(matrix[total_row, c])]
            agree = sum(np.nansum(matrix[members, c]) == matrix[total_row, c] for c in columns)
            if columns and (best is None or agree > best[0]):
                best = (agree, members, columns)
        if best and best[0] >= 0.5 * len(best[2]):
            checks.append({"row": total_row, "members": best[1], "columns": best[2]})
        previous_total = total_row
    return checks


def mine_rules(headers, matrix, row_indices):
    """Find columns that equal the sum of other columns on nearly every row.

    Rejects the coincidences that mostly-zero columns create: every term has to
    actually contribute on at least two rows the rule holds for.
    """
    data = matrix[row_indices][:MAX_MINING_ROWS]
    if len(data) < MIN_RULE_ROWS:
        return []

    def is_serial(column):
        values = column[~np.isnan(column)]
        return len(values) >= 3 and np.all(np.diff(values) == 1)

    columns = [
        c for c in range(matrix.shape[1])
        if np.mean(~np.isnan(data[:, c])) >= 0.5
        and len(set(data[:, c][~np.isnan(data[:, c])])) > 1
        and not is_serial(data[:, c])
    ]

    rules = []
    for target in columns:
        y = data[:, target]
        if len(set(y[~np.isnan(y)])) < 3:
            continue
        others = [c for c in columns if c != target]
        for size in range(2, MAX_RULE_TERMS + 1):
            if math.comb(len(others), size) > MAX_COMBINATIONS_PER_TARGET or len(others) < size:
                break
            combos = np.array(list(itertools.combinations(others, size)))
            terms = data[:, combos]
            sums = terms.sum(axis=2)
            valid = ~np.isnan(sums) & ~np.isnan(y)[:, None]
            hits = valid & (np.abs(sums - y[:, None]) < 1e-9)
            valid_counts = valid.sum(axis=0)
            support = hits.sum(axis=0) / np.maximum(valid_counts, 1)
            contributes = ((terms != 0) & hits[:, :, None]).sum(axis=0).min(axis=1) >= 2
            nontrivial = (hits & (y[:, None] != 0)).sum(axis=0) >= 3
            good = np.where(
                (support >= MIN_RULE_SUPPORT) & (valid_counts >= 0.7 * len(y)) & contributes & nontrivial
            )[0]
            if len(good):
                top = support[good].max()
                for index in [g for g in good if support[g] == top][:MAX_RULES_PER_TARGET]:
                    rules.append({
                        "target": target,
                        "terms": [int(c) for c in combos[index]],
                        "support": round(float(support[index]), 3),
                        "description": f"[{headers[target]}] = " + " + ".join(f"[{headers[int(c)]}]" for c in combos[index]),
                    })
                break
    return rules


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def printed_columns(source, centres):
    """Which words the document prints in each column of one line, from their positions."""
    tokens = {}
    for raw_text, x in zip(source["raws"], source["positions"]):
        if x is not None:
            tokens.setdefault((raw_text, x), None)
    printed = {}
    for raw_text, x in sorted(tokens, key=lambda token: token[1]):
        column = min(centres, key=lambda c: abs(centres[c] - x))
        printed.setdefault(column, []).append(raw_text)
    return {column: " ".join(texts) for column, texts in printed.items()}


def same_text(model_value, printed_value):
    model_words, printed_words = words_of(model_value), words_of(printed_value)
    return len(model_words) == len(printed_words) and all(same_word(a, b) for a, b in zip(model_words, printed_words))


def find_shifted_rows(rows, source_results, row_assignment, evidence, confirmed, contradicted):
    """Rows whose values match the document only after sliding them sideways.

    Returns {row: {column: printed text}}. Each row is compared with what is printed in each
    column, at every small offset. A shift has to beat the row as written by at least two
    cells and explain most of it, so a coincidence or a single misread can't trigger it.
    """
    placed = {}
    for (row, col), result in source_results.items():
        if result[3] is not None and ((row, col) in confirmed or (row, col) in contradicted):
            placed.setdefault(col, []).append(result[3])
    centres = {
        col: float(np.median(values))
        for col, values in placed.items()
        if len(values) >= MIN_COLUMN_POSITION_SAMPLES
    }
    if len(centres) < 2:
        return {}

    shifted = {}
    for row, source_index in row_assignment.items():
        printed = printed_columns(evidence.rows[source_index], centres)
        filled = [col for col, cell in enumerate(rows[row]) if str(cell).strip() and words_of(cell)]
        if len(filled) < 2:
            continue

        def agreement(offset):
            return sum(col + offset in printed and same_text(rows[row][col], printed[col + offset]) for col in filled)

        as_written = agreement(0)
        best_offset = max(ROW_SHIFT_OFFSETS, key=agreement)
        best = agreement(best_offset)
        if best >= as_written + MIN_SHIFT_ADVANTAGE and best >= 0.5 * len(filled):
            shifted[row] = printed
    return shifted


def misplaced_cells(headers, rows, source_results, confirmed):
    """Returns [((row, col), printed_col)] for confirmed values printed in another column."""
    placed = {}
    for (row, col), result in source_results.items():
        x = result[3]
        if x is not None and (row, col) in confirmed and has_digit(str(rows[row][col])):
            placed.setdefault(col, []).append(x)
    centres = {
        col: float(np.median(values))
        for col, values in placed.items()
        if len(values) >= MIN_COLUMN_POSITION_SAMPLES
    }
    if len(centres) < 2:
        return []

    numeric_columns = {
        col for col in range(len(headers))
        if sum(not math.isnan(to_number(row[col])) for row in rows) >= 0.5 * len(rows)
    }
    found = []
    for (row, col), result in source_results.items():
        x = result[3]
        if x is None or (row, col) not in confirmed:
            continue
        if col in numeric_columns and TOTAL_LABEL_RE.search(str(rows[row][col])):
            nearest = min(centres, key=lambda c: abs(centres[c] - x))
            found.append(((row, col), nearest if nearest != col else None))
            continue
        # Long text is centred anywhere in its cell, so only short values can be placed.
        if not has_digit(str(rows[row][col])) and col not in numeric_columns:
            continue
        nearest = min(centres, key=lambda c: abs(centres[c] - x))
        if col not in centres:
            if str(headers[col]).startswith("Extra Column"):
                found.append(((row, col), nearest))
            continue
        if nearest != col and abs(centres[col] - x) > 1.5 * abs(centres[nearest] - x):
            found.append(((row, col), nearest))
    return found


def check_header(header, evidence):
    """Headers can wrap across lines and be flattened from groups, so match them word by word."""
    if not evidence.available:
        return {"status": UNCHECKED, "evidence": []}
    missing = [w for w in words_of(header) if w not in evidence.vocabulary]
    if not missing:
        return {"status": VERIFIED, "evidence": ["every word is in the document"]}
    near = []
    for word in missing:
        close = difflib.get_close_matches(word, evidence.vocabulary, n=1, cutoff=0.6)
        near.append(f'"{word}" (closest: "{close[0]}")' if close else f'"{word}"')
    return {"status": CONFLICT, "evidence": ["not in the document: " + ", ".join(near[:4])]}


def verify_table(table, evidence):
    headers, rows = table["headers"], table["rows"]
    n_rows, n_cols = len(rows), len(headers)
    matrix = np.array(
        [[to_number(cell) for cell in row] for row in rows], dtype=float
    ).reshape(n_rows, n_cols) if n_rows else np.zeros((0, n_cols))

    if table.get("origin") == FILE_ORIGIN:
        # Read from the file itself, so there is nothing to misread. The arithmetic still runs,
        # and a total that doesn't add up is the document's own error.
        source_results = {
            (r, c): (True, None, "the file itself", None)
            for r, row in enumerate(rows) for c, cell in enumerate(row) if words_of(cell)
        }
        missing_rows, row_assignment = [], {}
    elif evidence.available:
        source_results, missing_rows, row_assignment = source_match_table(headers, rows, evidence)
    else:
        source_results, missing_rows, row_assignment = {}, [], {}
    total_checks = find_total_rows(headers, rows, matrix)
    total_rows = {check["row"] for check in total_checks}
    data_rows = [r for r in range(n_rows) if r not in total_rows]
    rules = mine_rules(headers, matrix, data_rows)

    support = {}  # (row, col) -> [evidence text]
    against = {}  # (row, col) -> [evidence text]
    confirmed, contradicted = set(), set()

    for (row, col), (matched, contradiction, source, _) in source_results.items():
        if contradiction is not None:
            contradicted.add((row, col))
            against.setdefault((row, col), []).append(f'{source} reads "{contradiction}"')
        elif matched:
            confirmed.add((row, col))
            support.setdefault((row, col), []).append(f"matches {source}")

    # In a row that slid sideways, each value is judged against what is printed in its column.
    # A value that still lands on the right text is right, even though its neighbours moved.
    shifted_rows = find_shifted_rows(rows, source_results, row_assignment, evidence, confirmed, contradicted)
    for row, printed in shifted_rows.items():
        for col in range(n_cols):
            cell = (row, col)
            value = rows[row][col]
            printed_text = printed.get(col, "")
            if not words_of(value) and not words_of(printed_text):
                continue
            support.pop(cell, None)
            against.pop(cell, None)
            if same_text(value, printed_text):
                contradicted.discard(cell)
                confirmed.add(cell)
                support.setdefault(cell, []).append("matches what the document prints in this column")
            else:
                confirmed.discard(cell)
                contradicted.add(cell)
                against.setdefault(cell, []).append(
                    f'row is shifted: the document prints "{printed_text}" in this column'
                    if printed_text else "row is shifted: the document prints nothing in this column"
                )

    # A value can match the source and still sit under the wrong header, as when a whole
    # row slides one column sideways. Only printed positions can show that.
    for (row, col), target_col in misplaced_cells(headers, rows, source_results, confirmed):
        if row in shifted_rows:
            continue  # already judged column by column above
        confirmed.discard((row, col))
        contradicted.add((row, col))
        support.pop((row, col), None)
        place = f'under "{headers[target_col]}"' if target_col is not None else "away from this column"
        against.setdefault((row, col), []).append(f"the document prints this value {place}")

    column_positions = {}
    for (row, col), result in source_results.items():
        # Contradicted cells count too: a whole column can be misread, and it's still printed there.
        if result[3] is not None and ((row, col) in confirmed or (row, col) in contradicted):
            column_positions.setdefault(col, []).append(result[3])

    inconsistent = set()

    def note_violation(cells, message):
        """If the source confirms every cell, the document itself disagrees with itself.

        The first cell is the one the arithmetic points at: a rule's target or a column's
        total. Its neighbours stay as they are, because each was read correctly.
        """
        if all(cell in confirmed for cell in cells):
            inconsistent.add(cells[0])
            against.setdefault(cells[0], []).append(f"{message}, but every value matches the source")
            return
        if any(cell in contradicted for cell in cells):
            return  # the contradicted cell already explains this; don't accuse its neighbours
        for cell in cells:
            if cell not in confirmed:
                against.setdefault(cell, []).append(message)

    def source_disproves(rule):
        """True when the rule fails on rows where the source confirms every value."""
        cols = [rule["target"], *rule["terms"]]
        disproved = 0
        for row in data_rows:
            values = matrix[row, cols]
            if (
                not np.isnan(values).any()
                and abs(values[0] - values[1:].sum()) >= 1e-9
                and all((row, c) in confirmed for c in cols)
            ):
                disproved += 1
        return disproved >= MAX_CONFIRMED_RULE_FAILURES

    # Row rules.
    rule_results = []
    rejected_rules = [rule["description"] for rule in rules if source_disproves(rule)]
    for rule in rules:
        if rule["description"] in rejected_rules:
            continue
        cols = [rule["target"], *rule["terms"]]
        held = violated = 0
        for row in data_rows:
            values = matrix[row, cols]
            if np.isnan(values).any():
                continue
            cells = [(row, c) for c in cols]
            if abs(values[0] - values[1:].sum()) < 1e-9:
                held += 1
                if values[0] != 0:
                    for cell in cells:
                        support.setdefault(cell, []).append(f"row rule holds: {rule['description']}")
            else:
                violated += 1
                note_violation(
                    cells,
                    f"row rule fails ({values[1:].sum():g} expected, {values[0]:g} found): {rule['description']}",
                )
        rule_results.append({**rule, "rows_held": held, "rows_violated": violated})

    # Total rows.
    total_results = []
    for check in total_checks:
        total_row, members = check["row"], check["members"]
        failed_columns = []
        for col in check["columns"]:
            total = matrix[total_row, col]
            expected = float(np.nansum(matrix[members, col]))
            cells = [(total_row, col)] + [(r, col) for r in members if not math.isnan(matrix[r, col])]
            if expected == total:
                for cell in cells:
                    support.setdefault(cell, []).append("column adds up to its total")
                continue
            failed_columns.append(col)
            blank_members = [(r, col) for r in members if math.isnan(matrix[r, col])]
            if any(cell in contradicted for cell in blank_members):
                continue  # a blank the source says is filled in already explains the difference
            note_violation(cells, f"column adds up to {expected:g}, total says {total:g}")
        total_results.append({
            "row": total_row,
            "members": [int(r) for r in members],
            "columns": [int(c) for c in check["columns"]],
            "failed_columns": failed_columns,
        })

    cells = []
    for row in range(n_rows):
        out_row = []
        for col in range(n_cols):
            cell = (row, col)
            if cell in contradicted or (cell in against and cell not in inconsistent):
                status = CONFLICT
            elif cell in inconsistent:
                status = SOURCE_INCONSISTENT
            elif cell in support:
                status = VERIFIED
            else:
                status = UNCHECKED
            evidence_notes = against.get(cell, []) + list(dict.fromkeys(support.get(cell, [])))
            out_row.append({"status": status, "evidence": evidence_notes[:4]})
        cells.append(out_row)

    return {
        "title": table.get("title"),
        "headers": [check_header(header, evidence) for header in headers],
        "cells": cells,
        "rules": rule_results,
        "rejected_rules": rejected_rules,
        "row_sources": [row_assignment.get(row) for row in range(n_rows)],
        "column_centres": [
            float(np.median(column_positions[col])) if len(column_positions.get(col, [])) >= MIN_COLUMN_POSITION_SAMPLES else None
            for col in range(n_cols)
        ],
        "total_checks": total_results,
        "suspected_missing_rows": missing_rows,
    }


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

DATE_FORMATS = (
    "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
)
DATE_LIKE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$")


def parse_date(value):
    text = str(value).strip()
    for date_format in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    return None


def check_format(key: str, value) -> tuple[list[str], list[str]]:
    """Returns (passed, failed) for checks that follow from the value's own meaning."""
    passed, failed = [], []
    name, text = str(key).lower(), str(value).strip()
    number = to_number(text.rstrip("%"))

    if "date" in name or DATE_LIKE_RE.match(text):
        if parse_date(text):
            passed.append("valid date")
        elif DATE_LIKE_RE.match(text):
            failed.append(f'"{text}" is not a real calendar date')
    if "email" in name or "e_mail" in name:
        (passed if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", text) else failed).append("email format")
    if "phone" in name or "mobile" in name:
        digits = re.sub(r"\D", "", text)
        (passed if 7 <= len(digits) <= 15 else failed).append("phone number length")
    if ("percent" in name or text.endswith("%")) and not math.isnan(number):
        (passed if 0 <= number <= 100 else failed).append("percentage between 0 and 100")
    if ("cgpa" in name or "sgpa" in name) and not math.isnan(number):
        (passed if 0 <= number <= 10 else failed).append("grade point between 0 and 10")
    if re.search(r"(^|_)year($|_)", name) and re.fullmatch(r"\d{4}", text):
        (passed if 1900 <= int(text) <= 2100 else failed).append("plausible year")
    return passed, failed


def check_field_logic(fields: dict) -> dict:
    """Checks across fields. Returns {field_key: [failure text]}."""
    failures = {}

    def find(*names):
        return [k for k in fields if any(n in k.lower() for n in names)]

    # An end date can't come before its start date.
    for start_key in find("start", "from", "issue_date"):
        start_prefix = re.sub(r"(start|from|issue_date).*$", "", start_key.lower())
        for end_key in find("end", "to_date", "expir", "valid_until"):
            if re.sub(r"(end|to_date|expir|valid_until).*$", "", end_key.lower()) != start_prefix:
                continue
            start, end = parse_date(fields[start_key]), parse_date(fields[end_key])
            if start and end and end < start:
                failures.setdefault(end_key, []).append(f"comes before {start_key} ({fields[start_key]})")

    # Invoice arithmetic, where the fields are present.
    subtotal_keys = find("subtotal", "sub_total")
    total_keys = [k for k in find("total", "amount_due", "grand") if k not in subtotal_keys]
    tax_keys = [k for k in find("tax", "gst", "vat") if not k.lower().endswith("rate") and "percent" not in k.lower()]
    if subtotal_keys and total_keys and tax_keys:
        subtotal = to_number(re.sub(r"[^\d.,-]", "", str(fields[subtotal_keys[0]])))
        taxes = [to_number(re.sub(r"[^\d.,-]", "", str(fields[k]))) for k in tax_keys]
        for total_key in total_keys:
            total = to_number(re.sub(r"[^\d.,-]", "", str(fields[total_key])))
            if not any(math.isnan(v) for v in [subtotal, total, *taxes]) and abs(subtotal + sum(taxes) - total) > 0.01:
                failures.setdefault(total_key, []).append(f"subtotal plus tax is {subtotal + sum(taxes):g}, not {total:g}")
    return failures


def source_match_field(value, evidence):
    """Returns (status or None, note). Short numbers can't be confirmed by being on the page."""
    words = words_of(value)
    if not words:
        return None, None
    if all(is_number(w) and len(w) <= 3 for w in words):
        return None, "too short to confirm by searching the document"
    missing = [w for w in words if w not in evidence.vocabulary]
    if not missing:
        # A short number like the "26" in "2025-26" is on nearly every page of numbers.
        if any(is_number(w) and len(w) <= 3 for w in words) and f" {' '.join(words)} " not in evidence.word_stream:
            return None, "its parts appear in the document, but not together"
        return VERIFIED, "found in the document"
    near = []
    for word in missing:
        close = difflib.get_close_matches(word, evidence.vocabulary, n=1, cutoff=0.7)
        near.append(f'"{word}" (closest: "{close[0]}")' if close else f'"{word}"')
    return CONFLICT, "not found in the document: " + ", ".join(near[:4])


def verify_fields(parsed, evidence):
    flat = {}
    if parsed.get("primary_name"):
        flat["primary_name"] = parsed["primary_name"]
    key_info = parsed.get("key_info")
    if isinstance(key_info, dict):
        for key, value in key_info.items():
            flat[f"key_info.{key}"] = value
    sections = parsed.get("sections")
    if isinstance(sections, list):
        for s, section in enumerate(sections):
            items = section.get("items") if isinstance(section, dict) else None
            for i, item in enumerate(items if isinstance(items, list) else []):
                for key, value in (item.items() if isinstance(item, dict) else [("value", item)]):
                    flat[f"sections[{s}].items[{i}].{key}"] = value

    # Logic only applies within one entry: each job has its own start and end date.
    entries = {}
    for path, value in flat.items():
        prefix, _, key = path.rpartition(".")
        if value is not None:
            entries.setdefault(prefix, {})[key] = value
    logic_failures = {}
    for prefix, entry in entries.items():
        for key, failures in check_field_logic(entry).items():
            logic_failures[f"{prefix}.{key}" if prefix else key] = failures

    results = {}
    for path, value in flat.items():
        short_key = path.rsplit(".", 1)[-1]
        if value is None or str(value).strip() == "":
            results[path] = {"value": value, "status": UNCHECKED, "evidence": ["no value extracted"]}
            continue
        if isinstance(value, (dict, list)):
            results[path] = {"value": value, "status": UNCHECKED, "evidence": ["nested value, not checked"]}
            continue

        passed, failed = check_format(short_key, value)
        failed = failed + logic_failures.get(path, [])
        status, note = source_match_field(value, evidence) if evidence.available else (None, None)
        notes = [f"failed: {f}" for f in failed]
        if note:
            notes.append(note)
        notes.extend(f"format ok: {p}" for p in passed)

        if failed or status == CONFLICT:
            final = CONFLICT
        elif status == VERIFIED:
            final = VERIFIED
        else:
            final = UNCHECKED  # a passed format check alone can't show the value was read correctly
        results[path] = {"value": value, "status": final, "evidence": notes}
    return results


# ---------------------------------------------------------------------------
# Whole record
# ---------------------------------------------------------------------------

def verify_record(parsed: dict, evidence: SourceEvidence) -> dict:
    fields = verify_fields(parsed, evidence)
    tables = [verify_table(table, evidence) for table in parsed.get("tables") or []]

    summary = {status: 0 for status in STATUSES}
    for field in fields.values():
        summary[field["status"]] += 1
    for table in tables:
        for header in table["headers"]:
            summary[header["status"]] += 1
        for row in table["cells"]:
            for cell in row:
                summary[cell["status"]] += 1
    summary["total"] = sum(summary[status] for status in STATUSES)

    return {
        "summary": summary,
        "evidence_sources": evidence.sources,
        "notes": evidence.notes,
        "fields": fields,
        "tables": tables,
    }
