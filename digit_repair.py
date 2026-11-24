"""Repairing misread digits from the model's own second choices, with no extra model call.

The first extraction asks the model for the alternatives it weighed at each token. Digits are
single tokens, so every digit in a table cell comes with the other digits the model nearly wrote.
When a row rule or a column total fails, those alternatives are searched for a one-digit change
that makes the arithmetic hold, and it is applied only if exactly one such change exists.

The search needs three independent things to line up: the model was unsure of that digit, the
new digit makes the sum exact, and nothing else could. Values the source confirms are never
touched, so a total the document itself prints wrongly stays as printed. A corrected value isn't
reported as verified on the strength of the sum it was chosen to satisfy.
"""

import itertools
import json
import re

from verification import NOT_ADDITIVE_HEADER_RE, TOTAL_LABEL_RE, to_number

# How far below the chosen digit an alternative may be, as a difference in log probability.
MAX_ALTERNATIVE_GAP = 3.0
SUM_TOLERANCE = 1e-9
MIN_TOTAL_COLUMNS = 3  # an unlabelled row needs this many numbers to be taken for a subtotal
MIN_LABELLED_TOTAL_SHARE = 0.5  # of a total row's columns must add up, directly or with one alternative
MIN_UNLABELLED_TOTAL_SHARE = 0.8
MIN_TOLERANT_RULE_ROWS = 5
MIN_EXACT_RULE_SHARE = 0.5  # of rows must obey a rule exactly as read
MIN_TOLERANT_RULE_SHARE = 0.85  # of rows must obey it once alternatives are allowed
SUM_HEADER_RE = re.compile(r"\b(?:sum|total)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
LITERAL_RE = re.compile(r"true|false|null")
WHITESPACE_RE = re.compile(r"\s*")


def _leaves(text, position, path, found):
    """Walks JSON from `position`, recording each scalar's (start, end, value) by its path."""
    position = WHITESPACE_RE.match(text, position).end()
    char = text[position]
    if char == "{":
        position = WHITESPACE_RE.match(text, position + 1).end()
        if text[position] == "}":
            return position + 1
        while True:
            key, position = json.decoder.scanstring(text, WHITESPACE_RE.match(text, position).end() + 1)
            position = WHITESPACE_RE.match(text, position).end() + 1  # past the colon
            position = _leaves(text, position, path + (key,), found)
            position = WHITESPACE_RE.match(text, position).end()
            if text[position] == "}":
                return position + 1
            position += 1
    if char == "[":
        position = WHITESPACE_RE.match(text, position + 1).end()
        if text[position] == "]":
            return position + 1
        index = 0
        while True:
            position = _leaves(text, position, path + (index,), found)
            position = WHITESPACE_RE.match(text, position).end()
            if text[position] == "]":
                return position + 1
            position += 1
            index += 1
    if char == '"':
        value, end = json.decoder.scanstring(text, position + 1)
        found[path] = (position + 1, end - 1, value)
        return end
    match = NUMBER_RE.match(text, position) or LITERAL_RE.match(text, position)
    if not match:
        raise ValueError(f"Unexpected JSON at {position}")
    found[path] = (match.start(), match.end(), match.group())
    return match.end()


def digit_alternatives(content, logprobs):
    """Maps (table, row, column) to the cell's text and the alternatives for each of its digits."""
    if not logprobs or "".join(entry["token"] for entry in logprobs) != content:
        return {}

    by_offset, offset = {}, 0
    for entry in logprobs:
        if len(entry["token"]) == 1 and entry["token"].isascii() and entry["token"].isdigit():
            by_offset[offset] = entry
        offset += len(entry["token"])

    think_end = content.find("</think>")
    start = content.find("{", think_end + 1 if think_end != -1 else 0)
    leaves = {}
    try:
        _leaves(content, start, (), leaves)
    except (ValueError, IndexError):
        return {}

    cells = {}
    for path, (begin, end, value) in leaves.items():
        if len(path) != 5 or path[0] != "tables" or path[2] != "rows":
            continue
        if not all(isinstance(path[i], int) for i in (1, 3, 4)) or content[begin:end] != value:
            continue  # escapes would put the text out of step with the reply's offsets
        digits = {}
        for index, char in enumerate(value):
            entry = by_offset.get(begin + index)
            if entry is None or entry["token"] != char:
                continue
            options = [
                (other["token"], entry["logprob"] - other["logprob"])
                for other in entry.get("top_logprobs") or []
                if other["token"] != char and len(other["token"]) == 1 and other["token"].isascii() and other["token"].isdigit()
            ]
            if options:
                digits[index] = options
        if digits:
            cells[(path[1], path[3], path[4])] = {"text": value, "digits": digits}
    return cells


def by_row(cells, parsed):
    """Re-keys alternatives by row object, because merging parts renumbers rows but keeps them.

    The row is stored with them so that its id can't be reused while the map is alive.
    """
    out = {}
    tables = parsed.get("tables") or []
    for (table_index, row_index, column), cell in cells.items():
        if table_index >= len(tables) or row_index >= len(tables[table_index]["rows"]):
            continue
        row = tables[table_index]["rows"][row_index]
        if column < len(row) and str(row[column]) == cell["text"]:
            out.setdefault(id(row), (row, {}))[1][column] = cell
    return out


def _is_number(cell):
    value = to_number(cell)
    return value == value


def _holds(equation, rows):
    target, terms = equation
    values = [to_number(rows[r][c]) for r, c in terms]
    return abs(to_number(rows[target[0]][target[1]]) - sum(v for v in values if v == v)) < SUM_TOLERANCE


def _changes_that_fix(equation, rows, alternatives, max_gap):
    """The one-digit changes, from the model's alternatives, that would make this sum hold."""
    found = []
    for r, c in [equation[0], *equation[1]]:
        entry = alternatives.get(id(rows[r]))
        cell = entry[1].get(c) if entry else None
        before = rows[r][c]
        if cell is None or str(before) != cell["text"]:
            continue
        for index, choices in cell["digits"].items():
            for digit, gap in choices:
                if gap <= max_gap:
                    rows[r][c] = before[:index] + digit + before[index + 1:]
                    if _holds(equation, rows):
                        found.append((r, c, rows[r][c]))
                    rows[r][c] = before
    return found


def _tolerant_count(equations, rows, alternatives, max_gap):
    """How many sums hold as read, and how many more one alternative digit would make hold."""
    exact = sum(_holds(e, rows) for e in equations)
    fixable = sum(not _holds(e, rows) and bool(_changes_that_fix(e, rows, alternatives, max_gap)) for e in equations)
    return exact, fixable


def _find_total_rows(table, alternatives, max_gap):
    """Total rows, judged with tolerance: misreads shouldn't hide the sums that would expose them."""
    headers, rows = table["headers"], table["rows"]
    additive = [c for c in range(len(headers)) if not NOT_ADDITIVE_HEADER_RE.search(str(headers[c]))]
    labelled = [
        r for r, row in enumerate(rows)
        if any(TOTAL_LABEL_RE.search(str(cell)) for cell in row[:3] if not _is_number(cell))
    ]
    totals, previous = [], -1
    candidates = sorted(set(labelled) | {r for r in range(2, len(rows)) if not any(str(c).strip() and not _is_number(c) for c in rows[r])})
    accepted = set()
    for row_index in candidates:
        columns = [c for c in additive if _is_number(rows[row_index][c])]
        if not columns or (row_index not in labelled and len(columns) < MIN_TOTAL_COLUMNS):
            continue
        cumulative = [r for r in range(row_index) if r not in accepted and r not in labelled]
        since_last = [r for r in range(previous + 1, row_index) if r not in accepted and r not in labelled]
        best = None
        for members in (cumulative, since_last):
            if not members:
                continue
            equations = [((row_index, c), [(m, c) for m in members]) for c in columns]
            exact, fixable = _tolerant_count(equations, rows, alternatives, max_gap)
            if best is None or exact + fixable > best[0]:
                best = (exact + fixable, exact, equations)
        needed = MIN_LABELLED_TOTAL_SHARE if row_index in labelled else MIN_UNLABELLED_TOTAL_SHARE
        if best and best[0] >= needed * len(columns) and (row_index in labelled or best[1] >= 1):
            totals.append(best[2])
            accepted.add(row_index)
            previous = row_index
    return accepted, [e for equations in totals for e in equations]


def _find_row_rules(table, skip_rows, alternatives, max_gap):
    """Row sums that hold on most rows as read and on nearly all with one alternative digit."""
    headers, rows = table["headers"], table["rows"]
    data_rows = [r for r in range(len(rows)) if r not in skip_rows]
    if len(data_rows) < MIN_TOLERANT_RULE_ROWS:
        return []

    def usable(column):
        values = [to_number(rows[r][column]) for r in data_rows if _is_number(rows[r][column])]
        serial = len(values) >= 3 and all(b - a == 1 for a, b in zip(values, values[1:]))
        return len(values) >= 0.7 * len(data_rows) and len(set(values)) > 1 and not serial and not NOT_ADDITIVE_HEADER_RE.search(str(headers[column]))

    columns = [c for c in range(len(headers)) if usable(c)]
    out = []
    for target in columns:
        others = [c for c in columns if c != target]
        options = [combo for size in (2, 3) for combo in itertools.combinations(others, size)]
        # A column headed Sum or Total usually adds the columns just before it, however many there are.
        if SUM_HEADER_RE.search(str(headers[target])):
            left = [c for c in others if c < target]
            options += [tuple(left[i:]) for i in range(len(left) - 3)]
        qualifying = []
        for terms in options:
            equations = [
                ((r, target), [(r, t) for t in terms])
                for r in data_rows
                if _is_number(rows[r][target]) and all(_is_number(rows[r][t]) for t in terms)
            ]
            if len(equations) < 0.7 * len(data_rows):
                continue
            exact = sum(_holds(e, rows) for e in equations)
            if exact < max(3, MIN_EXACT_RULE_SHARE * len(equations)):
                continue
            _, fixable = _tolerant_count(equations, rows, alternatives, max_gap)
            contributes = all(
                sum(to_number(rows[e[0][0]][t]) != 0 for e in equations if _holds(e, rows)) >= 2 for t in terms
            )
            if contributes and exact + fixable >= MIN_TOLERANT_RULE_SHARE * len(equations):
                qualifying.append((exact, equations))
        if qualifying:
            top = max(item[0] for item in qualifying)
            best = [item for item in qualifying if item[0] == top]
            # Two rules that fit equally well mean the table doesn't say which is real.
            if len(best) == 1:
                out.extend(best[0][1])
    return out


def _equations(table, check, alternatives=None, max_gap=MAX_ALTERNATIVE_GAP):
    """Every sum the table obeys, as (cell that should equal the sum, cells to add).

    Verification's rules are strict, so that they can vouch for values. Here misreads are the
    very thing being looked for, so a sum also counts if the model's alternatives could explain
    where it fails. Otherwise the worst tables would have no sums to repair them with.
    """
    rows = table["rows"]
    total_rows = {item["row"] for item in check.get("total_checks") or []}
    equations = []
    for rule in check.get("rules") or []:
        columns = [rule["target"], *rule["terms"]]
        for index, row in enumerate(rows):
            if index in total_rows or any(column >= len(row) or not _is_number(row[column]) for column in columns):
                continue
            equations.append(((index, rule["target"]), [(index, term) for term in rule["terms"]]))
    for item in check.get("total_checks") or []:
        for column in item["columns"]:
            if _is_number(rows[item["row"]][column]):
                equations.append(((item["row"], column), [(member, column) for member in item["members"]]))

    if alternatives:
        found_totals, total_equations = _find_total_rows(table, alternatives, max_gap)
        known = {(target, tuple(terms)) for target, terms in equations}
        for equation in total_equations + _find_row_rules(table, total_rows | found_totals, alternatives, max_gap):
            if (equation[0], tuple(equation[1])) not in known:
                known.add((equation[0], tuple(equation[1])))
                equations.append(equation)
    return equations


def _touchable(cell_check):
    """A value the source confirms is right, and a slid row needs moving rather than re-digiting."""
    return not any(
        note.startswith("matches ") or "row is shifted" in note or "prints this value" in note
        for note in cell_check.get("evidence") or []
    )


def repair_table(table, check, alternatives, max_gap=MAX_ALTERNATIVE_GAP):
    """Applies each one-digit correction the arithmetic singles out. Returns [(row, column, before, after)]."""
    rows = table["rows"]
    equations = _equations(table, check, alternatives, max_gap)
    containing = {}
    for equation in equations:
        for cell in [equation[0], *equation[1]]:
            containing.setdefault(cell, []).append(equation)

    changes, changed = [], set()
    progress = True
    while progress:
        progress = False
        for equation in equations:
            if _holds(equation, rows):
                continue
            options = []
            for r, c in [equation[0], *equation[1]]:
                entry = alternatives.get(id(rows[r]))
                cell = entry[1].get(c) if entry else None
                if cell is None or (r, c) in changed or str(rows[r][c]) != cell["text"] or not _touchable(check["cells"][r][c]):
                    continue
                before = rows[r][c]
                failing_before = [e for e in containing[(r, c)] if not _holds(e, rows)]
                for index, choices in cell["digits"].items():
                    for digit, gap in choices:
                        if gap > max_gap:
                            continue
                        rows[r][c] = before[:index] + digit + before[index + 1:]
                        fixed = sum(_holds(e, rows) for e in failing_before)
                        broke = any(not _holds(e, rows) for e in containing[(r, c)] if e not in failing_before)
                        if _holds(equation, rows) and not broke:
                            options.append((fixed, r, c, rows[r][c]))
                        rows[r][c] = before
            if not options:
                continue
            # A change that repairs a row and a column at once outranks one that repairs only this sum.
            best = max(option[0] for option in options)
            top = [option for option in options if option[0] == best]
            if len(top) != 1:
                continue
            _, r, c, after = top[0]
            changes.append((r, c, rows[r][c], after))
            rows[r][c] = after
            changed.add((r, c))
            progress = True
    return changes


def repair_tables(parsed, verification, alternatives):
    """Runs the search over every table. Returns repair entries in the same shape as a re-read's."""
    repairs = []
    if not alternatives:
        return repairs
    for table_index, table in enumerate(parsed.get("tables") or []):
        if table.get("origin") == "file":
            continue  # read straight from the file, not by the model
        for row, column, before, after in repair_table(table, verification["tables"][table_index], alternatives):
            repairs.append({
                "table": table_index,
                "row": row,
                "column": table["headers"][column],
                "before": before,
                "after": after,
                "method": "second_choice_digit",
            })
    return repairs
