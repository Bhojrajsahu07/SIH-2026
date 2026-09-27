"""
INFRA NIRMAAN — Early Warning Alert System (EWAS)
MoSPI Infrastructure Project Delay/Cost-Overrun Prediction Pipeline
=====================================================================

Built and validated against a real ~175k-row sample of the source data
(mospi_sample.csv), not a generic/assumed CSV shape. Key facts this script
relies on, discovered by inspecting real rows before writing any regex:

  * Raw columns are: File_Name, Page, Sector, Column_1 .. Column_15.
    Column_1 is almost always the table's Sl.No, Column_2 is almost always
    the Project Name (+ Agency + bracketed Project Code) block. Columns 3+
    hold dates/costs/expenditure/progress, and WHICH Column_N holds what
    genuinely shifts row to row depending on which of several table
    layouts got scraped (main project table vs sector-summary table vs
    delay-summary table). So Column_1/2 are read positionally; everything
    from Column_3 onward is classified by CONTENT, not position.

  * Report date (the PDF's as-of month) is not a column — it has to be
    parsed out of File_Name, which uses ~6 different naming conventions
    across 2001-2025 (FR_MAR_2013.pdf, MonthlyFR_apr_2004.pdf,
    FRMarch2025.pdf, December.pdf with no year at all, etc).

  * MoSPI's bracketed Project Code, e.g. "[N22000602]", only exists in
    the data from ~2013 onward (0% before 2012, ~60-75% from 2013+). It
    is NOT a reliable universal join key — a name+sector fallback key is
    required for 2001-2012 and for uncoded rows in later years. That
    fallback key must reject too-short/blank names: an early version of
    this pipeline let short names form a key, and several unrelated
    projects with blank/garbled name cells collapsed into one fake
    "project" whose Original_Cost bounced between snapshots and produced
    a 300,000%+ Cost_Overrun_Pct outlier. Names under 8 normalized
    characters are now treated as unresolvable and dropped instead.

  * Cost/date/expenditure cells appear in two formats depending on
    report vintage:
      - packed:   "645.63 (-) [645.63]"      -> Original (Revised) [Anticipated]
      - unpacked: three separate plain cells with the same three values
    The SECOND cost-like cell in a row is not a second cost triple — its
    bracketed component is Physical Progress % (a 0-100 integer), not a
    third cost figure. Confirmed by cross-referencing against a literal
    header row that leaked into the data as a garbage row:
      "State | Sector | Sl No | Project Name (Agency Name) (Project Code) |
       Date of Approval (MM/YYYY) | Date of Commissioning Original (Revised)
       {Anticipated} (MM/YYYY) | Cost Original (Revised) {Anticipated} in
       Rs. Crore | Cumulative Expenditure in Rs. Crore"

  * A handful of "Original_Cost" extractions come out implausibly tiny
    (<5 crore) — almost certainly a misclassified cell (a percentage or
    Sl.No that slipped into the cost slot) rather than a real project
    cost, since MoSPI's monitored list has a substantial cost floor.
    These are nulled out rather than trusted, and the project's *true*
    original cost is taken as the mode across its own snapshots (not
    just the first chronological one), since Original Cost should be
    stable snapshot-to-snapshot and the mode is robust to one-off
    extraction noise in a way "first observed value" is not.

  * ~2% of rows come from files with no parseable year and are dropped.
  * A meaningful fraction of rows are not project snapshots at all —
    they're sector-summary tables, delay-count tables, or literal header
    rows re-scraped as data, or symbol-only cells like "- - ( )". These
    are filtered out (see is_real_project_row).

Everything below is written against these confirmed facts. Where a field
truly cannot be recovered with confidence from the raw text (State,
Agency, Project_Tier), it is produced as a clearly-labeled best-effort /
engineered value rather than presented as a clean extraction.
"""

import re
import csv
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CHUNK_SIZE = 50_000
LEAKAGE_GUARD_MONTHS = 12
N_SPLITS = 5
RANDOM_STATE = 42
MIN_PLAUSIBLE_COST_CR = 5.0  # below this, treat an extracted "cost" as noise

RAW_COLUMNS = [f"Column_{i}" for i in range(1, 16)]

INDIAN_STATES = [
    "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH",
    "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JHARKHAND", "KARNATAKA",
    "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA",
    "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM",
    "TAMIL NADU", "TELANGANA", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND",
    "WEST BENGAL", "DELHI", "JAMMU AND KASHMIR", "LADAKH", "PUDUCHERRY",
    "CHANDIGARH", "MULTI STATE", "MULTI-STATE",
]

# Row-level signatures that mean "this is not a project snapshot row" —
# literal header text re-scraped as data, or a different table entirely
# (sector-summary / delay-summary tables), all confirmed present verbatim
# in the sample.
GARBAGE_SIGNATURES = [
    "sl.no", "sl. no", "sl no", "si.no", "si. no",
    "date of approval", "project name", "cost original",
    "cumulative expenditure", "projects on monitor",
    "number of projects", "projects additionally delayed",
    "projects delayed", "with respect to", "grand total",
]

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_RE = re.compile(r"(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")", re.IGNORECASE)
_YEAR_RE = re.compile(r"(20[0-2]\d|19[89]\d)")

TRIPLE_RE = re.compile(r"([\d,]+\.\d+|\d{1,2}\s*/\s*\d{4})\s*\(([^)]*)\)\s*\[([^\]]*)\]")
PLAIN_DATE_RE = re.compile(r"^\s*\d{1,2}\s*/\s*\d{4}\s*$")
PLAIN_COST_RE = re.compile(r"^\s*[\d,]+\.\d+\s*$")
PLAIN_PCT_RE = re.compile(r"^\s*\d{1,3}\s*$")
PID_RE = re.compile(r"\[\s*([A-Za-z]\d{5,})\s*\]")
FOOTNOTE_NUM_RE = re.compile(r"^\d+\.$")
SPACED_TOKEN_RE = re.compile(r"(?:[\d./]\s*){4,}")
AGENCY_STATE_TAIL_RE = re.compile(r"\]\s*([A-Za-z.&\s]{2,40}?)\s*,\s*([A-Za-z\s]{3,30})\s*$")


# ---------------------------------------------------------------------------
# LOW-LEVEL PARSERS
# ---------------------------------------------------------------------------

def parse_report_date(filename: str):
    """Extract (year, month) from a source PDF filename. Returns (None, None)
    if no year token is present (~2% of files — mostly month-only names like
    'December.pdf' with no year at all)."""
    if not isinstance(filename, str):
        return None, None
    m = _MONTH_RE.search(filename)
    y = _YEAR_RE.search(filename)
    month = MONTHS[m.group(1).lower()] if m else None
    year = int(y.group(1)) if y else None
    return year, month


def despace(token: str) -> str:
    """Collapse OCR/extraction artifacts like '0 2 / 2 0 0 8' -> '02/2008'.
    Only touches tokens that are entirely digits/slashes/dots/spaces, so it
    never mangles real multi-word text."""
    t = token.strip()
    if SPACED_TOKEN_RE.fullmatch(t):
        return re.sub(r"\s+", "", t)
    return t


def to_float(s):
    if s is None:
        return np.nan
    s = str(s).replace(",", "").strip()
    if s in ("", "-", "--", ".", "NA", "N/A"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_mm_yyyy(s):
    """Parse an 'MM/YYYY' style token into a pandas Timestamp (1st of month)."""
    if s is None:
        return pd.NaT
    s = despace(str(s))
    m = re.match(r"^\s*(\d{1,2})\s*/\s*(\d{4})\s*$", s)
    if not m:
        return pd.NaT
    mm, yyyy = int(m.group(1)), int(m.group(2))
    if not (1 <= mm <= 12) or not (1990 <= yyyy <= 2035):
        return pd.NaT
    try:
        return pd.Timestamp(year=yyyy, month=mm, day=1)
    except ValueError:
        return pd.NaT


def classify_row_tokens(cells):
    """Content-based classification of Column_3..Column_15 (dates vs costs
    vs bare progress percentages), in order of appearance, ignoring which
    raw column index they came from. Each date/cost entry is either
    ('plain', value) or ('triple', original, revised, anticipated).
    `bare_pcts` catches standalone small integers like '6' or '22' that are
    Physical Progress % values with no decimal point — these were
    previously being silently dropped since they don't match a date or
    cost pattern.
    """
    dates, costs, bare_pcts = [], [], []
    for raw in cells:
        c = despace(str(raw))
        if not c:
            continue
        tm = TRIPLE_RE.match(c)
        if tm:
            is_date = "/" in tm.group(1)
            (dates if is_date else costs).append(("triple", tm.group(1), tm.group(2), tm.group(3)))
        elif PLAIN_DATE_RE.match(c):
            dates.append(("plain", c.strip()))
        elif PLAIN_COST_RE.match(c):
            costs.append(("plain", c.strip()))
        elif PLAIN_PCT_RE.match(c) and 0 <= int(c.strip()) <= 100:
            bare_pcts.append(c.strip())
    return dates, costs, bare_pcts


def is_real_project_row(row1, row2_text, pid, dates, costs, bare_pcts=None):
    """Positive-evidence gate: a real project snapshot row must show either
    a Project Code, or at minimum one date AND one cost token. Everything
    else (sector summaries, delay-count tables, stray header rows, footnote
    lines, symbol-only cells) is dropped. Backed by a keyword blocklist as
    defense in depth."""
    text_l = (str(row1) + " " + str(row2_text)).lower()
    if any(sig in text_l for sig in GARBAGE_SIGNATURES):
        return False
    if FOOTNOTE_NUM_RE.match(str(row1).strip()):
        return False
    if str(row1).strip().upper() in (
        "ROAD", "RAILWAY", "POWER", "PORT", "HIGHWAY", "COAL", "PETROLEUM",
        "URBAN", "WATER", "STEEL", "TELECOMMUNICATION", "MINE", "ATOMIC",
        "HEALTH", "EDUCATION", "AVIATION", "FERTILIZER", "STATE",
    ):
        return False
    # a real project name has real substance — reject symbol/whitespace-only
    # cells like "- - ( )" that occasionally slip past the checks above
    if len(re.sub(r"[^A-Za-z0-9]", "", str(row2_text))) < 5:
        return False
    if pid:
        return True
    return len(dates) >= 1 and len(costs) >= 1


def extract_project_id(full_row_text: str):
    m = PID_RE.search(full_row_text)
    return m.group(1).upper() if m else None


def extract_agency_state(project_cell: str):
    """Best-effort: agency/state text is sometimes glued directly onto the
    end of the Project Name cell right after the closing bracket, e.g.
    '...- [N22000361]CR,MAHARASHTRA'. Low-confidence — returns (None, None)
    far more often than not, which is expected and fine."""
    if not isinstance(project_cell, str):
        return None, None
    m = AGENCY_STATE_TAIL_RE.search(project_cell)
    if not m:
        return None, None
    agency, state = m.group(1).strip(" ,"), m.group(2).strip().upper()
    if state in INDIAN_STATES:
        return (agency or None), state
    return None, None


def fallback_project_key(sector: str, project_cell: str):
    """Pre-2013 (and any un-coded row) fallback identity key: normalized
    sector + first ~60 chars of the project name text, since there is no
    Project Code to key off of in that era. Returns None when the name is
    too short to be a safe key, rather than letting many unrelated
    projects collapse into one bucket."""
    name = re.sub(r"[^A-Z0-9]", "", str(project_cell).upper())[:60]
    if len(name) < 8:
        return None
    return f"NOCODE::{str(sector).upper()}::{name}"


# ---------------------------------------------------------------------------
# ROW -> RAW FIELD EXTRACTION
# ---------------------------------------------------------------------------

def extract_row_fields(rec: dict):
    """Turn one raw CSV record into a dict of extracted raw fields. Returns
    None if the row fails the garbage/off-topic-table filter or has no
    resolvable identity (no code AND name too short for a fallback key)."""
    file_name = rec.get("File_Name", "")
    sector = rec.get("Sector", "")
    col1 = rec.get("Column_1", "")
    col2 = rec.get("Column_2", "")
    rest = [rec.get(c, "") for c in RAW_COLUMNS[2:]]  # Column_3..Column_15
    full_text = " ".join([str(col1), str(col2)] + [str(x) for x in rest])

    pid = extract_project_id(full_text)
    dates, costs, bare_pcts = classify_row_tokens(rest)

    if not is_real_project_row(col1, col2, pid, dates, costs, bare_pcts):
        return None

    fallback_key = None
    if pid is None:
        fallback_key = fallback_project_key(sector, col2)
        if fallback_key is None:
            return None  # no code, and name too short to key safely — unresolvable

    year, month = parse_report_date(file_name)
    if year is None:
        return None
    report_date = pd.Timestamp(year=year, month=month, day=1)

    # --- dates: [0]=Approval, [1]=Commissioning (packed O/R/A, or paired plains) ---
    approval_date = pd.NaT
    comm_original = comm_revised = comm_anticipated = pd.NaT
    if len(dates) >= 1:
        approval_date = parse_mm_yyyy(dates[0][1])
    if len(dates) >= 2:
        d1 = dates[1]
        if d1[0] == "triple":
            comm_original = parse_mm_yyyy(d1[1])
            comm_revised = parse_mm_yyyy(d1[2]) if d1[2] not in ("-", "", "--") else pd.NaT
            comm_anticipated = parse_mm_yyyy(d1[3]) if d1[3] not in ("-", "", "--") else pd.NaT
        else:
            comm_original = parse_mm_yyyy(d1[1])
            if len(dates) >= 3:
                comm_anticipated = parse_mm_yyyy(dates[2][1])

    # --- costs: [0]=Cost (packed O/R/A, or plains), [1]=Expenditure(+Progress%) ---
    cost_original = cost_revised = cost_anticipated = np.nan
    cum_expenditure = np.nan
    physical_progress_pct = np.nan
    if len(costs) >= 1:
        c0 = costs[0]
        if c0[0] == "triple":
            cost_original = to_float(c0[1])
            cost_revised = to_float(c0[2])
            cost_anticipated = to_float(c0[3])
        else:
            cost_original = to_float(c0[1])
            if len(costs) >= 2 and costs[1][0] == "plain":
                cost_anticipated = to_float(costs[1][1])
    if pd.isna(cost_anticipated):
        cost_anticipated = cost_original
    if len(costs) >= 2:
        c1 = costs[1]
        if c1[0] == "triple":
            cum_expenditure = to_float(c1[1])
            pct_raw = c1[3].strip()
            if re.fullmatch(r"\d{1,3}", pct_raw):
                pp = float(pct_raw)
                physical_progress_pct = pp if 0 <= pp <= 100 else np.nan
        elif len(costs) >= 3:
            cum_expenditure = to_float(costs[2][1]) if costs[2][0] == "plain" else np.nan
    if pd.isna(physical_progress_pct) and bare_pcts:
        # last bare integer on the row is the best candidate for progress %
        # (earlier ones are more likely to be stray Sl.No/page-index noise)
        physical_progress_pct = float(bare_pcts[-1])

    # sanity floor — an implausibly tiny "cost" is almost always a
    # misclassified cell, not a real MoSPI-monitored project cost
    if cost_original == cost_original and cost_original < MIN_PLAUSIBLE_COST_CR:
        cost_original = np.nan
    if cost_anticipated == cost_anticipated and cost_anticipated < MIN_PLAUSIBLE_COST_CR:
        cost_anticipated = np.nan

    agency, state = extract_agency_state(col2)

    return {
        "Project_ID_raw": pid,
        "Fallback_Key": fallback_key,
        "S_No": to_float(col1),
        "Sector": str(sector).strip().upper() if sector else "UNKNOWN",
        "Project_Name": str(col2).strip(),
        "State_Location": state or "UNKNOWN",
        "Implementing_Agency": agency or "UNKNOWN",
        "Report_Date": report_date,
        "Approval_Date": approval_date,
        "Comm_Date_Original": comm_original,
        "Comm_Date_Revised": comm_revised,
        "Comm_Date_Anticipated": comm_anticipated,
        "Cost_Original": cost_original,
        "Cost_Revised": cost_revised,
        "Cost_Anticipated": cost_anticipated,
        "Cumulative_Expenditure": cum_expenditure,
        "Physical_Progress_Pct": physical_progress_pct,
        "Source_File": file_name,
    }


# ---------------------------------------------------------------------------
# CHUNKED INGESTION
# ---------------------------------------------------------------------------

def ingest(csv_path: str, chunksize: int = CHUNK_SIZE, verbose=True):
    """Stream the raw scraped CSV in chunks, parsing row-by-row via csv
    semantics (quote-aware) rather than trusting fixed column positions for
    anything beyond Sl.No/Project Name. Returns a single consolidated
    DataFrame of extracted raw fields — this distilled frame is far smaller
    than the raw text file, so holding it fully in memory for the
    downstream groupby/leakage logic (which needs whole-dataset visibility)
    is the correct tradeoff; the chunking protects memory during the raw
    text-parsing pass, which is where the real size is."""
    records = []
    total_seen = 0
    total_kept = 0
    reader = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_MINIMAL,
    )
    for chunk_no, chunk in enumerate(reader):
        total_seen += len(chunk)
        for rec in chunk.to_dict(orient="records"):
            row = extract_row_fields(rec)
            if row is not None:
                records.append(row)
                total_kept += 1
        if verbose:
            print(f"  chunk {chunk_no}: seen={total_seen:,} kept_so_far={total_kept:,}", file=sys.stderr)
    if verbose:
        print(f"Ingestion done: {total_kept:,} / {total_seen:,} rows kept "
              f"({100*total_kept/max(total_seen,1):.1f}%)", file=sys.stderr)
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# PROJECT ID RESOLUTION
# ---------------------------------------------------------------------------

def resolve_project_id(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Project_ID: use the bracketed code where present; fall
    back to the sector+name composite key for the ~pre-2013 / uncoded rows
    (rows with neither a code nor a safe fallback key were already dropped
    during extraction). A tiny number of legacy rows may still fragment
    into multiple IDs for what was really one project (record linkage on
    free text is inherently imperfect) — documented limitation, not
    silently hidden."""
    df = df.copy()
    df["Project_ID"] = df["Project_ID_raw"].where(
        df["Project_ID_raw"].notna(), df["Fallback_Key"]
    )
    df = df.drop(columns=["Project_ID_raw", "Fallback_Key"])
    return df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING (the 15 input features)
# ---------------------------------------------------------------------------

def months_between(later, earlier):
    if pd.isna(later) or pd.isna(earlier):
        return np.nan
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def project_tier(cost):
    """Not present as a literal field anywhere in the source data — engineered
    from Original_Cost using standard infra-monitoring size bands. Labeled
    as derived, not extracted."""
    if pd.isna(cost):
        return "UNKNOWN"
    if cost >= 1000:
        return "MEGA"
    if cost >= 500:
        return "MAJOR"
    if cost >= 150:
        return "MEDIUM"
    return "MINOR"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Original_Cost/Duration should be stable across a project's own
    # snapshots. Use the per-project MODE rather than each row's own
    # (possibly noisy) extraction, so one bad parse in one snapshot
    # doesn't poison that snapshot's features.
    def _mode_or_nan(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else np.nan

    stable_cost = df.groupby("Project_ID")["Cost_Original"].transform(_mode_or_nan)
    df["Original_Cost"] = stable_cost

    df["Original_Duration_Months"] = df.apply(
        lambda r: months_between(r["Comm_Date_Original"], r["Approval_Date"]), axis=1
    )
    stable_duration = df.groupby("Project_ID")["Original_Duration_Months"].transform(_mode_or_nan)
    df["Original_Duration_Months"] = stable_duration

    df["Elapsed_Time_Months"] = df.apply(
        lambda r: months_between(r["Report_Date"], r["Approval_Date"]), axis=1
    )
    df["Project_Tier"] = df["Original_Cost"].apply(project_tier)

    # Financial progress uses Anticipated cost (falls back to Original if missing)
    denom_cost = df["Cost_Anticipated"].where(df["Cost_Anticipated"].notna(), df["Original_Cost"])
    df["Financial_Progress_Pct"] = np.where(
        (denom_cost.notna()) & (denom_cost != 0),
        100 * df["Cumulative_Expenditure"] / denom_cost,
        np.nan,
    )

    time_progress_pct = np.where(
        (df["Original_Duration_Months"].notna()) & (df["Original_Duration_Months"] != 0),
        100 * df["Elapsed_Time_Months"] / df["Original_Duration_Months"],
        np.nan,
    )
    df["Progress_Anomaly_Gap"] = time_progress_pct - df["Financial_Progress_Pct"]

    df["Expenditure_Burn_Rate"] = np.where(
        (df["Elapsed_Time_Months"].notna()) & (df["Elapsed_Time_Months"] > 0),
        df["Cumulative_Expenditure"] / df["Elapsed_Time_Months"],
        np.nan,
    )

    df["Actual_Physical_Run_Rate"] = np.where(
        (df["Elapsed_Time_Months"].notna()) & (df["Elapsed_Time_Months"] > 0),
        df["Physical_Progress_Pct"] / df["Elapsed_Time_Months"],
        np.nan,
    )
    df["Expected_Physical_Run_Rate"] = np.where(
        (df["Original_Duration_Months"].notna()) & (df["Original_Duration_Months"] > 0),
        100 / df["Original_Duration_Months"],
        np.nan,
    )
    df["Run_Rate_Gap"] = df["Actual_Physical_Run_Rate"] - df["Expected_Physical_Run_Rate"]

    df = df.rename(columns={"S_No": "S.No"})
    return df


FEATURE_COLUMNS = [
    "S.No", "Project_ID", "Sector", "State_Location", "Implementing_Agency",
    "Project_Tier", "Original_Cost", "Original_Duration_Months",
    "Elapsed_Time_Months", "Cumulative_Expenditure", "Physical_Progress_Pct",
    "Financial_Progress_Pct", "Progress_Anomaly_Gap", "Expenditure_Burn_Rate",
    "Actual_Physical_Run_Rate", "Expected_Physical_Run_Rate",
]

TARGET_COLUMNS = [
    "Revised_Cost_Actual", "Revised_Duration_Actual",
    "Cost_Overrun_Pct", "Time_Overrun_Months",
]


# ---------------------------------------------------------------------------
# TIME-HORIZON SEGREGATION (leakage-safe target derivation)
# ---------------------------------------------------------------------------

def derive_targets_and_split_leakage_safe(df: pd.DataFrame, guard_months=LEAKAGE_GUARD_MONTHS):
    """For each Project_ID: the chronologically latest snapshot is the
    ground-truth outcome (used ONLY to compute the 4 targets, never kept as
    a feature row itself). Every other snapshot at least `guard_months`
    before that outcome date becomes a training row, labeled with that same
    outcome. Snapshots inside the guard window are dropped entirely."""
    df = df.sort_values(["Project_ID", "Report_Date"])
    out_rows = []
    skipped_single_snapshot = 0
    skipped_within_guard = 0

    for pid, g in df.groupby("Project_ID", sort=False):
        g = g.sort_values("Report_Date")
        final = g.iloc[-1]
        final_date = final["Report_Date"]

        revised_cost_actual = final["Cost_Anticipated"] if pd.notna(final["Cost_Anticipated"]) else final["Original_Cost"]
        revised_duration_actual = months_between(
            final["Comm_Date_Anticipated"] if pd.notna(final["Comm_Date_Anticipated"]) else final["Comm_Date_Original"],
            final["Approval_Date"],
        )
        anchor_original_cost = final["Original_Cost"]  # already the stable per-project mode
        anchor_original_duration = final["Original_Duration_Months"]

        cost_overrun_pct = (
            100 * (revised_cost_actual - anchor_original_cost) / anchor_original_cost
            if pd.notna(revised_cost_actual) and pd.notna(anchor_original_cost) and anchor_original_cost != 0
            else np.nan
        )
        time_overrun_months = (
            revised_duration_actual - anchor_original_duration
            if pd.notna(revised_duration_actual) and pd.notna(anchor_original_duration)
            else np.nan
        )

        history = g.iloc[:-1]
        if history.empty:
            skipped_single_snapshot += len(g)
            continue

        cutoff = final_date - pd.DateOffset(months=guard_months)
        usable = history[history["Report_Date"] <= cutoff].copy()
        skipped_within_guard += len(history) - len(usable)
        if usable.empty:
            continue

        usable["Revised_Cost_Actual"] = revised_cost_actual
        usable["Revised_Duration_Actual"] = revised_duration_actual
        usable["Cost_Overrun_Pct"] = cost_overrun_pct
        usable["Time_Overrun_Months"] = time_overrun_months
        out_rows.append(usable)

    print(f"  Leakage-safe split: {skipped_single_snapshot:,} rows dropped (only snapshot for their project), "
          f"{skipped_within_guard:,} rows dropped (inside {guard_months}-month guard window)", file=sys.stderr)

    if not out_rows:
        return pd.DataFrame(columns=list(df.columns) + TARGET_COLUMNS)
    return pd.concat(out_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# MODEL ENSEMBLE
# ---------------------------------------------------------------------------

def get_model_backends():
    """Prefer LightGBM + XGBoost (what this ships with in production). Falls
    back to a scikit-learn ensemble ONLY if those libraries aren't
    installed, so the pipeline still runs somewhere without them (e.g. this
    sandbox, which has no internet access to pip install)."""
    try:
        import lightgbm as lgb
        import xgboost as xgb
        return "lightgbm_xgboost", lgb, xgb
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return "sklearn_fallback", HistGradientBoostingRegressor, HistGradientBoostingRegressor


def clean_xy(X, y):
    """XGBoost crashes on NaN/Inf in y (LightGBM only warns) — scrub both
    before every fit, and keep X/y aligned."""
    y = pd.Series(y).replace([np.inf, -np.inf], np.nan)
    mask = y.notna()
    return X.loc[mask], y.loc[mask], mask


def build_model_instance(backend_name, backend_module, model_kind, random_state):
    if backend_name == "lightgbm_xgboost":
        if model_kind == "lgbm":
            return backend_module.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                random_state=random_state, verbosity=-1,
            )
        else:
            return backend_module.XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                random_state=random_state, verbosity=0,
            )
    else:
        return backend_module(max_iter=400, learning_rate=0.05, random_state=random_state)


def encode_categoricals(df, cat_cols, encoders=None):
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    fit_mode = encoders is None
    encoders = encoders or {}
    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
        if fit_mode:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])
            encoders[c] = le
        else:
            le = encoders[c]
            known = set(le.classes_)
            df[c] = df[c].apply(lambda v: v if v in known else "UNKNOWN")
            if "UNKNOWN" not in known:
                le.classes_ = np.append(le.classes_, "UNKNOWN")
            df[c] = le.transform(df[c])
    return df, encoders


def train_ensemble(df: pd.DataFrame):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, r2_score

    cat_cols = ["Sector", "State_Location", "Implementing_Agency", "Project_Tier"]
    num_cols = [c for c in FEATURE_COLUMNS if c not in cat_cols and c not in ("Project_ID", "S.No")]

    X_full, encoders = encode_categoricals(df[cat_cols], cat_cols)
    for c in num_cols:
        X_full[c] = pd.to_numeric(df[c], errors="coerce")
    X_full = X_full[cat_cols + num_cols]
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    for c in num_cols:
        X_full[c] = X_full[c].fillna(X_full[c].median())

    groups = df["Project_ID"].values
    backend_name, backend_a, backend_b = get_model_backends()
    print(f"Model backend in use this run: {backend_name}", file=sys.stderr)

    n_groups = df["Project_ID"].nunique()
    n_splits = min(N_SPLITS, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    results = {}
    for target in TARGET_COLUMNS:
        y_all = pd.to_numeric(df[target], errors="coerce")
        X, y, mask = clean_xy(X_full, y_all)
        g = groups[mask.values]

        if len(y) < n_splits or y.nunique() < 2:
            print(f"  [{target}] skipped — insufficient usable rows ({len(y)})", file=sys.stderr)
            continue

        fold_mae, fold_r2 = [], []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=g)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            m1 = build_model_instance(backend_name, backend_a, "lgbm", RANDOM_STATE)
            m2 = build_model_instance(backend_name, backend_b, "xgb", RANDOM_STATE + 1)
            m1.fit(X_tr, y_tr)
            m2.fit(X_tr, y_tr)
            pred = (m1.predict(X_va) + m2.predict(X_va)) / 2.0

            fold_mae.append(mean_absolute_error(y_va, pred))
            fold_r2.append(r2_score(y_va, pred) if len(set(y_va)) > 1 else np.nan)

        # final models trained on ALL usable rows for this target
        m1_final = build_model_instance(backend_name, backend_a, "lgbm", RANDOM_STATE)
        m2_final = build_model_instance(backend_name, backend_b, "xgb", RANDOM_STATE + 1)
        m1_final.fit(X, y)
        m2_final.fit(X, y)

        results[target] = {
            "model_a": m1_final,
            "model_b": m2_final,
            "cv_mae_mean": float(np.nanmean(fold_mae)),
            "cv_r2_mean": float(np.nanmean(fold_r2)),
            "n_train_rows": int(len(y)),
        }
        print(f"  [{target}] n={len(y):,}  CV MAE={results[target]['cv_mae_mean']:.3f}  "
              f"CV R2={results[target]['cv_r2_mean']:.3f}", file=sys.stderr)

    return results, encoders, backend_name, cat_cols, num_cols


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str, output_pkl: str, chunksize: int = CHUNK_SIZE,
                  guard_months: int = LEAKAGE_GUARD_MONTHS, export_only: bool = False,
                  export_csv: str = None):
    """Full pipeline. If export_only=True, stops after feature engineering +
    leakage-safe target derivation and writes a clean CSV instead of
    training anything — the path to use if you're bringing your own
    LightGBM/XGBoost + feature engineering and just want filtered,
    model-ready data."""
    print("=== 1. Chunked ingestion + row-level parsing ===", file=sys.stderr)
    raw_df = ingest(csv_path, chunksize=chunksize)

    print("=== 2. Project ID resolution ===", file=sys.stderr)
    raw_df = resolve_project_id(raw_df)

    print("=== 3. Feature engineering ===", file=sys.stderr)
    feat_df = engineer_features(raw_df)

    print("=== 4. Time-horizon segregation (leakage-safe target derivation) ===", file=sys.stderr)
    model_df = derive_targets_and_split_leakage_safe(feat_df, guard_months=guard_months)
    n_proj = model_df["Project_ID"].nunique() if len(model_df) else 0
    print(f"  Final modeling rows: {len(model_df):,} across {n_proj:,} projects", file=sys.stderr)

    if export_only:
        out_path = export_csv or (output_pkl.rsplit(".", 1)[0] + "_features.csv")
        out_cols = FEATURE_COLUMNS + TARGET_COLUMNS + ["Report_Date", "Source_File"]
        model_df[out_cols].to_csv(out_path, index=False)
        print(f"=== Done. Saved clean features to {out_path} (no training run) ===", file=sys.stderr)
        return model_df, out_path

    print("=== 5. Model ensemble training (GroupKFold by Project_ID) ===", file=sys.stderr)
    results, encoders, backend_name, cat_cols, num_cols = train_ensemble(model_df)

    payload = {
        "models": {t: {"model_a": r["model_a"], "model_b": r["model_b"]} for t, r in results.items()},
        "encoders": encoders,
        "feature_columns": cat_cols + num_cols,
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "target_columns": list(results.keys()),
        "backend": backend_name,
        "metrics": {t: {"cv_mae": r["cv_mae_mean"], "cv_r2": r["cv_r2_mean"], "n_train_rows": r["n_train_rows"]}
                    for t, r in results.items()},
        "trained_at": datetime.utcnow().isoformat(),
        "leakage_guard_months": guard_months,
        "n_projects": int(n_proj),
        "n_training_rows": int(len(model_df)),
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(payload, f)
    print(f"=== Done. Saved {output_pkl} ===", file=sys.stderr)
    return payload, model_df


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="INFRA NIRMAAN EWAS pipeline — clean + feature-engineer MoSPI project "
                    "data, and optionally train a LightGBM+XGBoost ensemble."
    )
    p.add_argument("input_csv", help="Path to the raw scraped MoSPI CSV")
    p.add_argument("output", nargs="?", default=None,
                   help="Output path: .pkl for a trained model (default), or the target "
                        "CSV path if --export-only is set")
    p.add_argument("--export-only", action="store_true",
                   help="Stop after cleaning/feature-engineering; write a CSV instead of "
                        "training anything. Use this if you're training your own "
                        "LightGBM/XGBoost + feature engineering downstream.")
    p.add_argument("--chunksize", type=int, default=CHUNK_SIZE,
                   help=f"Rows per chunk during raw CSV ingestion (default {CHUNK_SIZE})")
    p.add_argument("--guard-months", type=int, default=LEAKAGE_GUARD_MONTHS,
                   help=f"Minimum months a snapshot must precede a project's final "
                        f"outcome to be used as a training row (default {LEAKAGE_GUARD_MONTHS})")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    default_out = "/home/claude/ewas_model.pkl" if not args.export_only else "/home/claude/mospi_model_ready_features.csv"
    output_path = args.output or default_out

    if args.export_only:
        run_pipeline(args.input_csv, output_path, chunksize=args.chunksize,
                     guard_months=args.guard_months, export_only=True, export_csv=output_path)
    else:
        run_pipeline(args.input_csv, output_path, chunksize=args.chunksize,
                     guard_months=args.guard_months, export_only=False)
