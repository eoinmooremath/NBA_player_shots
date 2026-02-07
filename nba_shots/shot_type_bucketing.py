# shot_type_bucketing.py
#
# Centralized shot-type bucketing for the entire project.
# This is intentionally simple and fast (vectorized), and it is the single source of truth.
#
# Buckets (UI):
#   Putback, Dunk, Layup, Jump, Other
#
# Policy:
#   - "Hook" is merged into "Jump" (never output a separate "Hook" bucket).
#   - Putback takes priority over Dunk for cases like "Putback Slam Dunk Shot" and "Tip Dunk Shot".
#
import numpy as np
import pandas as pd


def _norm_subtype(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[\s\-_]+", " ", regex=True).str.strip()
    return s


def shot_type_simple_from_subtype(subtype: pd.Series) -> pd.Series:
    """
    Vectorized bucketing from the raw `subType` field.

    Returns a Series of strings in:
      {"Putback","Dunk","Layup","Jump","Other"}
    """
    st = _norm_subtype(subtype)

    # Priority matters.
    is_putback = st.str.contains(r"\b(tip|putback|follow)\b", regex=True)
    is_dunk = st.str.contains(r"\bdunk\b", regex=True)
    is_layup = st.str.contains(r"\b(layup|finger roll)\b", regex=True)

    # Jump bucket includes hooks (per your choice), plus common jump variants.
    is_jump = st.str.contains(r"\b(jump|pull|fade|step|float|turnaround|bank|hook)\b", regex=True)

    out = np.select(
        [is_putback, is_dunk, is_layup, is_jump],
        ["Putback", "Dunk", "Layup", "Jump"],
        default="Other",
    )
    return pd.Series(out, index=subtype.index, dtype="string")
