# shot_type_bucketing.py
import polars as pl

def shot_type_simple_from_subtype_expr(col_name: str = "subType") -> pl.Expr:
    """
    Returns a Polars Expression that classifies shot types.
    Usage: df.with_columns(shot_type_simple_from_subtype_expr("subType"))
    """
    # Normalize: lowercase, strip, remove extra spaces
    # Note: Polars regex replacement is slightly different than Pandas
    st = (
        pl.col(col_name)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[\s\-_]+", " ")
        .str.strip_chars()
    )

    # We use a purely expression-based Case/When (Polars 'when-then-otherwise')
    
    # Check regex patterns
    is_putback = st.str.contains(r"\b(tip|putback|follow)\b")
    is_dunk    = st.str.contains(r"\bdunk\b")
    is_layup   = st.str.contains(r"\b(layup|finger roll)\b")
    is_jump    = st.str.contains(r"\b(jump|pull|fade|step|float|turnaround|bank|hook)\b")

    return (
        pl.when(is_putback).then(pl.lit("Putback"))
        .when(is_dunk).then(pl.lit("Dunk"))
        .when(is_layup).then(pl.lit("Layup"))
        .when(is_jump).then(pl.lit("Jump"))
        .otherwise(pl.lit("Other"))
        .alias("ShotType_Simple")
    )
