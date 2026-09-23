"""
_widen_review_columns widens tag_rationale (a sentence) alongside
claim_text / chunk_text / evidence_span. tag_draft (one word) stays narrow.
Spec 2026-09-10-golden-set-row-tagging-design.md §7 edit 1.
"""
import pandas as pd
from openpyxl import load_workbook

import golden_set_pipeline as gsp


def test_export_for_review_widens_tag_rationale(tmp_path):
    df = pd.DataFrame([{
        "claim_text": "c", "chunk_text": "t", "evidence_span": "s",
        "tag_draft": "extractive",
        "tag_rationale": "a full sentence explaining why this row is extractive",
    }])
    path = str(tmp_path / "review.xlsx")
    gsp.export_for_review(df, path)

    ws = load_workbook(path).active
    widths = {c: ws.column_dimensions[chr(ord("A") + i)].width
              for i, c in enumerate(df.columns)}
    assert widths["tag_rationale"] == 60
    assert widths["claim_text"] == 60
    assert widths["tag_draft"] == 18
