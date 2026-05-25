from __future__ import annotations

from scripts.fisher_eval_gemma import _summary_passed


def test_fisher_eval_summary_predicate() -> None:
    summary = {
        "sample_count": 4,
        "slot_count": 8,
        "finite_fisher": True,
        "positive_fisher_slots": 3,
        "approved_snapshot_hash": "hash",
        "drift_within_bound": True,
        "artificial_over_threshold_rejected": True,
        "gcs_output_uri": "",
        "outputs_uploaded": False,
    }

    assert _summary_passed(summary)
    assert not _summary_passed({**summary, "positive_fisher_slots": 0})
    assert not _summary_passed({**summary, "artificial_over_threshold_rejected": False})
    assert not _summary_passed({**summary, "approved_snapshot_hash": ""})
    assert not _summary_passed({**summary, "gcs_output_uri": "gs://bucket/out"})
    assert _summary_passed(
        {**summary, "gcs_output_uri": "gs://bucket/out", "outputs_uploaded": True}
    )
