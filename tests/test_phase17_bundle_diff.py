import json
import pytest
from qocc.core.bundle_diff import BundleDiff
from qocc.api import compare_bundles

def test_bundle_diff_dataclass():
    b_diff = BundleDiff(
        metric_deltas={"depth": {"a": 10, "b": 15, "pct_change": 50.0}},
        circuit_hash_change=True,
        regression_cause="PASS_PARAM"
    )
    d = b_diff.to_dict()
    assert d["circuit_hash_change"] is True
    assert d["regression_cause"] == "PASS_PARAM"
    assert "depth" in d["metric_deltas"]

def test_bundle_diff_schema():
    from jsonschema import validate
    import pathlib
    schema_path = pathlib.Path(__file__).parent.parent / "schemas" / "bundle_diff.schema.json"
    schema = json.loads(schema_path.read_text())
    
    b_diff = BundleDiff(
        metric_deltas={"depth": {"a": 10, "b": 15, "pct_change": 50.0}},
        circuit_hash_change=True,
        regression_cause="PASS_PARAM"
    )
    validate(instance=b_diff.to_dict(), schema=schema)
