import pytest
import tempfile
from pathlib import Path
from qocc.core.artifacts import ArtifactStore

def test_export_max_size_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(tmpdir)
        store.write_circuit("input.qasm", "A" * (1024 * 1024 + 100)) # > 1MB
        
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            store.export_zip(Path(tmpdir) / "out.zip", max_size_mb=1.0)
            
def test_export_compression_formats():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(tmpdir)
        store.write_circuit("test.qasm", "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];")
        
        # Test basic export with deflate
        zip1 = store.export_zip(Path(tmpdir) / "out_deflate.zip", compress="deflate")
        assert zip1.exists()
        
        # Test export with none
        zip2 = store.export_zip(Path(tmpdir) / "out_none.zip", compress="none")
        assert zip2.exists()
        
        # Test export with zstd fallback
        zip3 = store.export_zip(Path(tmpdir) / "out_zstd.zip", compress="zstd")
        assert zip3.exists()

def test_stream_bundle():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(tmpdir)
        store.write_circuit("test.qasm", "OPENQASM 3.0;")
        
        data = bytearray()
        def callback(b):
            data.extend(b)
            
        store.stream_bundle(callback)
        assert len(data) > 20 # valid zip headers plus content
        assert data.startswith(b"PK")
