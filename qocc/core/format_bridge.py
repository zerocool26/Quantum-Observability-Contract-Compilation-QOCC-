import logging
from typing import Any
from qocc.core.circuit_handle import CircuitHandle
from qocc.adapters.base import get_adapter
from qocc.trace.emitter import TraceEmitter

logger = logging.getLogger(__name__)

def convert(circuit_handle: CircuitHandle, target_format: str, emitter: TraceEmitter | None = None) -> Any:
    """
    Convert a given CircuitHandle to the requested target format.
    
    Args:
        circuit_handle: The circuit.
        target_format: The format to export to ("qasm2", "qasm3", "qiskit", "cirq", "tket", "stim").
    
    Returns:
        A string (for QASM/Stim) or native object (Qiskit QuantumCircuit, etc).
    """
    
    # We attempt to load the target adapter if format maps to an adapter
    adapter_map = {
        "qiskit": "qiskit",
        "cirq": "cirq",
        "tket": "tket",
        "stim": "stim"
    }
    
    source_format = circuit_handle.source_format
    
    # Track metrics if possible
    lossy_formats = {"stim"} # stim loses non-cliffords usually
    is_lossy = target_format in lossy_formats and source_format not in lossy_formats
    
    span_attrs = {
        "source_format": source_format,
        "target_format": target_format,
        "is_lossy": is_lossy
    }
    
    result = None
    
    with (emitter.span("format_bridge/convert", attributes=span_attrs) if emitter else _null_context()):
        try:
            # If target is qasm2 or qasm3, we can just use the source adapter to export it
            if target_format in ("qasm2", "qasm3"):
                source_adapter = get_adapter(source_format) if source_format in adapter_map else get_adapter("qiskit")
                result = source_adapter.export(circuit_handle, fmt=target_format)
            elif target_format in adapter_map:
                # To convert between memory formats, we typically go through QASM3 as intermediate bridge
                # unless source is already what we need
                if source_format == target_format:
                    result = circuit_handle.native_circuit
                else:
                    intermediate_adapter = get_adapter(source_format) if source_format in adapter_map else get_adapter("qiskit")
                    qasm3_str = intermediate_adapter.export(circuit_handle, fmt="qasm3")
                    
                    target_adapter = get_adapter(adapter_map[target_format])
                    new_handle = target_adapter.ingest(qasm3_str)
                    result = new_handle.native_circuit
            else:
                raise ValueError(f"Unknown target format: {target_format}")
                
        except Exception as e:
            if emitter:
                emitter.emit_event("WARNING", attributes={"error": str(e), "message": "Conversion loss or failure"})
            raise RuntimeError(f"Format bridge conversion failed: {e}") from e
            
        if is_lossy and emitter:
            emitter.emit_event("WARNING", attributes={"message": f"Conversion to {target_format} is lossy", "lost_operations": ["non-clifford"]})
            
    return result

class _null_context:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_val, exc_tb): pass