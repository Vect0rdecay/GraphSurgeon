from graph_surgeon.analysis.motifs import OPERATOR_REFERENCE_DB
OPERATOR_SECURITY_DB = OPERATOR_REFERENCE_DB

def get_operator_info(op_type: str) -> dict:
    return OPERATOR_REFERENCE_DB.get(op_type, OPERATOR_REFERENCE_DB.get("UNKNOWN", {}))

def list_operators() -> list:
    return sorted(k for k in OPERATOR_REFERENCE_DB if k != "UNKNOWN")
