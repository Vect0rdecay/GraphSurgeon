"""GraphSurgeon: ONNX DAG reverse-engineering toolkit."""

from graph_surgeon._env import configure_runtime_quiet

configure_runtime_quiet()

__version__ = "0.1.0"

__all__ = [
    "GraphSurgeon",
    "GraphTopology",
    "GraphTopologyConfig",
    "GraphValidationLevel",
    "GraphValidationResult",
    "LayerPosition",
    "NodeTopology",
    "__version__",
]

_LAZY_EXPORTS = {
    "GraphSurgeon": ("graph_surgeon.graph.surgeon", "GraphSurgeon"),
    "GraphTopology": ("graph_surgeon.graph.topology", "GraphTopology"),
    "GraphTopologyConfig": ("graph_surgeon.graph.topology", "GraphTopologyConfig"),
    "LayerPosition": ("graph_surgeon.graph.topology", "LayerPosition"),
    "NodeTopology": ("graph_surgeon.graph.topology", "NodeTopology"),
    "GraphValidationLevel": ("graph_surgeon.graph.validation", "GraphValidationLevel"),
    "GraphValidationResult": ("graph_surgeon.graph.validation", "GraphValidationResult"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
