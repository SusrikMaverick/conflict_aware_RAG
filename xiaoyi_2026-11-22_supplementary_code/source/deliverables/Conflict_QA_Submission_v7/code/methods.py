"""Public method names and stable archive identifiers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    name: str
    title: str
    legacy: str


METHOD_REGISTRY = (
    Method("cards", "Docs + Cards", "ours_1024"),
    Method("evidence_prune", "Evidence Pruning", "FR4_M03"),
    Method("memory_cards", "Memory + Cards", "WR5_M02"),
    Method("context_review", "Context Review", "WR3_M01"),
    Method("answer_agreement", "Answer Agreement", "WR2_M08"),
    Method("dual_review", "Dual Review", "WR1_M05"),
)
BASELINE_REGISTRY = (
    Method("plain", "Plain RAG", "plain_1024"),
    Method("madam", "MADAM-RAG", "madam"),
    Method("reflection", "Self-reflection", "self_reflection"),
    Method("astute", "Astute RAG", "astute_structured1024"),
    Method("faithful", "FaithfulRAG", "faithfulrag_adaptation"),
    Method("trust", "TrustRAG", "trust"),
    Method("robust", "RobustRAG (Keyword)", "robust_keyword"),
)
ALL_REGISTRY = BASELINE_REGISTRY + METHOD_REGISTRY
BY_LEGACY = {method.legacy: method for method in ALL_REGISTRY}
BY_NAME = {method.name: method for method in ALL_REGISTRY}
LEGACY_METHODS = [method.legacy for method in METHOD_REGISTRY]


def resolve_method(value):
    """Accept public names and original IDs without changing archived records."""
    if not isinstance(value, str):
        raise ValueError("Method must be a string")
    method = BY_NAME.get(value) or BY_LEGACY.get(value)
    if method is None:
        raise ValueError("Unknown method: " + value)
    return method
