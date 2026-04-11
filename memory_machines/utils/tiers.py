# Canonical mapping from tier labels to task type labels.
# This defines the 1:1 correspondence between the two rating representations.
from memory_machines.utils.types import MemoryPromptTaskType, MemoryPromptTier


_TASK_TYPES_BY_TIER: dict[MemoryPromptTier, MemoryPromptTaskType] = {
    "T0": "off-target",
    "T1": "needs-refactor",
    "T2": "needs-polish",
    "T3": "ready-to-review",
}

# Inverse mapping from task type labels to tier labels.
_TIERS_BY_TASK_TYPE: dict[MemoryPromptTaskType, MemoryPromptTier] = {
    task_type: tier for tier, task_type in _TASK_TYPES_BY_TIER.items()
}


def get_tier_from_task_type(task_type: MemoryPromptTaskType) -> MemoryPromptTier:
    """
    Convert a task type label to its corresponding tier label.

    Args:
        task_type: A descriptive task type label (e.g. "ready-to-review", "needs-polish")

    Returns:
        The corresponding tier label (e.g. "T3", "T2")

    Example:
        >>> get_tier_from_task_type("ready-to-review")
        "T3"
        >>> get_tier_from_task_type("off-target")
        "T0"
    """
    return _TIERS_BY_TASK_TYPE[task_type]
