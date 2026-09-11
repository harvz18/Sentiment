"""Edit names only AFTER inspecting the current model's discovered terms.

Retraining can permute topic IDs; review this mapping each time.
"""
TOPIC_NAMES = {i: f"Unassigned Topic {i}" for i in range(10)}


def topic_name(topic_id: int) -> str:
    return TOPIC_NAMES.get(topic_id, f"Unassigned Topic {topic_id}")
