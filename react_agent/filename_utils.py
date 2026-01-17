import re


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe file stem for anime names."""
    if not name:
        return "unknown"

    safe = name.strip()
    safe = re.sub(r"[\\/]+", "_", safe)
    safe = re.sub(r'[:*?"<>|]', "_", safe)
    safe = safe.replace("\0", "")
    safe = re.sub(r"\s+", " ", safe).strip()
    if not safe:
        return "unknown"

    if len(safe) > 150:
        safe = safe[:150].rstrip("_ ")

    return safe or "unknown"
