"""Idempotency for voice-clip generation.

Generation runs at the top of every job, so it MUST be a no-op on the second run. A
`.voice_clone.json` in the target dataset folder records a fingerprint of the settings
that produced the clips, the list of files this feature created, and the regenerate token
it last consumed.

The four states, and why the last one is an error rather than a guess:

    no manifest              -> generate (normal first run)
    fingerprint matches      -> skip (the common case; resumes cost nothing)
    unconsumed token         -> delete our files, regenerate, record the token
    fingerprint differs      -> ERROR naming what changed

If the reference file is swapped, silently skipping trains the OLD voice and silently
regenerating destroys clips that may have been curated. Both are quietly wrong, so the
run stops before any GPU time is spent, with a message naming the changed field.

The token is a nonce, not a boolean: a "regenerate" checkbox would re-run on every resume
until someone remembered to untick it. The UI stamps a timestamp, the manifest records
what it consumed, and a resume carrying an already-consumed token does nothing.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

MANIFEST_NAME = ".voice_clone.json"

# Fields whose change means the clips no longer match the config. Deliberately excludes
# the reference PATH (moving a file is not a change -- its content hash is what counts).
FINGERPRINT_FIELDS = (
    "mode",
    "instruct",
    "target_seconds",
    "duration_mix",
    "voice_description",
    "trigger_word",
    "backend",
    "seed",
)


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def build_fingerprint(config, dialogue_lines: List[str]) -> Dict:
    """Settings identity. Content-hashes the reference, and hashes the EFFECTIVE dialogue
    so editing one line registers -- otherwise the staleness check has a hole in it."""
    fp = {k: getattr(config, k, None) for k in FINGERPRINT_FIELDS}
    fp["dialogue_hash"] = hashlib.sha256(
        "\n".join(dialogue_lines).encode("utf-8")
    ).hexdigest()[:32]
    ref = getattr(config, "reference_path", None)
    fp["reference_hash"] = (
        _file_hash(ref) if ref and os.path.exists(ref) else None
    )
    return fp


def manifest_path(target_dataset: str) -> str:
    return os.path.join(target_dataset, MANIFEST_NAME)


def read(target_dataset: str) -> Optional[Dict]:
    p = manifest_path(target_dataset)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None  # corrupt manifest: treat as absent and regenerate


def write(target_dataset: str, fingerprint: Dict, files: List[str], token: str) -> None:
    payload = {
        "version": 1,
        "fingerprint": fingerprint,
        "files": sorted(files),
        "consumed_token": token or "",
    }
    with open(manifest_path(target_dataset), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def describe_changes(old_fp: Dict, new_fp: Dict) -> List[str]:
    """Human-readable list of what differs, for the error message."""
    out = []
    for k in sorted(set(old_fp) | set(new_fp)):
        a, b = old_fp.get(k), new_fp.get(k)
        if a != b:
            if k == "reference_hash":
                out.append("reference file (contents differ)")
            elif k == "dialogue_hash":
                out.append("dialogue lines")
            else:
                out.append(f"{k}: {a!r} -> {b!r}")
    return out


def decide(target_dataset: str, fingerprint: Dict, token: str) -> Tuple[str, Optional[Dict]]:
    """-> ('generate' | 'skip' | 'regenerate', manifest_or_None). Raises on stale."""
    m = read(target_dataset)
    if m is None:
        return "generate", None

    token = token or ""
    if token and token != m.get("consumed_token", ""):
        return "regenerate", m

    if m.get("fingerprint") == fingerprint:
        return "skip", m

    changes = describe_changes(m.get("fingerprint", {}), fingerprint)
    raise ValueError(
        "Voice clone settings changed since the clips in "
        f"{target_dataset} were generated:\n  - "
        + "\n  - ".join(changes)
        + "\n\nSkipping would train the OLD voice; regenerating would delete the existing "
        "clips. Neither is safe to guess, so nothing has been changed. Press Regenerate "
        "in the Voice card to rebuild them, or put the setting back."
    )


def delete_generated(target_dataset: str, manifest: Optional[Dict]) -> int:
    """Remove ONLY the files this feature created.

    Never globs the folder: a voice dataset may also hold real recordings cut by hand,
    which train equally well and are not ours to delete.
    """
    if not manifest:
        return 0
    n = 0
    for name in manifest.get("files", []):
        p = os.path.join(target_dataset, name)
        if os.path.exists(p):
            os.remove(p)
            n += 1
        cap = os.path.splitext(p)[0] + ".txt"
        if os.path.exists(cap):
            os.remove(cap)
    return n
