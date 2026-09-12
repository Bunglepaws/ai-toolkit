"""
Disk cache for text encoder / VAE work that is identical on every run.

Startup used to spend ~50s shuttling the 8.3GB text encoder to the GPU just to
re-encode prompts that had not changed since the last run (see the startup
timing notes). The embeddings are deterministic given the model and the prompt,
so they can live on disk instead.

The failure mode here is silent: a stale hit means training against embeddings
for the wrong prompt or the wrong model, and nothing crashes. So the key is
deliberately over-broad — the whole model config goes into it — and the full key
material is stored *inside* the payload and re-compared on load. A hit therefore
requires both the filename hash and the recorded key material to match.

Set AITK_DISABLE_EMBED_CACHE=1 to bypass it entirely.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional

import torch

from toolkit.print import print_acc

# Bump this whenever the *meaning* of a cached payload changes (a different
# encode path, a different tensor layout). It invalidates every entry.
# v3: static embeds (unconditional / blank / trigger / DOP class) are now
# encoded as plain text via encode_static_prompt, falling back to a blank
# control image only for models that cannot encode without one. Entries
# written before this were encoded against a solid black control.
CACHE_FORMAT_VERSION = 3

CACHE_DIRNAME = '_embed_cache'


def cache_disabled() -> bool:
    return os.environ.get('AITK_DISABLE_EMBED_CACHE', '0').lower() in ('1', 'true', 'yes')


def _json_default(obj: Any) -> str:
    # key material only needs to be stable and distinguishing, not readable
    return repr(obj)


def _canonical(key_material: Dict[str, Any]) -> str:
    return json.dumps(key_material, sort_keys=True, default=_json_default)


def build_key_material(
    model_config: Any,
    kind: str,
    **extra: Any,
) -> Dict[str, Any]:
    """Assemble the dict that identifies a cache entry.

    The entire model config is included rather than a curated subset of the
    fields that "should" matter. Over-invalidating costs one re-encode; under-
    invalidating trains against the wrong embeddings.
    """
    try:
        model_dict = dict(vars(model_config))
    except TypeError:
        model_dict = {'repr': repr(model_config)}
    return {
        'version': CACHE_FORMAT_VERSION,
        'kind': kind,
        'model_config': model_dict,
        **extra,
    }


def file_stamp(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Identify a file for key material: path plus mtime and size.

    Cheap enough to run on every startup, unlike hashing the contents. The gap
    is a file rewritten with both its mtime and its exact size preserved.
    """
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return {'path': path, 'missing': True}
    return {'path': path, 'mtime_ns': stat.st_mtime_ns, 'size': stat.st_size}


def cache_path(cache_root: str, key_material: Dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical(key_material).encode('utf-8')).hexdigest()[:32]
    kind = str(key_material.get('kind', 'embed'))
    return os.path.join(cache_root, CACHE_DIRNAME, f'{kind}_{digest}.pt')


def exists(cache_root: str, key_material: Dict[str, Any]) -> bool:
    """Is there an entry for this key? Decision-only — cheaper than load().

    A True here is not a promise: load() still re-verifies the key material and
    can degrade to a miss. Only use this to decide, never to skip verification.
    """
    if cache_disabled():
        return False
    return os.path.exists(cache_path(cache_root, key_material))


def load(cache_root: str, key_material: Dict[str, Any]) -> Optional[Any]:
    """Return the cached payload, or None on any miss/corruption/mismatch."""
    if cache_disabled():
        return None
    path = cache_path(cache_root, key_material)
    if not os.path.exists(path):
        return None
    try:
        blob = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print_acc(f"Embed cache: ignoring unreadable entry {os.path.basename(path)} ({e})")
        return None
    if not isinstance(blob, dict) or 'key_material' not in blob:
        return None
    # the hash matched, but verify the key material itself before trusting it
    if _canonical(blob['key_material']) != _canonical(key_material):
        print_acc(f"Embed cache: key mismatch on {os.path.basename(path)}, re-encoding")
        return None
    return blob.get('payload')


def save(cache_root: str, key_material: Dict[str, Any], payload: Any) -> None:
    if cache_disabled():
        return
    path = cache_path(cache_root, key_material)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f'{path}.tmp'
    try:
        torch.save({'key_material': key_material, 'payload': payload}, tmp_path)
        # os.replace used to fail across /mnt/c under WSL; fall back to a copy in place.
        # Harmless on native Windows - kept as a fallback, no longer a live concern.
        try:
            os.replace(tmp_path, path)
        except OSError:
            with open(tmp_path, 'rb') as src, open(path, 'wb') as dst:
                dst.write(src.read())
            os.remove(tmp_path)
    except Exception as e:
        print_acc(f"Embed cache: could not write {os.path.basename(path)} ({e})")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def prompt_embeds_to_dict(embeds: Any) -> Dict[str, Any]:
    """Flatten a prompt embed container into plain tensors for torch.save.

    Models return one of two containers: the classic `PromptEmbeds`
    (text/pooled/mask) or `AdvancedPromptEmbeds` (an open key -> list-of-tensors
    store, used by krea2 among others). Both round-trip through here.
    """
    from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
    if isinstance(embeds, AdvancedPromptEmbeds):
        return {
            '__kind__': 'advanced',
            'store': {k: list(embeds[k]) for k in embeds.keys()},
            'frozen_dtype_keys': list(embeds.frozen_dtype_keys),
        }
    return {
        '__kind__': 'prompt_embeds',
        'text_embeds': embeds.text_embeds,
        'pooled_embeds': embeds.pooled_embeds,
        'attention_mask': embeds.attention_mask,
    }


def prompt_embeds_from_dict(data: Dict[str, Any]) -> Any:
    if data.get('__kind__') == 'advanced':
        from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
        embeds = AdvancedPromptEmbeds(**data['store'])
        embeds.frozen_dtype_keys = data.get('frozen_dtype_keys') or []
        return embeds

    from toolkit.prompt_utils import PromptEmbeds
    pooled = data.get('pooled_embeds')
    text = data['text_embeds']
    # PromptEmbeds unpacks a list/tuple as (text, pooled), so a list of text
    # embeds with no pooled embeds still has to be wrapped — same as clone()
    if pooled is not None or isinstance(text, (list, tuple)):
        args = [text, pooled]
    else:
        args = text
    return PromptEmbeds(args, attention_mask=data.get('attention_mask'))
