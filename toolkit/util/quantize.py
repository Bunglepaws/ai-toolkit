from fnmatch import fnmatch
from typing import List, Optional, Union, TYPE_CHECKING, Tuple
import hashlib
import json
import os
import re
import time
import torch

from optimum.quanto.quantize import _quantize_submodule, quantization_map
from optimum.quanto.tensor import Optimizer, qtype, qtypes
from torchao.quantization.quant_api import (
    quantize_ as torchao_quantize_,
    Float8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
    Int8WeightOnlyConfig
)
from optimum.quanto import freeze
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download

from toolkit.print import print_acc
from toolkit.util.ostris_quant import (
    OstrisLinear,
    OstrisQuantizer,
    convert_linear_to_ostris,
    get_ostris_quantizer,
)

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel

# the quantize function in quanto had a bug where it was using exclude instead of include

Q_MODULES = [
    "QLinear",
    "QConv2d",
    "QEmbedding",
    "QBatchNorm2d",
    "QLayerNorm",
    "QConvTranspose2d",
    "QEmbeddingBag",
    "OstrisLinear",
]


def filter_lora_state_dict_for_quantized_model(
    transformer: "torch.nn.Module",
    lora_state_dict: dict,
) -> dict:
    """Remove LoRA keys targeting quanto-quantized layers.

    PEFT's load_lora_weights with assign=True calls QModule._load_from_state_dict,
    which unconditionally pops weight._data from the state dict — raising KeyError
    if the key isn't present (i.e. when loading a LoRA, not a full checkpoint).

    LoRA key prefixes vary by training tool:
      - kohya / custom: 'diffusion_model.<module>.<lora_A/B>...'
      - diffusers:      'transformer.<module>.<lora_A/B>...'
    Both must be stripped before matching against named_modules() names.
    """
    quantized_modules = set()
    for name, mod in transformer.named_modules():
        if mod.__class__.__name__ in Q_MODULES:
            quantized_modules.add(name)
            # When a training LoRA is already applied, the QLinear lives inside a
            # PEFT LoraLinear as `base_layer`. LoRA keys target the parent path
            # (e.g. "linear_1"), not "linear_1.base_layer", so we add the parent
            # too so the filter correctly strips those keys.
            if name.endswith('.base_layer'):
                quantized_modules.add(name[:-len('.base_layer')])
    if not quantized_modules:
        return lora_state_dict

    _LORA_MARKERS = (
        ".lora_A.", ".lora_B.", ".lora_alpha",
        ".lora_embedding_A.", ".lora_embedding_B.",
    )
    _STRIP_PREFIXES = ("diffusion_model.", "transformer.")

    def _module_name(key: str) -> str:
        for pfx in _STRIP_PREFIXES:
            if key.startswith(pfx):
                key = key[len(pfx):]
                break
        for marker in _LORA_MARKERS:
            i = key.find(marker)
            if i >= 0:
                return key[:i]
        return key

    return {
        k: v for k, v in lora_state_dict.items()
        if _module_name(k) not in quantized_modules
    }

torchao_qtypes = {
    # "int4": Int4WeightOnlyConfig(),
    "uint2": UIntXWeightOnlyConfig(torch.uint2),
    "uint3": UIntXWeightOnlyConfig(torch.uint3),
    "uint4": UIntXWeightOnlyConfig(torch.uint4),
    "uint5": UIntXWeightOnlyConfig(torch.uint5),
    "uint6": UIntXWeightOnlyConfig(torch.uint6),
    "uint7": UIntXWeightOnlyConfig(torch.uint7),
    "uint8": UIntXWeightOnlyConfig(torch.uint8),
    "int8": Int8WeightOnlyConfig(),
    "float8": Float8WeightOnlyConfig(),
}


class aotype:
    def __init__(self, name: str):
        self.name = name
        self.config = torchao_qtypes[name]


class ostristype:
    # custom quantization backend (see toolkit/util/ostris_quant.py), e.g. orbit2/3/4
    def __init__(self, name: str, quantizer: OstrisQuantizer):
        self.name = name
        self.quantizer = quantizer


def get_qtype(qtype: Union[str, qtype]) -> qtype:
    if qtype in torchao_qtypes:
        return aotype(qtype)
    if isinstance(qtype, str):
        ostris_quantizer = get_ostris_quantizer(qtype)
        if ostris_quantizer is not None:
            return ostristype(qtype, ostris_quantizer)
        return qtypes[qtype]
    else:
        return qtype


def is_quantized_tensor(t) -> bool:
    # torchao stores quantized weights as tensor subclasses (e.g. AffineQuantizedTensor) under torchao.*
    # that still report as nn.Parameter and expose .dequantize(). (quanto is handled separately.)
    # _is_ostris_weight tags two OstrisLinear tensors: the .weight property's eager tensor
    # (already dequantized; .dequantize() is a no-op) so the merge paths route through
    # requantize_module_weight, and the lazy OstrisLazyWeight emitted by state_dict()
    # (holds no data; .dequantize() materializes) so save loops dequantize it per key.
    if getattr(t, '_is_ostris_weight', False):
        return True
    return 'torchao' in type(t).__module__ and hasattr(t, 'dequantize')


def dequantize_if_quantized(t):
    return t.dequantize() if is_quantized_tensor(t) else t


def get_torchao_config(qtype):
    # returns the requantization config for a given qtype string (a torchao config, or the
    # ostristype for custom backends), or None if the qtype supports neither
    if qtype is None:
        return None
    try:
        q = get_qtype(qtype)
    except Exception:
        return None
    if isinstance(q, aotype):
        return q.config
    if isinstance(q, ostristype):
        return q
    return None


def requantize_module_weight(module, fp_weight, orig_dtype, config) -> None:
    """Write a full precision weight back into module.weight, re-quantizing in place if a
    requantization config is provided so the module stays quantized (used by the continuous
    merge/reset method). If config is None the weight is left in full precision."""
    if isinstance(module, OstrisLinear):
        # the module's backend reuses its existing quantization state; config is not needed
        module.requantize_(fp_weight)
        return
    if isinstance(config, ostristype):
        # custom backend config but the module was never converted (e.g. skipped at
        # quantize time); leave it in full precision
        config = None
    module.weight = torch.nn.Parameter(fp_weight.to(orig_dtype), requires_grad=False)
    if config is not None:
        torchao_quantize_(module, config)


def quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype, aotype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    quantize_device: Optional[torch.device] = None,
):
    """Quantize the specified model submodules

    Recursively quantize the submodules of the specified parent model.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.

    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Note: quantization happens in-place and modifies the original model and its descendants.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
        quantize_device (`Optional[torch.device]`):
            If provided, each module is moved to this device to quantize, then moved
            back to the device its weights were on initially. Lets a CPU-resident
            model (low vram) quantize layer-by-layer on the GPU.
    """
    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    for name, m in model.named_modules():
        if include is not None and not any(
            fnmatch(name, pattern) for pattern in include
        ):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        try:
            # check if m is QLinear or QConv2d
            if m.__class__.__name__ in Q_MODULES:
                continue
            if (
                isinstance(weights, aotype)
                and not isinstance(m, torch.nn.Linear)
                and (
                    quantize_device is not None
                    or include is not None
                    or exclude is not None
                )
            ):
                # torchao only quantizes nn.Linear; when a device round-trip or
                # include/exclude filtering is in play, skip containers so each
                # linear is handled individually (a container-level torchao call
                # would quantize excluded children too)
                continue
            orig_device = None
            if quantize_device is not None and next(m.children(), None) is None:
                param = next(m.parameters(recurse=False), None)
                if param is not None:
                    orig_device = param.device
                    m.to(quantize_device)
            try:
                if isinstance(weights, ostristype):
                    if isinstance(m, torch.nn.Linear):
                        convert_linear_to_ostris(m, weights.quantizer)
                elif isinstance(weights, aotype):
                    torchao_quantize_(m, weights.config)
                else:
                    _quantize_submodule(
                        model,
                        name,
                        m,
                        weights=weights,
                        activations=activations,
                        optimizer=optimizer,
                    )
            finally:
                if orig_device is not None:
                    # quanto replaces the module in its parent, so re-fetch by name
                    model.get_submodule(name).to(orig_device)
        except Exception as e:
            print(f"Failed to quantize {name}: {e}")
            # raise e


def _compute_quant_cache_key(base_model: "BaseModel", extra: str = "") -> Tuple[str, str]:
    """Return (slug, hash20) for the quantization cache filename.

    The slug is a sanitised basename of name_or_path so cache files are
    human-readable and clearly tied to a specific checkpoint. `extra`
    disambiguates multiple quantized submodels of one base model (e.g.
    wan2.2 14b high/low noise transformers).
    """
    name = base_model.model_config.name_or_path
    qtype_name = base_model.model_config.qtype
    parts = [name, qtype_name, "v1"]
    if extra:
        parts.append(extra)
    if os.path.isfile(name) or os.path.isdir(name):
        try:
            parts.append(str(int(os.path.getmtime(name))))
        except OSError:
            pass
    hash20 = hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]
    # human-readable slug from the last path component (no extension)
    basename = os.path.splitext(os.path.basename(name))[0] or name
    if extra:
        # keep the slug human-readable, e.g. ..._high-noise_...
        extra_slug = re.sub(r'[^A-Za-z0-9._-]', '-', extra.split('|')[0])[:20].strip('-')
        if extra_slug:
            basename = f"{basename}_{extra_slug}"
    slug = re.sub(r'[^A-Za-z0-9._-]', '-', basename)[:40].strip('-')
    return slug, hash20


def _save_to_quant_cache(
    model_to_quantize: torch.nn.Module,
    cache_path: str,
    cache_qmap_path: str,
):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    raw_state = model_to_quantize.orig_state_dict()
    raw_state = {k: v.cpu().contiguous() for k, v in raw_state.items()}
    save_file(raw_state, cache_path)
    qmap = quantization_map(model_to_quantize)
    with open(cache_qmap_path, "w", encoding="utf-8") as f:
        json.dump(qmap, f, indent=2)


def _load_from_quant_cache(
    model_to_quantize: torch.nn.Module,
    cache_path: str,
    cache_qmap_path: str,
    stop_check=None,
):
    with open(cache_qmap_path, "r", encoding="utf-8") as f:
        qmap = json.load(f)

    # Rebuild QLinear module structure from the qmap.
    #
    # The naive approach calls _quantize_submodule per module, which quantizes
    # the existing full-size weight — that's as slow as re-quantizing from scratch.
    # Instead, replace each weight with a 1×1 dummy before the call so the
    # quantization math is instant.  quanto's _load_from_state_dict then
    # reconstructs the full QBytesTensor from _data/_scale in the cache,
    # completely replacing the dummy regardless of its shape.
    quantized_items = [
        (name, m) for name, m in model_to_quantize.named_modules()
        if name in qmap
    ]
    t_rebuild = time.time()
    for name, m in tqdm(quantized_items, desc=" - rebuilding quantization structure"):
        if stop_check is not None:
            stop_check()
        qconfig = qmap[name]
        w = qconfig["weights"]
        a = qconfig["activations"]
        weights_qt = get_qtype(w) if w != "none" else None
        activations_qt = get_qtype(a) if a != "none" else None
        if weights_qt is not None and not isinstance(weights_qt, aotype):
            # Temporarily swap in a 1×1 dummy weight so _quantize_submodule
            # runs the quantization math on a single element (instant) instead
            # of the full multi-MB weight tensor.  in_features/out_features are
            # read from module attributes, not weight shape, so QLinear metadata
            # stays correct.  The real weights come from load_state_dict below.
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight = torch.nn.Parameter(
                    torch.zeros(1, 1, dtype=m.weight.dtype, device='cpu'),
                    requires_grad=False,
                )
            _quantize_submodule(model_to_quantize, name, m, weights=weights_qt, activations=activations_qt)

    print_acc(f" - structure rebuild: {time.time() - t_rebuild:.1f}s")
    if stop_check is not None:
        stop_check()
    # Load the frozen QTensor state — QModuleMixin._load_from_state_dict reconstructs
    # QBytesTensors from _data/_scale without re-running any quantization math.
    t_load = time.time()
    print_acc(f" - loading quantized weights from cache ({os.path.basename(cache_path)})...")
    state_dict = load_file(cache_path, device="cpu")
    print_acc(f" - cache file read: {time.time() - t_load:.1f}s")
    t_apply = time.time()
    print_acc(" - applying weights to model...")
    model_to_quantize.load_state_dict(state_dict, strict=False)
    print_acc(f" - weights applied: {time.time() - t_apply:.1f}s")
    freeze(model_to_quantize)


def has_quant_cache(base_model: "BaseModel", cache_key_extra: str = "") -> bool:
    """Return True if a usable quantization cache file exists for this model.

    Uses a glob fallback for the case where the source file no longer exists:
    when the file is present its mtime is mixed into the cache key, but when
    the file is gone we can't reconstruct that key exactly, so we look for any
    cache file whose slug matches.
    """
    import glob as _glob
    cache_dir = os.environ.get("AITK_QUANTIZATION_CACHE_DIR", None)
    if not cache_dir or not getattr(base_model.model_config, "cache_quantized_model", False):
        return False
    quantization_type = get_qtype(base_model.model_config.qtype)
    is_torchao = isinstance(quantization_type, aotype)
    slug, cache_key = _compute_quant_cache_key(base_model, cache_key_extra)
    ext = "pt" if is_torchao else "safetensors"
    exact = os.path.join(cache_dir, f"quant_{slug}_{cache_key}.{ext}")
    if os.path.exists(exact):
        return True
    # Source file may be gone so its mtime wasn't in cache_key, but it was when
    # the cache was written. Fall back to any slug match.
    matches = _glob.glob(os.path.join(cache_dir, f"quant_{slug}_*.{ext}"))
    return len(matches) > 0


def _resolve_cache_path(
    base_model: "BaseModel",
    cache_dir: str,
    slug: str,
    cache_key: str,
    ext: str,
) -> Optional[str]:
    """Return the cache file to load for this model, or None if there isn't one.

    Prefers the exact mtime-keyed filename. Falls back to a slug glob ONLY when the
    source checkpoint is missing: in that case its mtime could not be mixed into the
    key at load time (it was at save time), so the exact name won't match. We never
    glob-fall-back while the source exists, so a changed source (different mtime)
    correctly forces a re-quantize rather than loading a stale cache.

    This mirrors has_quant_cache(); the two MUST agree, otherwise has_quant_cache
    green-lights skipping the source load while quantize_model then fails to find
    the cache and quantizes an uninitialised (from_config) model.
    """
    import glob as _glob
    exact = os.path.join(cache_dir, f"quant_{slug}_{cache_key}.{ext}")
    if os.path.exists(exact):
        return exact
    name = base_model.model_config.name_or_path
    source_present = os.path.isfile(name) or os.path.isdir(name)
    if source_present:
        return None
    matches = sorted(
        _glob.glob(os.path.join(cache_dir, f"quant_{slug}_*.{ext}")),
        key=os.path.getmtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _save_torchao_cache(
    model_to_quantize: torch.nn.Module,
    cache_path: str,
):
    """Save a torchao-quantized model's state dict via torch.save.

    torchao quantized weights are tensor subclasses (AffineQuantizedTensor etc.)
    that carry their own quantization metadata. safetensors cannot store them,
    but torch.save/load handles them natively via pickle.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    state = {k: v.cpu() for k, v in model_to_quantize.state_dict().items()}
    torch.save(state, cache_path)


def _load_torchao_cache(
    model_to_quantize: torch.nn.Module,
    cache_path: str,
    stop_check=None,
):
    """Restore a torchao-quantized model from a torch.save cache file.

    The state dict contains torchao tensor subclasses; load_state_dict with
    assign=True is required so PyTorch replaces parameter storage in-place
    rather than trying to copy into plain tensors.
    """
    if stop_check is not None:
        stop_check()
    print_acc(" - loading torchao quantized weights from cache...")
    state = torch.load(cache_path, map_location="cpu", weights_only=False)
    if stop_check is not None:
        stop_check()
    print_acc(" - applying torchao weights to model...")
    model_to_quantize.load_state_dict(state, strict=True, assign=True)


def quantize_model(
    base_model: "BaseModel",
    model_to_quantize: torch.nn.Module,
    cache_key_extra: str = "",
):
    from toolkit.dequantize import patch_dequantization_on_save

    if not hasattr(base_model, "get_transformer_block_names"):
        raise ValueError(
            "The model to quantize must have a method `get_transformer_block_names`."
        )

    # patch the state dict method
    patch_dequantization_on_save(model_to_quantize)

    # sensitive modules to keep in full precision (fnmatch patterns)
    exclude_modules = base_model.get_quantization_exclude_modules() or []

    if base_model.model_config.accuracy_recovery_adapter is not None:
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork

        # we need to load and quantize with an accuracy recovery adapter
        # todo handle hf repos
        load_lora_path = base_model.model_config.accuracy_recovery_adapter

        if not os.path.exists(load_lora_path):
            # not local file, grab from the hub

            path_split = load_lora_path.split("/")
            if len(path_split) > 3:
                raise ValueError(
                    "The accuracy recovery adapter path must be a local path or for a hf repo, 'username/repo_name/filename.safetensors'."
                )
            repo_id = f"{path_split[0]}/{path_split[1]}"
            print_acc(f"Grabbing lora from the hub: {load_lora_path}")
            new_lora_path = hf_hub_download(
                repo_id,
                filename=path_split[-1],
            )
            # replace the path
            load_lora_path = new_lora_path

        # build the lora config based on the lora weights
        lora_state_dict = load_file(load_lora_path)
        
        if hasattr(base_model, "convert_lora_weights_before_load"):
            lora_state_dict = base_model.convert_lora_weights_before_load(lora_state_dict)
        
        network_config = {
            "type": "lora",
            "network_kwargs": {"only_if_contains": []},
            "transformer_only": False,
        }
        first_key = list(lora_state_dict.keys())[0]
        first_weight = lora_state_dict[first_key]
        # if it starts with lycoris and includes lokr
        if first_key.startswith("lycoris") and any(
            "lokr" in key for key in lora_state_dict.keys()
        ):
            network_config["type"] = "lokr"
        
        network_kwargs = {}

        # find firse loraA weight
        if network_config["type"] == "lora":
            linear_dim = None
            for key, value in lora_state_dict.items():
                if "lora_A" in key:
                    linear_dim = int(value.shape[0])
                    break
            linear_alpha = linear_dim
            network_config["linear"] = linear_dim
            network_config["linear_alpha"] = linear_alpha

            # we build the keys to match every key
            only_if_contains = []
            for key in lora_state_dict.keys():
                contains_key = key.split(".lora_")[0]
                if contains_key not in only_if_contains:
                    only_if_contains.append(contains_key)

            network_kwargs["only_if_contains"] = only_if_contains
        elif network_config["type"] == "lokr":
            # find the factor
            largest_factor = 0
            for key, value in lora_state_dict.items():
                if "lokr_w1" in key:
                    factor = int(value.shape[0])
                    if factor > largest_factor:
                        largest_factor = factor
            network_config["lokr_full_rank"] = True
            network_config["lokr_factor"] = largest_factor

            only_if_contains = []
            for key in lora_state_dict.keys():
                if "lokr_w1" in key:
                    contains_key = key.split(".lokr_w1")[0]
                    contains_key = contains_key.replace("lycoris_", "")
                    if contains_key not in only_if_contains:
                        only_if_contains.append(contains_key)
            network_kwargs["only_if_contains"] = only_if_contains
        
        if hasattr(base_model, 'target_lora_modules'):
            network_kwargs['target_lin_modules'] = base_model.target_lora_modules

        # todo auto grab these
        # get dim and scale
        network_config = NetworkConfig(**network_config)

        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=model_to_quantize,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            # conv_lora_dim=self.network_config.conv,
            # conv_alpha=self.network_config.conv_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=base_model.is_transformer,
            base_model=base_model,
            is_ara=True,
            **network_kwargs
        )
        network.apply_to(
            None, model_to_quantize, apply_text_encoder=False, apply_unet=True
        )
        network.force_to(base_model.device_torch, dtype=base_model.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)
        network.eval()
        network.is_active = True
        network.can_merge_in = False
        base_model.accuracy_recovery_adapter = network

        # quantize it
        lora_exclude_modules = []
        quantization_type = get_qtype(base_model.model_config.qtype)
        for lora_module in tqdm(network.unet_loras, desc="Attaching quantization"):
            # the lora has already hijacked the original module
            orig_module = lora_module.org_module[0]
            orig_module.to(base_model.torch_dtype)
            # make the params not require gradients
            for param in orig_module.parameters():
                param.requires_grad = False
            quantize(orig_module, weights=quantization_type)
            freeze(orig_module)
            module_name = lora_module.lora_name.replace('$$', '.').replace('transformer.', '')
            lora_exclude_modules.append(module_name)
            if base_model.model_config.low_vram:
                # move it back to cpu
                orig_module.to("cpu")
        pass
        # quantize additional layers
        print_acc(" - quantizing additional layers")
        quantization_type = get_qtype('uint8')
        quantize(
            model_to_quantize,
            weights=quantization_type,
            exclude=lora_exclude_modules + exclude_modules
        )
    else:
        # quantize model the original way without an accuracy recovery adapter
        # move and quantize only certain pieces at a time.
        quantization_type = get_qtype(base_model.model_config.qtype)

        # Cache check: skip GPU quantization if a pre-computed cache exists
        cache_dir = os.environ.get("AITK_QUANTIZATION_CACHE_DIR", None)
        wants_cache = getattr(base_model.model_config, "cache_quantized_model", False)
        is_torchao = isinstance(quantization_type, aotype)
        use_cache = bool(cache_dir and wants_cache)
        if wants_cache and not cache_dir:
            base_model.print_and_status_update(
                f" - Note: cache_quantized_model is enabled but AITK_QUANTIZATION_CACHE_DIR "
                f"is not set. Configure a cache directory in Settings."
            )
        cache_path = cache_qmap_path = None
        if use_cache:
            slug, cache_key = _compute_quant_cache_key(base_model, cache_key_extra)
            ext = "pt" if is_torchao else "safetensors"
            # Path to save a freshly-quantized model to (exact key). The load side may
            # resolve to a glob-matched file when the source checkpoint is missing.
            resolved = _resolve_cache_path(base_model, cache_dir, slug, cache_key, ext)
            if is_torchao:
                # torchao: single .pt file (pickle-based; handles tensor subclasses)
                cache_path = os.path.join(cache_dir, f"quant_{slug}_{cache_key}.pt")
                cache_qmap_path = None
                if resolved is not None:
                    try:
                        base_model.print_and_status_update(" - loading cached torchao quantized model...")
                        _load_torchao_cache(model_to_quantize, resolved,
                                            stop_check=base_model.maybe_stop)
                        base_model.print_and_status_update(" - cached torchao quantized model loaded")
                        return
                    except Exception as e:
                        base_model.print_and_status_update(
                            f" - torchao cache load failed ({e}), re-quantizing from scratch"
                        )
            else:
                # quanto: safetensors + qmap JSON
                cache_path = os.path.join(cache_dir, f"quant_{slug}_{cache_key}.safetensors")
                cache_qmap_path = os.path.join(cache_dir, f"quant_{slug}_{cache_key}_qmap.json")
                if resolved is not None:
                    # qmap sits beside the safetensors with the same key
                    resolved_qmap = resolved[: -len(".safetensors")] + "_qmap.json"
                    if os.path.exists(resolved_qmap):
                        try:
                            base_model.print_and_status_update(" - loading cached quantized model...")
                            _load_from_quant_cache(model_to_quantize, resolved, resolved_qmap,
                                                   stop_check=base_model.maybe_stop)
                            base_model.print_and_status_update(" - cached quantized model loaded")
                            return
                        except Exception as e:
                            base_model.print_and_status_update(
                                f" - cache load failed ({e}), re-quantizing from scratch"
                            )

        # all_blocks = list(model_to_quantize.transformer_blocks)
        all_blocks: List[torch.nn.Module] = []
        transformer_block_names = base_model.get_transformer_block_names()
        for name in transformer_block_names:
            # name may be a dotted path for models that nest their blocks
            # (e.g. hidream_o1's "model.language_model.layers").
            block_list = model_to_quantize
            for part in name.split('.'):
                block_list = getattr(block_list, part, None)
                if block_list is None:
                    break
            if block_list is not None:
                all_blocks += list(block_list)
        base_model.print_and_status_update(
            f" - quantizing {len(all_blocks)} transformer blocks"
        )
        for block in tqdm(all_blocks):
            base_model.maybe_stop()
            block.to(base_model.device_torch, dtype=base_model.torch_dtype, non_blocking=True)
            quantize(block, weights=quantization_type)
            freeze(block)
            # NOT non_blocking: an async D2H allocates the cpu destination in pinned
            # memory, which the caching host allocator keeps forever (with power-of-2
            # bucket rounding on top) — that silently retained a model-sized chunk of
            # host ram after the weights moved back to the gpu for training
            block.to("cpu")

        # Quantize the extras (non-transformer-block children) that weren't handled block-by-block above.
        # We skip the already-frozen transformer block lists to avoid rescanning thousands of frozen
        # sub-modules, which would be extremely slow on large models like LTX2-22B.
        base_model.print_and_status_update(" - quantizing extras")
        # model_to_quantize.to(base_model.device_torch, dtype=base_model.torch_dtype)
        quantize(model_to_quantize, weights=quantization_type, exclude=exclude_modules)
        freeze(model_to_quantize)

        # Save to cache after successful quantization
        if use_cache and cache_path:
            try:
                base_model.print_and_status_update(" - saving quantization cache...")
                if is_torchao:
                    _save_torchao_cache(model_to_quantize, cache_path)
                else:
                    _save_to_quant_cache(model_to_quantize, cache_path, cache_qmap_path)
                base_model.print_and_status_update(f" - quantization cache saved to {cache_path}")
            except Exception as e:
                base_model.print_and_status_update(f" - warning: failed to save quantization cache: {e}")
