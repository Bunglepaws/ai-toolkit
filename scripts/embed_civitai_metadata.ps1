# ── SETTINGS ── edit these, then just hit Run in PowerShell ISE ───────────────

$folder            = "D:\Data\iCloudDrive\Comfy\260710"
#$folder            = "D:\Data\iCloudDrive\Comfy\camera_lab"
$overwriteMetadata = $false    # re-embed even if a parameters chunk already exists
$includeVideos     = $true    # also process .mp4 files (ffmpeg/ffprobe required)
$recursive         = $false    # also process files in subfolders

# embed_civitai_metadata.ps1
#
# Reads the ComfyUI metadata embedded in each PNG (and, with -IncludeVideos,
# each MP4) and writes an A1111-format "parameters" entry that CivitAI reads
# automatically on upload.
#   • PNG: a "parameters" tEXt chunk (via Pillow).
#   • MP4: a "parameters" (and "comment") format tag (via ffmpeg re-mux,
#     -movflags use_metadata_tags) — the same ffprobe-readable tag CivitAI and
#     ai-toolkit's own video sampler use. ComfyUI stores workflow/prompt as MP4
#     format tags, so the SAME graph trace below applies to video.
#
# ── HOW THE PROMPT IS FOUND (the robust part) ─────────────────────────────────
#
# Instead of guessing at a node by name, this script TRACES the graph to find
# the exact text that was fed into the sampler:
#
#   1. Pick the Save node that produced THIS file (matches filename_prefix, so
#      multi-stage graphs with several SaveImage nodes resolve to the right one).
#   2. Walk that Save node's image input back to the sampler that fed it.
#   3. From sampler.positive, walk the conditioning chain to the text encoder
#      (CLIPTextEncode, TextEncodeQwenImageEditPlus, TextEncodeBooguEdit, ...).
#   4. Resolve the encoder's text input THROUGH the graph: switches
#      (ComfySwitchNode / SimpleSelectorSwitch / rgthree Any Switch /
#      LenientSwitch), StringConcatenate (so LoRA trigger words are included),
#      StringReplace, PreviewAny passthroughs, and primitive string nodes — all
#      the way down to the literal prompt string.
#
# It works on the API `prompt` JSON, which ComfyUI flattens (subgraph nodes
# appear as "30:6") and strips of bypassed/muted nodes, so disconnected or
# bypassed nodes simply aren't on the path.
#
# ── THE DYNAMIC-PROMPT CASE (why the CoachBate node matters) ───────────────────
#
# When the prompt is generated dynamically upstream (LLM expander, batch
# prompter), ComfyUI does NOT persist that text on the encoder. Your
# "CoachBate Text Preview and Edit" node captures the resolved single line and
# stores it in the workflow JSON. So when the API graph's captured value is
# empty, the trace falls back to that node's stored value in the workflow JSON.
#
# Two safety rules (so a WRONG prompt is never embedded):
#   • CoachBateBatchPrompter holds the ENTIRE list of prompts (fired one line
#     per image). It is NEVER used as the prompt — if a trace can only reach the
#     batch prompter, the file is SKIPPED and logged.
#   • If the encoder text can't be resolved to a literal (dynamic value that was
#     never captured), the file is SKIPPED and logged — nothing is written.
#
# Generation settings (Steps, Sampler, Schedule, CFG, Seed, Size) are read from
# the actual sampler chain and appended in A1111 format. Size comes from the
# real image dimensions. Existing ComfyUI metadata (workflow, prompt) is
# preserved unchanged; only the `parameters` chunk is written. Re-running is
# safe and idempotent — the trace always re-derives from workflow/prompt, never
# from a previously written `parameters` chunk.



# WSL distro to use (must have the ai-toolkit venv with Pillow installed)
$wslDistro = "Ubuntu-22.04"
$pythonExe = "/mnt/c/Data/git/AIToolkitWSL/ai-toolkit/venv/bin/python3"

# ── CONVERT PATH TO WSL FORMAT ────────────────────────────────────────────────

$wslOverwrite = if ($overwriteMetadata) { "1" } else { "0" }
$wslVideos    = if ($includeVideos)     { "1" } else { "0" }
$wslRecursive = if ($recursive)         { "1" } else { "0" }
$wslFolder = $folder -replace '\\', '/'
if ($wslFolder -match '^([A-Za-z]):(.*)') {
    $drive = $Matches[1].ToLower()
    $rest  = $Matches[2]
    $wslFolder = "/mnt/$drive$rest"
}

# ── PYTHON SCRIPT (piped to python via stdin) ─────────────────────────────────

$pythonScript = @'
import sys, json, os, re, subprocess, shutil
from PIL import Image, PngImagePlugin

FOLDER         = sys.argv[1]
OVERWRITE      = (sys.argv[2] == "1") if len(sys.argv) > 2 else False
INCLUDE_VIDEOS = (sys.argv[3] == "1") if len(sys.argv) > 3 else False
RECURSIVE      = (sys.argv[4] == "1") if len(sys.argv) > 4 else False

# Encoder text-input field names, in priority order. Covers CLIPTextEncode
# (text), Qwen/Boogu (prompt), Flux (t5xxl/clip_l), Wan (positive_prompt),
# SDXL-JPS (text_pos) and core SDXL (text_g/text_l). Negative fields
# (negative_prompt / text_neg) are intentionally absent so they are never used.
ENCODER_TEXT_KEYS = ("text", "prompt", "t5xxl", "positive_prompt", "text_pos",
                     "text_g", "clip_l", "text_l", "populated_text",
                     "wildcard_text", "prompt_text")
SAVE_HINTS  = ("SaveImage", "SaveVideo", "VideoCombine", "SaveAnimated", "Image Save", "SaveImageWebsocket")
SEED_KEYS   = ("seed", "noise_seed", "rand_seed")


def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int))


# ── workflow widgets_values map (flattened, incl. subgraphs) ──────────────────
def build_wf_widgets(workflow):
    """Map flattened API node id -> widgets_values, descending into subgraphs.
    A subgraph-instance node (type == subgraph def id) contributes the prefix
    "<instance_id>:" to its inner nodes, matching API ids like "30:6"."""
    out = {}
    if not isinstance(workflow, dict):
        return out
    defs = {sg.get("id"): sg for sg in (workflow.get("definitions") or {}).get("subgraphs", [])}

    def walk(nodes, prefix=""):
        for n in nodes or []:
            key = prefix + str(n.get("id"))
            out[key] = n.get("widgets_values")
            t = n.get("type")
            if t in defs:
                walk(defs[t].get("nodes", []), key + ":")

    walk(workflow.get("nodes", []))
    return out


def wf_text(wf_widgets, nid):
    """First non-empty string in a node's persisted widgets_values, or None."""
    wv = wf_widgets.get(str(nid))
    if isinstance(wv, list):
        for item in wv:
            if isinstance(item, str) and item.strip():
                return item
    return None


# ── value resolvers ───────────────────────────────────────────────────────────
def resolve_bool(prompt, val, depth=0):
    if depth > 30:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() not in ("", "0", "false", "none", "no")
    if is_link(val):
        n = prompt.get(str(val[0]))
        if n:
            ins = n.get("inputs", {})
            for k in ("value", "boolean", "BOOL"):
                if k in ins:
                    return resolve_bool(prompt, ins[k], depth + 1)
    return False


def resolve_string(prompt, val, depth=0, visited=None, wf=None):
    """Resolve a value (literal or [id, slot] link) to the prompt string it
    represents, following the graph. `wf` is the workflow widgets_values map,
    used as a fallback for captured text the API JSON sent empty."""
    if visited is None:
        visited = set()
    if wf is None:
        wf = {}
    if depth > 60:
        return None
    if isinstance(val, str):
        return val
    if not is_link(val):
        return None

    nid = str(val[0])
    if nid in visited:
        return None
    visited.add(nid)
    n = prompt.get(nid)
    if not n:
        return None
    ins = n.get("inputs", {})
    ct = n.get("class_type", "")

    # CoachBateBatchPrompter holds the ENTIRE list of prompts (one fired per
    # image). It is never the per-image prompt -> hard stop (skip + log).
    if "Batch" in ct and "Coach" in ct:
        return None

    # Text capture/display nodes: CoachBate Text Preview and Edit, or
    # ShowText|pysssss (the common LTX/other equivalent). These hold the
    # resolved single line; when the API sent it empty the captured value
    # lives in the workflow JSON.
    if "CoachBate" in ct or "ShowText" in ct:
        for k in ("any", "text", "string", "value", "prompt"):
            v = ins.get(k)
            if isinstance(v, str) and v.strip():
                return v
        wtxt = wf_text(wf, nid)          # API sent it empty -> use workflow value
        if wtxt:
            return wtxt
        for k in ("prompt_in", "any", "text", "string", "value"):
            if is_link(ins.get(k)):
                r = resolve_string(prompt, ins[k], depth + 1, visited, wf)
                if r and r.strip():
                    return r
        return None

    # PreviewAny / display passthrough -> follow `source`.
    if "source" in ins and len(ins) <= 2:
        r = resolve_string(prompt, ins.get("source"), depth + 1, visited, wf)
        if r:
            return r

    # Boolean switch (ComfySwitchNode uses `switch`; Crystools uses `boolean`).
    if "on_true" in ins and "on_false" in ins:
        sw = ins.get("switch", ins.get("boolean"))
        branch = "on_true" if resolve_bool(prompt, sw, depth) else "on_false"
        return resolve_string(prompt, ins.get(branch), depth + 1, visited, wf)

    # Letter selector (SimpleSelectorSwitch: select="D" -> source_d).
    if "select" in ins and isinstance(ins["select"], str) and ins["select"].strip():
        key = "source_" + ins["select"].strip().lower()
        if key in ins:
            return resolve_string(prompt, ins[key], depth + 1, visited, wf)

    # LenientSwitch (pass_if_a chooses source_a vs source_b).
    if "pass_if_a" in ins and ("source_a" in ins or "source_b" in ins):
        pick = "source_a" if resolve_bool(prompt, ins.get("pass_if_a"), depth) else "source_b"
        r = resolve_string(prompt, ins.get(pick), depth + 1, visited, wf)
        if r:
            return r

    # String concatenate (string_a/string_b etc.) -> keeps LoRA trigger words.
    pair = None
    for a, b in (("string_a", "string_b"), ("text_a", "text_b"), ("string1", "string2")):
        if a in ins or b in ins:
            pair = (a, b)
            break
    if pair:
        a = resolve_string(prompt, ins.get(pair[0], ""), depth + 1, set(visited), wf) or ""
        b = resolve_string(prompt, ins.get(pair[1], ""), depth + 1, set(visited), wf) or ""
        delim = ins.get("delimiter", "")
        if not isinstance(delim, str):
            delim = ""
        joined = (a + delim + b) if (a and b) else (a or b)
        if joined.strip():
            return joined

    # String replace.
    if "string" in ins and "find" in ins and "replace" in ins:
        base = resolve_string(prompt, ins.get("string"), depth + 1, visited, wf) or ""
        find = ins.get("find", "")
        repl = ins.get("replace", "")
        if isinstance(find, str):
            base = base.replace(find, repl if isinstance(repl, str) else "")
        return base

    # Direct literal string fields (links under the same names are handled
    # below; conditioning `positive` is always a link, so grabbing a literal
    # `positive` string here only matches text nodes like "easy positive").
    for k in ("value", "text", "string", "any", "multiline_text", "prompt",
              "positive", "positive_prompt", "text_pos", "t5xxl",
              "populated_text", "wildcard_text"):
        v = ins.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # Generic: follow the first text-ish input link (rgthree Any Switch, etc.).
    SKIP = {"clip", "image", "vae", "model", "conditioning", "samples", "latent",
            "select", "delimiter", "find", "replace", "switch", "pass_if_a",
            "seed", "noise_seed", "max_length", "width", "height"}
    any_keys = sorted([k for k in ins if re.match(r"any_?\d+$", k) or k == "any"],
                      key=lambda k: (len(k), k))
    for k in any_keys + [k for k in ins if k not in any_keys]:
        if k in SKIP or k.startswith(("label", "source_")):
            continue
        v = ins.get(k)
        if is_link(v):
            r = resolve_string(prompt, v, depth + 1, visited, wf)
            if r and r.strip():
                return r
    for k in sorted(k for k in ins if k.startswith("source_")):
        r = resolve_string(prompt, ins.get(k), depth + 1, visited, wf)
        if r and r.strip():
            return r
    # Last resort: a literal value persisted in the workflow JSON for this node.
    return wf_text(wf, nid)


# ── graph walks ────────────────────────────────────────────────────────────────
def bfs_back(prompt, start_ref, predicate, max_nodes=400):
    from collections import deque
    seen, q = set(), deque()
    if is_link(start_ref):
        q.append(str(start_ref[0]))
    while q and len(seen) < max_nodes:
        nid = q.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        n = prompt.get(nid)
        if not n:
            continue
        if predicate(nid, n):
            return nid
        for v in n.get("inputs", {}).values():
            if is_link(v):
                q.append(str(v[0]))
    return None


def is_sampler(nid, n):
    ins = n.get("inputs", {})
    return ("sampler_name" in ins) or ("guider" in ins) or \
           ("positive" in ins and "latent_image" in ins)


def sampler_positive_ref(prompt, sampler_nid):
    """The positive-conditioning link for a sampler. For guider-based samplers
    (SamplerCustomAdvanced) the positive lives on the CFGGuider/BasicGuider."""
    sins = prompt.get(sampler_nid, {}).get("inputs", {})
    if is_link(sins.get("positive")):
        return sins["positive"]
    if is_link(sins.get("guider")):
        gins = prompt.get(str(sins["guider"][0]), {}).get("inputs", {})
        if is_link(gins.get("positive")):
            return gins["positive"]
        if is_link(gins.get("conditioning")):
            return gins["conditioning"]
    if is_link(sins.get("conditioning")):
        return sins["conditioning"]
    return None


def is_encoder(nid, n):
    ct = n.get("class_type", "")
    ins = n.get("inputs", {})
    if "TextEncode" in ct or "CLIPText" in ct:
        return True
    return ("conditioning" not in ins) and any(k in ins for k in ("text", "prompt"))


def trace_to_encoder(prompt, ref, depth=0, visited=None):
    """Walk the conditioning chain from `ref` to the text encoder, following the
    ACTIVE branch of any switch (so dual-encoder graphs pick the right prompt).
    Falls back to a plain upstream search if the active path can't be followed."""
    if visited is None:
        visited = set()
    if not is_link(ref) or depth > 80:
        return None
    nid = str(ref[0])
    if nid in visited:
        return None
    visited.add(nid)
    n = prompt.get(nid)
    if not n:
        return None
    if is_encoder(nid, n):
        return nid
    ins = n.get("inputs", {})
    if "on_true" in ins and "on_false" in ins:
        sw = ins.get("switch", ins.get("boolean"))
        branch = "on_true" if resolve_bool(prompt, sw, depth) else "on_false"
        return trace_to_encoder(prompt, ins.get(branch), depth + 1, visited)
    if "select" in ins and isinstance(ins["select"], str) and ins["select"].strip():
        return trace_to_encoder(prompt, ins.get("source_" + ins["select"].strip().lower()),
                                depth + 1, visited)
    for k in ("conditioning", "positive", "cond", "base", "c", "guider"):
        if is_link(ins.get(k)):
            r = trace_to_encoder(prompt, ins[k], depth + 1, visited)
            if r:
                return r
    for v in ins.values():
        if is_link(v):
            r = trace_to_encoder(prompt, v, depth + 1, visited)
            if r:
                return r
    return None


def file_base(stem):
    m = re.match(r"^(.*?)[_\- ]?(\d{2,})[_]?$", stem)
    return m.group(1).rstrip("_- ") if m else stem


def pick_save_node(prompt, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    base = file_base(stem)
    saves = [(nid, n) for nid, n in prompt.items()
             if any(h in n.get("class_type", "") for h in SAVE_HINTS)]
    if not saves:
        return None

    def prefix_base(n):
        p = n.get("inputs", {}).get("filename_prefix", "")
        return os.path.basename(p.replace("\\", "/")) if isinstance(p, str) else ""

    for nid, n in saves:                       # exact base match
        if prefix_base(n) and prefix_base(n) == base:
            return nid
    cand = [(nid, n) for nid, n in saves       # stem startswith prefix -> longest
            if prefix_base(n) and stem.startswith(prefix_base(n))]
    if cand:
        cand.sort(key=lambda x: len(prefix_base(x[1])), reverse=True)
        return cand[0][0]
    if len(saves) == 1:
        return saves[0][0]
    return None


def find_sampler_for_save(prompt, save_nid):
    ins = prompt.get(save_nid, {}).get("inputs", {})
    img_ref = ins.get("images") or ins.get("image") or ins.get("video") or ins.get("frames")
    if is_link(img_ref):
        s = bfs_back(prompt, img_ref, is_sampler)
        if s:
            return s
    for nid, nn in prompt.items():
        if is_sampler(nid, nn):
            return nid
    return None


def collect_upstream(prompt, start_nid, max_nodes=400):
    from collections import deque
    seen, order, q = set(), [], deque([start_nid])
    while q and len(seen) < max_nodes:
        nid = q.popleft()
        if nid in seen or nid not in prompt:
            continue
        seen.add(nid)
        order.append(nid)
        for v in prompt[nid].get("inputs", {}).values():
            if is_link(v):
                q.append(str(v[0]))
    return order


def extract_settings(prompt, sampler_nid, width=None, height=None, frames=None, fps=None):
    parts = []
    if sampler_nid:
        sins  = prompt.get(sampler_nid, {}).get("inputs", {})
        chain = collect_upstream(prompt, sampler_nid)

        def first(keys):
            for nid in chain:
                ins = prompt.get(nid, {}).get("inputs", {})
                for k in keys:
                    if k in ins and not is_link(ins[k]):
                        return ins[k]
            return None

        steps = first(("steps",))
        if steps is not None:
            parts.append("Steps: %s" % steps)
        sn = first(("sampler_name",))
        if sn:
            parts.append("Sampler: %s" % sn)
        sched = first(("scheduler",))
        if sched:
            parts.append("Schedule type: %s" % sched)
        cfg = sins.get("cfg")
        if cfg is None or is_link(cfg):
            cfg = first(("cfg", "guidance"))
        if cfg is not None and not is_link(cfg):
            parts.append("CFG scale: %s" % cfg)
        seed = first(SEED_KEYS)
        if seed is not None:
            parts.append("Seed: %s" % seed)
    if width and height:
        parts.append("Size: %dx%d" % (width, height))
    if frames and frames > 1:
        parts.append("Frames: %s" % frames)
    if fps:
        parts.append("FPS: %s" % fps)
    return ", ".join(parts)


def extract(prompt, fname, wf_widgets, width=None, height=None, frames=None, fps=None):
    """Return (text, settings, debug_or_reason). text is None when nothing safe
    to embed could be found (reason explains why)."""
    save_nid    = pick_save_node(prompt, fname)
    sampler_nid = find_sampler_for_save(prompt, save_nid) if save_nid else None
    if not sampler_nid:
        for nid, n in prompt.items():
            if is_sampler(nid, n):
                sampler_nid = nid
                break
    if not sampler_nid:
        return None, None, "no sampler node in graph"

    pos_ref = sampler_positive_ref(prompt, sampler_nid)
    enc = None
    if is_link(pos_ref):
        enc = trace_to_encoder(prompt, pos_ref) or bfs_back(prompt, pos_ref, is_encoder)
    if not enc:
        return None, None, "no text encoder on sampler %s positive input" % sampler_nid

    # Take the first encoder text field that resolves to a non-empty string.
    eins = prompt[enc].get("inputs", {})
    text = None
    for k in ENCODER_TEXT_KEYS:
        if k in eins:
            r = resolve_string(prompt, eins[k], wf=wf_widgets)
            if r and r.strip():
                text = r
                break
    if not text or not text.strip():
        return None, None, ("prompt feeding encoder %s could not be resolved to "
                            "a literal (dynamic/batch source not captured)" % enc)

    settings = extract_settings(prompt, sampler_nid, width, height, frames, fps)
    dbg = "save=%s sampler=%s enc=%s(%s)" % (save_nid, sampler_nid, enc, prompt[enc].get("class_type"))
    return text.strip(), settings, dbg


def build_parameters(text, settings):
    result = text.strip()
    if settings:
        result += "\n" + settings
    return result


def process_png(filepath):
    name = os.path.basename(filepath)
    try:
        img = Image.open(filepath)
        info = img.info
    except Exception as e:
        print("  SKIP (cannot open: %s): %s" % (e, name))
        return

    if info.get("parameters") and not OVERWRITE:
        print("  SKIP (already has parameters): %s" % name)
        return

    raw_prompt   = info.get("prompt")
    raw_workflow = info.get("workflow")
    if not raw_prompt:
        print("  SKIP (no ComfyUI prompt metadata%s): %s"
              % (" - has workflow only" if raw_workflow else "", name))
        return

    try:
        prompt = json.loads(raw_prompt)
        wf_widgets = build_wf_widgets(json.loads(raw_workflow)) if raw_workflow else {}
    except Exception as e:
        print("  SKIP (JSON parse error: %s): %s" % (e, name))
        return

    text, settings, reason = extract(prompt, filepath, wf_widgets, img.width, img.height)
    if not text:
        print("  SKIP (%s): %s" % (reason, name))
        return

    parameters = build_parameters(text, settings)

    # Preserve every existing text chunk; overwrite only `parameters`.
    pnginfo = PngImagePlugin.PngInfo()
    for key, val in info.items():
        if isinstance(val, str) and key != "parameters":
            pnginfo.add_text(key, val)
    pnginfo.add_text("parameters", parameters)

    try:
        img.save(filepath, "PNG", pnginfo=pnginfo)
    except Exception as e:
        print("  SKIP (write failed: %s): %s" % (e, name))
        return

    preview = parameters[:130].replace("\n", " | ")
    print("  OK   %s" % name)
    print("       [%s]" % reason)
    print("       %s%s" % (preview, "..." if len(parameters) > 130 else ""))


# ── MP4 handling ──────────────────────────────────────────────────────────────
#
# ComfyUI stores `workflow` and `prompt` as MP4 *format tags* (same JSON as the
# PNG chunks), readable with ffprobe. We trace the prompt identically, then
# re-mux with ffmpeg writing a `parameters` (and `comment`) tag — the format
# CivitAI reads (it uses ffprobe's format.tags.parameters). We keep the existing
# tags (`-map_metadata 0`) so the embedded workflow/prompt survive, and add
# `-movflags use_metadata_tags` so the freeform `parameters` key is written.
# ai-toolkit's own sample videos already carry `parameters` and no workflow JSON,
# so they are detected and skipped (nothing to re-derive).

def ffprobe_info(filepath):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", filepath],
        capture_output=True, text=True)
    try:
        return json.loads(out.stdout)
    except Exception:
        return {}


def _rate_to_fps(r):
    try:
        if isinstance(r, str) and "/" in r:
            a, b = r.split("/")
            return round(float(a) / float(b), 3) if float(b) else None
        return round(float(r), 3)
    except Exception:
        return None


def process_mp4(filepath):
    name = os.path.basename(filepath)
    d = ffprobe_info(filepath)
    if not d:
        print("  SKIP (ffprobe failed): %s" % name)
        return
    tags = d.get("format", {}).get("tags", {}) or {}

    if tags.get("parameters") and not OVERWRITE:
        print("  SKIP (already has parameters): %s" % name)
        return

    raw_prompt   = tags.get("prompt")
    raw_workflow = tags.get("workflow")
    if not raw_prompt:
        why = "no ComfyUI prompt metadata"
        if tags.get("parameters"):
            why += " - already has parameters (ai-toolkit sample?)"
        print("  SKIP (%s): %s" % (why, name))
        return

    try:
        prompt = json.loads(raw_prompt)
        wf_widgets = build_wf_widgets(json.loads(raw_workflow)) if raw_workflow else {}
    except Exception as e:
        print("  SKIP (JSON parse error: %s): %s" % (e, name))
        return

    vstreams = [s for s in d.get("streams", []) if s.get("codec_type") == "video"]
    width = height = frames = fps = None
    if vstreams:
        v = vstreams[0]
        width  = int(v["width"])  if v.get("width")  else None
        height = int(v["height"]) if v.get("height") else None
        fps    = _rate_to_fps(v.get("r_frame_rate") or v.get("avg_frame_rate"))
        nb = v.get("nb_frames")
        if nb and str(nb).isdigit():
            frames = int(nb)
        elif v.get("duration") and fps:
            try:
                frames = int(round(float(v["duration"]) * fps))
            except Exception:
                frames = None

    text, settings, reason = extract(prompt, filepath, wf_widgets, width, height, frames, fps)
    if not text:
        print("  SKIP (%s): %s" % (reason, name))
        return

    parameters = build_parameters(text, settings)

    # Re-mux preserving existing tags, adding parameters + comment.
    tmp = filepath + ".tmp.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", filepath,
           "-map", "0",                # keep ALL streams (video + audio)
           "-map_metadata", "0",       # keep existing tags (workflow / prompt)
           "-metadata", "parameters=" + parameters,
           "-metadata", "comment=" + parameters,
           "-c", "copy", "-movflags", "use_metadata_tags", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(tmp):
            print("  SKIP (ffmpeg failed: %s): %s" % (r.stderr.strip()[:120], name))
            if os.path.exists(tmp):
                os.unlink(tmp)
            return
        try:
            os.replace(tmp, filepath)
        except OSError:
            # NTFS via WSL (DrvFs) forbids atomic rename-over-existing; copy bytes.
            with open(tmp, "rb") as src, open(filepath, "wb") as dst:
                shutil.copyfileobj(src, dst)
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception as e:
        print("  SKIP (write failed: %s): %s" % (e, name))
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return

    preview = parameters[:130].replace("\n", " | ")
    print("  OK   %s" % name)
    print("       [%s]" % reason)
    print("       %s%s" % (preview, "..." if len(parameters) > 130 else ""))


def main():
    exts = (".png",) + ((".mp4",) if INCLUDE_VIDEOS else ())
    try:
        if RECURSIVE:
            files = []
            for root, _dirs, names in os.walk(FOLDER):
                for n in names:
                    if n.lower().endswith(exts):
                        files.append(os.path.join(root, n))
            files.sort()
        else:
            files = sorted(os.path.join(FOLDER, f) for f in os.listdir(FOLDER)
                           if f.lower().endswith(exts))
    except Exception as e:
        print("ERROR: cannot list folder %s: %s" % (FOLDER, e))
        return
    print("\nProcessing %d file(s) in: %s%s%s\n"
          % (len(files), FOLDER,
             "  (incl. video)" if INCLUDE_VIDEOS else "",
             "  (recursive)" if RECURSIVE else ""))
    for fp in files:
        try:
            if fp.lower().endswith(".mp4"):
                process_mp4(fp)
            else:
                process_png(fp)
        except Exception as e:
            print("  SKIP (unexpected error: %s): %s" % (e, os.path.basename(fp)))
    print("\nDone.")


if __name__ == "__main__":
    main()
'@

# ── RUN ───────────────────────────────────────────────────────────────────────

Write-Host "Folder    : $folder"
Write-Host "WSL       : $wslDistro"
Write-Host "Overwrite : $overwriteMetadata"
Write-Host "Videos    : $includeVideos"
Write-Host "Recursive : $recursive"
Write-Host ""

# Pipe the Python script directly to Python's stdin ("-" = read from stdin).
# This avoids writing a temp file via the UNC share, which fails when WSL
# hasn't been started yet (\\wsl.localhost isn't mounted until first wsl call).
$pythonScript | wsl -d $wslDistro -- $pythonExe - $wslFolder $wslOverwrite $wslVideos $wslRecursive
