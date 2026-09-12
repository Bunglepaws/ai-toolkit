"""The default dialogue bank, and the non-verbal tag vocabulary.

Lines are written to take roughly 5 seconds to speak, matching the 5.167 s grid slot so
the TTS `duration` conditioning barely has to stretch or compress the delivery. Clips are
taken IN ORDER, as many as the length setting needs -- 60 s takes all 12, 30 s takes the
first 6 -- so the first six are ordered to stand alone.

**No line here opens with a non-verbal tag, deliberately.** A tag at the START of a clip
puts a laugh/grunt/sigh at the start of nearly every training item, and the LoRA learns
that as part of the voice -- every generation then wants to open with the same noise.
Observed in practice and it is very audible.

Tags are also kept SPARSE: two out of twelve lines, mid-sentence or trailing. The long
breathy tags are the dangerous ones -- a `[sigh]` or `[surprise-oh]` renders as ~1.9 s of
broadband noise, which in a ~5 s slot is nearly 40% of the clip, and that much non-speech
in a training item bleeds into the cloned voice itself. Short tags (`[laughter]`,
`[confirmation-en]`, `[question-en]`) measured ~0.3-0.4 s and are harmless. `split_tags`
finds tags anywhere in the line, so mid-sentence and end-of-line placement both work.

Keep edited lines in the 13-16 word range. Much shorter and `duration` stretches them into
absurdly slow speech; much longer and they compress into a rush. This is a quality
heuristic, not a correctness constraint -- exact-duration control means a badly sized line
produces an odd-sounding clip rather than a rejected one.
"""

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Non-verbal tags
# ---------------------------------------------------------------------------
# OmniVoice takes these inline in the input text. This is the COMPLETE supported set --
# anything else renders as literal spoken words, so `[sniff]` would produce a clip of
# someone saying "sniff". Validated against user dialogue before a job spends any GPU time.
#
# The caption phrase is what replaces the tag when writing the training caption: the
# bracket syntax means something to the TTS and nothing to a diffusion model's text
# encoder, which would read it as literal tokens.
NON_VERBAL_TAGS = {
    "[laughter]": "laughing",
    "[sigh]": "sighing",
    "[confirmation-en]": "murmuring agreement",
    "[question-en]": "a questioning hum",
    "[question-ah]": "a questioning 'ah?'",
    "[question-oh]": "a questioning 'oh?'",
    "[question-ei]": "a questioning 'ei?'",
    "[question-yi]": "a questioning 'yi?'",
    "[surprise-ah]": "a surprised 'ah'",
    "[surprise-oh]": "a surprised 'oh'",
    "[surprise-wa]": "a surprised 'wa'",
    "[surprise-yo]": "a surprised 'yo'",
    "[dissatisfaction-hnn]": "a disgruntled 'hnn'",
}

_TAG_RE = re.compile(r"\[[^\]\[]{1,40}\]")


class UnknownTagError(ValueError):
    """A bracket token that is not in the model's vocabulary. User-facing message."""


def validate_tags(lines: List[str]) -> None:
    """Raise UnknownTagError naming the supported set if any line carries a bad tag.

    Called before generation starts so a typo costs a message, not a dataset of someone
    reading brackets aloud.
    """
    bad = []
    for line in lines:
        for tag in _TAG_RE.findall(line):
            if tag not in NON_VERBAL_TAGS:
                bad.append(tag)
    if bad:
        uniq = sorted(set(bad))
        raise UnknownTagError(
            f"Unsupported non-verbal tag(s) in the dialogue: {', '.join(uniq)}. "
            f"The model only knows: {', '.join(sorted(NON_VERBAL_TAGS))}. "
            f"Anything else is spoken aloud as literal words."
        )


def split_tags(line: str) -> Tuple[List[str], str]:
    """'[laughter] You got me.' -> (['laughing'], 'You got me.')

    Returns the caption phrases for any tags present and the line with tags stripped, so
    the caption can read as natural language while the TTS still gets its brackets.
    """
    phrases = [NON_VERBAL_TAGS[t] for t in _TAG_RE.findall(line) if t in NON_VERBAL_TAGS]
    spoken = _TAG_RE.sub("", line).strip()
    spoken = re.sub(r"\s{2,}", " ", spoken)
    return phrases, spoken


# ---------------------------------------------------------------------------
# Default bank -- 12 lines, ~5 s each
# ---------------------------------------------------------------------------
DEFAULT_DIALOGUE: List[str] = [
    "Alright boys, hit the showers. You've earned every bit of that one out there today.",
    "Nice work, big guy. [laughter] I honestly didn't think you had that last set in you.",
    "Come here a second, let me get a look at that shoulder before you take off.",
    "Yeah, that's it. Slow, controlled, all the way down. Just like that, perfect.",
    "You've been holding out on me. Where's all of this been hiding lately, huh?",
    "You're going to be the death of me one day, you know that, right? [sigh]",
    "Towel off and come meet me in my office, we've got some things to talk about.",
    "Well now, somebody's been putting in the extra work over the summer, haven't they?",
    "Chin up, chest out, own the room. That's how a champion walks in here.",
    "Lock the door behind you, I don't want anybody walking in on this tonight.",
    "You sure you can handle another round, or do you need a minute first?",
    "Good boy. That's exactly what I've been wanting to see out of you.",
]


def resolve_dialogue(dialogue: List[str], count: int) -> List[str]:
    """The `count` lines to generate, cycling the bank if it is shorter than needed.

    Cycling rather than erroring is deliberate: TTS is stochastic, so the same line twice
    is two usable clips, not a duplicate. The caller reports the ratio so a thin list is
    visible rather than silently accepted.
    """
    lines = [l.strip() for l in (dialogue or DEFAULT_DIALOGUE) if l and l.strip()]
    if not lines:
        raise ValueError("dialogue is empty -- nothing to say")
    return [lines[i % len(lines)] for i in range(count)]
