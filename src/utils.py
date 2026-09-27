import re
from collections import Counter, defaultdict
import statistics


def extract_conclusion(text):
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return text.strip()[-1000:]
    for n in [3, 2, 1]:
        if len(paragraphs) >= n:
            candidate = '\n\n'.join(paragraphs[-n:])
            if len(candidate) <= 1000:
                return candidate
    return paragraphs[-1][:1000]


def extract_conclusion_negotiation(text):
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) >= 2:
        return '\n\n'.join(paragraphs[-2:])
    elif paragraphs:
        return paragraphs[-1]
    return text.strip()


def extract_vote_local(text: str) -> str:
    """Rule-based final-vote extraction with signal priority.

    Priority order (first hit wins):
      0. "X pump/option is the only/sole/confirmed/reliable ..." -> X
      1. First color inside any **bold** segment (answers are usually bolded first)
      2. Conclusion segment: first action verb after stripping conditionals/negations,
         else first color
      3. Tail (last 600 chars) with conditionals & negations stripped:
         last action verb, then last color
      4. Raw tail fallback -> last color

    Returns "Red", "Yellow" or "Unknown".
    """
    if not text:
        return "Unknown"

    COLOR_RE = re.compile(r'\b(red|yellow)\b', re.IGNORECASE)

    # ---------- helpers ----------
    def _all_bold_colors(span: str):
        """All colors that appear inside **...** pairs, in order."""
        out = []
        for m in re.finditer(r'\*\*(.*?)\*\*', span, re.DOTALL):
            for c in re.findall(r'\b(red|yellow)\b', m.group(1), re.IGNORECASE):
                out.append(c.lower())
        return out

    def _action_color_first(span: str):
        """Return the FIRST color tied to a decision verb, or None."""
        patterns = [
            r'(?:we|i|our\s+team)\s+'
            r'(?:will\s+|should\s+|must\s+|have\s+to\s+|are\s+going\s+to\s+|decided\s+to\s+|choose\s+to\s+|need\s+to\s+|all\s+)?'
            r'(?:go|drive|head|move|choose|pick|take|select|vote\s+for|opt\s+for)\s+'
            r'(?:to\s+|on\s+|towards?\s+|with\s+)?\**\s*(red|yellow)\b',
            r'\b(?:go|drive|head|move|choose|pick|take|select)\s+'
            r'(?:to\s+|towards?\s+)?\**\s*(red|yellow)\b',
            r'(?:final|ultimate|overall|our)\s+(?:choice|decision|answer|vote)\s*'
            r'(?:is\s+)?[:\-—]?\s*\**\s*(red|yellow)\b',
            r'(?:correct|right|better|best)\s+(?:choice|answer|option|pump)\s+(?:is\s+)?\**\s*(red|yellow)\b',
        ]
        for pat in patterns:
            m = re.search(pat, span, re.IGNORECASE)
            if m:
                return m.group(1).lower()
        return None

    def _action_color_last(span: str):
        """Return the LAST color tied to a decision verb, or None."""
        matches = []
        for pat in [
            r'(?:we|i|our\s+team)\s+'
            r'(?:will\s+|should\s+|must\s+|have\s+to\s+|are\s+going\s+to\s+|decided\s+to\s+|choose\s+to\s+|need\s+to\s+|all\s+)?'
            r'(?:go|drive|head|move|choose|pick|take|select|vote\s+for|opt\s+for)\s+'
            r'(?:to\s+|on\s+|towards?\s+|with\s+)?\**\s*(red|yellow)\b',
            r'\b(?:go|drive|head|move|choose|pick|take|select)\s+'
            r'(?:to\s+|towards?\s+)?\**\s*(red|yellow)\b',
        ]:
            matches.extend(re.findall(pat, span, re.IGNORECASE))
        return matches[-1].lower() if matches else None

    def _strip_conditionals(span: str) -> str:
        return re.sub(
            r'(?:^|(?<=[.!?]))\s*(?:if|when|unless|suppose|supposing|in\s+case|provided\s+that)\b'
            r'[^.!?]{0,400}?[.!?]',
            ' ',
            span,
            flags=re.IGNORECASE
        )

    def _strip_negations(span: str) -> str:
        return re.sub(
            r'\b(?:not|no|never|avoid|avoids|avoiding|reject|rejects|rejecting|rule\s+out|ruled\s+out)'
            r'\s+\**\s*(?:go\s+|going\s+to\s+|choosing\s+|picking\s+|taking\s+)?\**\s*(red|yellow)\b',
            ' ',
            span,
            flags=re.IGNORECASE
        )

    # ---------- Step 0: uniqueness / reliability claim ----------
    # "Yellow pump is the only confirmed reliable source" -> Yellow
    m = re.search(
        r'\b(?:the\s+)?(red|yellow)\s+(?:pump|option|choice|one)\s+is\s+'
        r'(?:the\s+)?(?:only|sole|unique|confirmed|reliable|actual|real)\b',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).capitalize()

    # ---------- Step 1: first color inside any **bold** ----------
    bolds = _all_bold_colors(text)
    if bolds:
        return bolds[0].capitalize()

    # ---------- Step 2: conclusion segment, first action verb ----------
    concl_match = re.search(
        r'(?:^|[\n.!?])\s*(?:\**\s*)?(?:conclusion|final\s+(?:decision|choice|answer|vote)|'
        r'our\s+(?:final\s+)?decision|bottom\s+line)\s*(?:\**\s*)?[:\-—]\s*'
        r'(.{0,1500}?)(?=\n\s*\n|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    if concl_match:
        seg_clean = _strip_conditionals(_strip_negations(concl_match.group(1)))
        picked = _action_color_first(seg_clean)
        if picked:
            return picked.capitalize()
        colors = [c.lower() for c in COLOR_RE.findall(seg_clean)]
        if colors:
            return colors[0].capitalize()

    # ---------- Step 3: tail with conditionals & negations stripped ----------
    tail = text[-600:]
    tail_clean = _strip_conditionals(_strip_negations(tail))
    picked = _action_color_last(tail_clean)
    if picked:
        return picked.capitalize()
    colors = [c.lower() for c in COLOR_RE.findall(tail_clean)]
    if colors:
        return colors[-1].capitalize()

    # ---------- Step 4: raw tail fallback ----------
    raw_colors = [c.lower() for c in COLOR_RE.findall(text[-300:])]
    if raw_colors:
        return raw_colors[-1].capitalize()

    return "Unknown"