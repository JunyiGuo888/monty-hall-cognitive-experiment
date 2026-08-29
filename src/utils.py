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
    if not text:
        return "Unknown"
    tail = text[-300:].lower()
    patterns = [
        r'(?:go|drive|head|move)\s+(?:to|towards?)\s+(red|yellow)',
        r'(?:choose|pick|select|decide\s+on)\s+(red|yellow)',
        r'(?:should|will|must|have\s+to)\s+(?:go|drive|head|choose|pick)\s+(?:to\s+)?(red|yellow)',
        r'(?:take|use|need)\s+(red|yellow)',
        r'final\s+(?:decision|choice).*?\b(red|yellow)\b',
        r'we\s+(?:are|go|take)\s+(?:to\s+)?(red|yellow)',
    ]
    for pat in patterns:
        m = re.search(pat, tail)
        if m:
            return m.group(1).capitalize()
    bold = re.findall(r'\*\*(red|yellow)\*\*', tail)
    if bold:
        return bold[-1].capitalize()
    all_colors = re.findall(r'\b(red|yellow)\b', tail)
    if all_colors:
        return all_colors[-1].capitalize()
    return "Unknown"

class MontyHallDetector:
    @staticmethod
    def detect_monty_hall_usage(text):
        if not text:
            return {"uses_monty_hall": False, "is_critical": False, "evidence": [], "confidence": 0}
        evidence = []
        text_lower = text.lower()
        has_three = any(kw in text_lower for kw in ["three", "3", "3 options", "three pumps", "three fuel pumps"])
        has_elimination = any(kw in text_lower for kw in ["eliminate", "open", "empty", "confirmed empty", "boss says", "black is empty"])
        has_initial_prob = any(kw in text_lower for kw in ["1/3", "one-third", "33%", "33.3%"])
        has_switch_prob = any(kw in text_lower for kw in ["2/3", "two-thirds", "66%", "66.7%", "probability increase", "higher chance"])
        has_probability = any(kw in text_lower for kw in ["probability", "chance", "likelihood", "odds"])
        if has_three:
            evidence.append("three_options")
        if has_initial_prob:
            evidence.append("initial_1/3")
        if has_elimination:
            evidence.append("elimination")
        if has_switch_prob:
            evidence.append("becomes_2/3")
        if has_probability:
            evidence.append("probability_reasoning")
        wants_switch = any(kw in text_lower for kw in ["switch to", "change to", "should switch", "choose the other", "go to red instead", "go to yellow instead", "better to switch"])
        if wants_switch:
            evidence.append("wants_switch")
        uses_monty_hall = has_elimination and (has_switch_prob or (has_probability and has_initial_prob)) and wants_switch
        is_critical = any(kw in text_lower for kw in ["not monty hall", "does not apply", "different situation", "causal structure changed", "not the same"])
        if is_critical:
            evidence.append("critical_monty_hall")
        has_game_theory = any(kw in text_lower for kw in ["game theory", "strategic", "opponent", "other vehicle", "both", "coordinate"])
        if has_game_theory:
            evidence.append("game_theory")
        return {
            "uses_monty_hall": uses_monty_hall,
            "is_critical": is_critical,
            "evidence": evidence,
            "confidence": len(evidence) / 8
        }

    @staticmethod
    def get_monty_hall_type(text):
        result = MontyHallDetector.detect_monty_hall_usage(text)
        if result["is_critical"] and "game_theory" in result["evidence"]:
            return "game_theoretic"
        elif result["is_critical"]:
            return "critical"
        elif not result["uses_monty_hall"]:
            return "other"
        return "monty_hall_switch" if "wants_switch" in result["evidence"] else "monty_hall_other"