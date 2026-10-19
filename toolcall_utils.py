"""Utilitaires partagés par train_qlora.py et eval_tool_calls.py.
"""

import json
import re
from collections import Counter

NO_CALL_TEXT = "None of the available tools can answer this request."

_TYPE_MAP = {
    "str": "string", "string": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
}


def _json_type(t) -> str:
    t = str(t).strip().lower()
    if t in _TYPE_MAP:
        return _TYPE_MAP[t]
    if t.startswith(("list", "tuple", "set", "array")):
        return "array"
    if t.startswith(("dict", "object")):
        return "object"
    return "string"


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Outils xlam {name, description, parameters:{p:{type,description,default}}} -> schéma OpenAI."""
    out = []
    for t in tools:
        props, required = {}, []
        for pname, spec in (t.get("parameters") or {}).items():
            spec = spec if isinstance(spec, dict) else {}
            props[pname] = {
                "type": _json_type(spec.get("type", "string")),
                "description": str(spec.get("description", "")),
            }
            if spec.get("default", "") in ("", None):
                required.append(pname)
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": str(t.get("description", "")),
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return out


def to_native_messages(rec: dict) -> tuple[list[dict], list[dict]]:
    """Renvoie (messages, tools). Le dernier message est la réponse attendue."""
    tools = to_openai_tools(rec["tools"])
    messages = [{"role": "user", "content": rec["query"]}]
    if rec["answers"]:
        calls = [
            {"type": "function", "function": {"name": a["name"], "arguments": a["arguments"]}}
            for a in rec["answers"]
        ]
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    else:
        messages.append({"role": "assistant", "content": NO_CALL_TEXT})
    return messages, tools


_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> tuple[list[dict], bool]:
    """Extrait les appels du texte généré. Renvoie (appels, valide).

    valide=False si une balise <tool_call> contient un JSON incorrect ou n'est pas fermée.
    """
    blocks = _CALL_RE.findall(text)
    valid = text.count("<tool_call>") == len(blocks)
    calls = []
    for block in blocks:
        try:
            obj = json.loads(block)
            if not isinstance(obj, dict) or "name" not in obj:
                raise ValueError("appel sans nom")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("arguments invalides")
            calls.append({"name": obj["name"], "arguments": args})
        except (json.JSONDecodeError, ValueError, TypeError):
            valid = False
    return calls, valid


def _lenient(v):
    """Normalisation tolérante : 5 == "5", casse et espaces ignorés."""
    if isinstance(v, dict):
        return {k: _lenient(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_lenient(x) for x in v]
    return str(v).strip().lower()


def _key(call: dict, lenient: bool = False) -> str:
    args = _lenient(call["arguments"]) if lenient else call["arguments"]
    return call["name"] + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)


def score(records: list[dict], generations: list[str]) -> dict:
    """Métriques par type d'exemple."""
    stats = {"call": Counter(), "no_call": Counter()}
    for rec, text in zip(records, generations):
        calls, valid = parse_tool_calls(text)
        s = stats[rec["kind"]]
        s["n"] += 1
        s["valid_format"] += valid
        if rec["kind"] == "no_call":
            s["correct_no_call"] += bool(valid and not calls)
            continue
        gold = rec["answers"]
        s["missed_call"] += bool(valid and not calls)
        s["tool_names_ok"] += bool(valid and Counter(c["name"] for c in calls) == Counter(a["name"] for a in gold))
        s["exact_match"] += bool(valid and Counter(_key(c) for c in calls) == Counter(_key(a) for a in gold))
        s["exact_match_lenient"] += bool(
            valid and Counter(_key(c, True) for c in calls) == Counter(_key(a, True) for a in gold)
        )
    report = {}
    for kind, s in stats.items():
        n = s["n"]
        if n:
            report[kind] = {"n": n, **{k: round(v / n, 4) for k, v in s.items() if k != "n"}}
    return report
