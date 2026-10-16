import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a function-calling assistant. You are given a list of tools (JSON) "
    "and a user request. Respond ONLY with a JSON list of tool calls, each of the form "
    '{"name": <tool name>, "arguments": {<argument>: <value>}}. '
    "If no tool is suitable for the request, respond with []."
)


def load_rows(input_path: str | None):
    if input_path:
        with open(input_path, encoding="utf-8") as f:
            return json.load(f)
    from datasets import load_dataset

    try:
        return load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    except Exception as exc:
        raise SystemExit(
            "Impossible de charger le jeu de données.\n"
            f"Détail : {exc}"
        )


def parse_row(row: dict) -> dict | None:
    """Renvoie l'exemple normalisé, ou None s'il est inutilisable."""
    try:
        tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
    if not isinstance(tools, list) or not isinstance(answers, list) or not answers:
        return None
    if not all(isinstance(t, dict) and "name" in t for t in tools):
        return None
    if not all(isinstance(a, dict) and "name" in a and isinstance(a.get("arguments"), dict) for a in answers):
        return None
    tool_names = {t["name"] for t in tools}
    answer_names = {a["name"] for a in answers}
    if not answer_names <= tool_names:
        return None
    query = (row.get("query") or "").strip()
    if not query:
        return None
    return {
        "query": query,
        "tools": tools,
        "answers": answers,
        "tool_names": frozenset(tool_names),
        "answer_names": frozenset(answer_names),
    }


def to_negative(ex: dict, tool_pool: list[dict], rng: random.Random) -> dict | None:
    """Retire les outils nécessaires : la bonne réponse devient « aucun appel ».

    Si l'exemple n'avait qu'un seul outil, on lui en donne 1 à 3 autres, tirés au
    hasard parmi les outils du même split.
    """
    remaining = [t for t in ex["tools"] if t["name"] not in ex["answer_names"]]
    if not remaining:
        sample = rng.sample(tool_pool, min(10, len(tool_pool)))
        remaining = [t for t in sample if t["name"] not in ex["answer_names"]][:rng.randint(1, 3)]
    if not remaining:
        return None
    return {**ex, "tools": remaining, "answers": [],
            "tool_names": frozenset(t["name"] for t in remaining), "answer_names": frozenset()}


def make_record(ex: dict, kind: str) -> dict:
    user = "Tools:\n" + json.dumps(ex["tools"], ensure_ascii=False) + "\n\nRequest: " + ex["query"]
    return {
        "id": ex["id"],
        "kind": kind,
        "query": ex["query"],
        "tools": ex["tools"],
        "answers": ex["answers"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(ex["answers"], ensure_ascii=False)},
        ],
    }


def build_split(pool: list[dict], neg_ratio: float, rng: random.Random) -> list[dict]:
    n_neg = int(len(pool) * neg_ratio)
    tool_pool = list({t["name"]: t for ex in pool for t in ex["tools"]}.values())
    records = []
    for k, ex in enumerate(pool):
        neg = to_negative(ex, tool_pool, rng) if k < n_neg else None
        records.append(make_record(neg, "no_call") if neg else make_record(ex, "call"))
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="fichier JSON local")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--neg-ratio", type=float, default=0.10, help="part d'exemples « aucun outil »")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    #chargement, validation, dédoublonnage
    rows = load_rows(args.input)
    examples, seen, invalid, duplicates = [], set(), 0, 0
    for i, row in enumerate(rows):
        ex = parse_row(row)
        if ex is None:
            invalid += 1
            continue
        key = " ".join(ex["query"].lower().split())
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        ex["id"] = i
        examples.append(ex)
    print(f"{len(examples)} exemples valides ({invalid} invalides, {duplicates} doublons)")

    # test par fonctions
    names = sorted({n for ex in examples for n in ex["answer_names"]})
    rng.shuffle(names)
    held: set[str] = set()
    eligible: list[dict] = []
    for start in range(0, len(names), 25):
        held.update(names[start:start + 25])
        eligible = [ex for ex in examples if ex["answer_names"] <= held]
        if len(eligible) >= args.n_test:
            break
    rng.shuffle(eligible)
    test_pool = eligible[:args.n_test]

    # entraînement
    train_pool = [ex for ex in examples if not (ex["tool_names"] & held)]
    rng.shuffle(train_pool)
    val_pool = train_pool[:args.n_val]
    train_pool = train_pool[args.n_val:][:args.n_train]
    dropped = len(examples) - len(train_pool) - len(val_pool) - len(test_pool)

    if len(train_pool) < args.n_train or len(test_pool) < args.n_test:
        print(f"ATTENTION : train={len(train_pool)}/{args.n_train}, test={len(test_pool)}/{args.n_test}")

    # négatifs et écriture
    train = build_split(train_pool, args.neg_ratio, rng)
    val = build_split(val_pool, args.neg_ratio, rng)
    test = build_split(test_pool, args.neg_ratio, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        write_jsonl(args.out / f"{name}.jsonl", split)

    #aucune fonction du test dans les outils de train / val
    leak = sum(1 for r in train + val for t in r["tools"] if t["name"] in held)
    stats = {
        "seed": args.seed,
        "valid_examples": len(examples),
        "invalid": invalid,
        "duplicates": duplicates,
        "held_out_functions": len(held),
        "dropped_to_avoid_leakage": dropped,
        "splits": {
            name: {"total": len(s), "no_call": sum(r["kind"] == "no_call" for r in s)}
            for name, s in (("train", train), ("val", val), ("test", test))
        },
        "leakage_check_train_val_tools_in_test_functions": leak,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if leak:
        raise SystemExit("Fuite détectée : vérifie le script.")


if __name__ == "__main__":
    main()