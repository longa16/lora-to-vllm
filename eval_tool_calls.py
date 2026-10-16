"""Évalue un modèle (de base ou avec adaptateur LoRA) sur test.jsonl.

Le modèle de base évalué avec le même format d'outils sert de baseline « prompt seul » :
lance le script deux fois, sans puis avec --adapter, et compare les rapports.

Sorties : <out-prefix>_report.json (métriques) et <out-prefix>_generations.jsonl
(une ligne par exemple, pour l'analyse d'erreurs).

Exemples :
    python eval_tool_calls.py --model Qwen/Qwen2.5-1.5B-Instruct --test data/processed/test.jsonl --out-prefix results/base_1.5b
    python eval_tool_calls.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/qwen15b/adapter \
        --test data/processed/test.jsonl --out-prefix results/lora_1.5b
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from toolcall_utils import score, to_native_messages

# transformers >= 5 renomme torch_dtype en dtype
DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

STRIP = ("<|im_end|>", "<|endoftext|>")


def clean(text: str) -> str:
    for s in STRIP:
        text = text.replace(s, "")
    return text.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="dossier de l'adaptateur LoRA (absent = modèle de base)")
    p.add_argument("--test", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--no-4bit", action="store_true", help="charge le modèle en fp16 au lieu de 4 bits")
    args = p.parse_args()

    records = [json.loads(line) for line in open(args.test, encoding="utf-8")]
    if args.limit:
        records = records[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {DTYPE_KW: torch.float16, "device_map": {"": 0}}
    if not args.no_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = []
    for rec in records:
        messages, tools = to_native_messages(rec)
        prompts.append(tok.apply_chat_template(messages[:-1], tools=tools, tokenize=False, add_generation_prompt=True))

    generations = []
    for i in range(0, len(prompts), args.batch):
        batch = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True,
                    add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new_tokens = out[:, batch["input_ids"].shape[1]:]
        generations += [clean(t) for t in tok.batch_decode(new_tokens, skip_special_tokens=False)]
        print(f"{min(i + args.batch, len(prompts))}/{len(prompts)}")

    report = {"model": args.model, "adapter": args.adapter, "metrics": score(records, generations)}
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(f"{prefix}_generations.jsonl", "w", encoding="utf-8") as f:
        for rec, gen in zip(records, generations):
            f.write(json.dumps({"id": rec["id"], "kind": rec["kind"], "query": rec["query"],
                                "gold": rec["answers"], "generation": gen}, ensure_ascii=False) + "\n")
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
