"""Fine-tuning QLoRA d'un modèle Qwen pour le function calling (GPU T4 16 Go : Kaggle / Colab).

Entrées : train.jsonl et val.jsonl produits par prepare_data.py.
Sorties : <out>/adapter (adaptateur LoRA final), <out>/checkpoints, <out>/train_info.json.

Choix de conception :
    - format natif du modèle : les outils et les appels passent par le chat template
      (balises <tool_call>), donc le même format est utilisé à l'inférence avec vLLM ;
    - la perte est calculée uniquement sur la réponse de l'assistant ;
    - fp16 (le T4 ne gère pas le bf16) ;
    - checkpoints réguliers, reprise possible (--resume), envoi optionnel vers le Hub
      Hugging Face en dépôt privé (--hub-repo, avec HF_TOKEN dans l'environnement).

Exemple (Kaggle, après `pip install -U transformers peft bitsandbytes accelerate`) :
    python train_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --train data/processed/train.jsonl --val data/processed/val.jsonl --out runs/qwen15b
"""

import argparse
import inspect
import json
import math
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

from toolcall_utils import to_native_messages

# transformers >= 5 renomme torch_dtype en dtype
DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode_example(rec: dict, tok, max_len: int):
    """Renvoie (exemple tokenisé, None) ou (None, raison du rejet)."""
    messages, tools = to_native_messages(rec)
    prompt = tok.apply_chat_template(messages[:-1], tools=tools, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tools=tools, tokenize=False)
    if not full.startswith(prompt):
        return None, "template"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        return None, "boundary"
    if len(full_ids) > max_len:
        return None, "too_long"
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]  # perte sur la réponse seulement
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}, None


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        longest = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), longest), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), longest), dtype=torch.long)
        lab = torch.full((len(batch), longest), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            att[i, :n] = torch.tensor(b["attention_mask"])
            lab[i, :n] = torch.tensor(b["labels"])
        return {"input_ids": ids, "attention_mask": att, "labels": lab}


def build_dataset(path: str, tok, max_len: int, limit: int | None):
    items = []
    rejected = {"template": 0, "boundary": 0, "too_long": 0, "bad_json": 0}
    n_lines = 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            if limit and n_lines >= limit:
                break
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected["bad_json"] += 1
                if rejected["bad_json"] <= 3:
                    print(f"{path} ligne {n} : JSON invalide ({exc.msg}, caractère {exc.pos}, "
                          f"longueur de la ligne {len(line)})\n   ...{line[max(0, exc.pos - 80):exc.pos + 40]!r}")
                continue
            enc, why = encode_example(rec, tok, max_len)
            if enc is None:
                rejected[why] += 1
            else:
                items.append(enc)
    print(f"{path}: {len(items)} exemples retenus sur {n_lines} lignes, rejetés : {rejected}")
    if n_lines and rejected["bad_json"] > 0.05 * n_lines:
        raise SystemExit(f"{path} : plus de 5 % de lignes invalides, le fichier est probablement abîmé "
                         "ou tronqué : régénère-le avec prepare_data.py et téléverse-le de nouveau.")
    return ListDataset(items), rejected


def make_training_args(kwargs: dict) -> TrainingArguments:
    """Crée TrainingArguments en ignorant les paramètres inconnus de la version installée."""
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" not in valid and "evaluation_strategy" in valid:  # anciennes versions
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = [k for k in kwargs if k not in valid]
    if dropped:
        print(f"ATTENTION : paramètres ignorés (inconnus de cette version de transformers) : {dropped}")
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in valid})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="ex. Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", type=Path, default=Path("runs/default"))
    p.add_argument("--max-len", type=int, default=1536, help="exemples plus longs écartés (baisse-le en cas de manque de mémoire)")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--max-val", type=int, default=150, help="nombre d'exemples de validation utilisés")
    p.add_argument("--max-train", type=int, default=None, help="limite pour un essai rapide")
    p.add_argument("--resume", default=None, help="chemin d'un checkpoint à reprendre")
    p.add_argument("--hub-repo", default=None, help="dépôt Hugging Face PRIVÉ pour sauvegarder les checkpoints")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds, rej_train = build_dataset(args.train, tok, args.max_len, args.max_train)
    val_ds, _ = build_dataset(args.val, tok, args.max_len, args.max_val)
    if len(train_ds) == 0:
        raise SystemExit("Aucun exemple d'entraînement : vérifie le chat template du modèle (rejets 'template').")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0}, **{DTYPE_KW: torch.float16},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(0.03 * total_steps))
    print(f"{total_steps} pas d'optimisation prévus, dont {warmup_steps} de chauffe")

    ta_kwargs = dict(
        output_dir=str(args.out / "checkpoints"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        push_to_hub=bool(args.hub_repo),
        hub_model_id=args.hub_repo,
        hub_strategy="checkpoint",
        hub_private_repo=True,
        seed=args.seed,
    )
    targs = make_training_args(ta_kwargs)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=PadCollator(tok.pad_token_id))
    trainer.train(resume_from_checkpoint=args.resume)

    adapter_dir = args.out / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    final_eval = trainer.evaluate()
    info = {"args": {k: str(v) for k, v in vars(args).items()}, "n_train": len(train_ds),
            "n_val": len(val_ds), "rejected_train": rej_train, "final_eval": final_eval}
    (args.out / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Adaptateur sauvegardé dans {adapter_dir}")


if __name__ == "__main__":
    main()
