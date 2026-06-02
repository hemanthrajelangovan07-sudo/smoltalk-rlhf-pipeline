import subprocess, sys

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-300:])

run("pip install -qU unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git")
run("pip install -q --no-deps trl>=1.0.0 peft>=0.14.0 bitsandbytes>=0.45.0 "
    "accelerate>=1.0.0 datasets>=3.0.0 xformers einops wandb evaluate sentencepiece huggingface_hub")
print("[OK] Install done — restart runtime, then run cells below")



import unsloth
import os, time, gc, torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import CPOTrainer, CPOConfig
from huggingface_hub import login, list_repo_files

IS_COLAB = os.path.exists("/content")
IS_KAGGLE = os.path.exists("/kaggle/working")

BASE = "/content/PostTraining_2026" if IS_COLAB else \
       "/kaggle/working/PostTraining_2026" if IS_KAGGLE else \
       "./PostTraining_2026"

CONFIG = {
    "sft_hub":       "hemanthrajelangovan/Qwen2.5-1.5B-SFT-2026",
    "simpo_hub":     "hemanthrajelangovan/Qwen2.5-1.5B-SimPO-2026",
    "max_seq_len":   1024,
    "base_dir":      BASE,
    "wandb_project": "rlhf-simpo-grpo-2026",
}
BASE_DIR = CONFIG["base_dir"]
for d in [f"{BASE_DIR}/checkpoints/simpo"]:
    os.makedirs(d, exist_ok=True)

def vram(tag=""):
    a = torch.cuda.memory_allocated(0)/1e9
    r = torch.cuda.memory_reserved(0)/1e9
    print(f"VRAM [{tag}]: {a:.2f} GB | free={r-a:.2f} GB")


hf_token = os.environ.get("HF_TOKEN", "")
print(f"HF_TOKEN present: {bool(hf_token)}")
if hf_token:
    login(token=hf_token, add_to_git_credential=True)

try:
    files = list_repo_files(CONFIG["sft_hub"], token=hf_token or None)
    print(f"Repo files ({len(files)}):", [f.split("/")[-1] for f in files])
except Exception as e:
    print(f"[ERROR] Cannot access repo: {e}")
    print("Either set HF_TOKEN Colab secret or make repo public")

vram("before load")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["sft_hub"],
    max_seq_length=CONFIG["max_seq_len"],
    dtype=None,
    load_in_4bit=True,
)
model.train()
model = model.to("cuda")

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable > 0, "No trainable parameters — model not in training mode!"
print(f"Trainable params: {trainable/1e6:.2f}M (total: {sum(p.numel() for p in model.parameters())/1e6:.0f}M)")
vram("after load")



print("Loading UltraFeedback...")
ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

print(f"Dataset columns: {ds.column_names}")
required = {"chosen", "rejected", "score_chosen", "score_rejected"}
assert required.issubset(set(ds.column_names)), f"Missing columns: {required - set(ds.column_names)}"

before = len(ds)
ds = ds.filter(lambda x: x["score_chosen"] - x["score_rejected"] >= 2)
print(f"Filtered {before} -> {len(ds)} (removed {before-len(ds)} low-margin pairs)")

assert len(ds) > 0, "No samples left after margin filter — threshold too strict"

ds = ds.shuffle(seed=42)
ds = ds.map(lambda x: {"margin": x["score_chosen"] - x["score_rejected"]})
ds = ds.sort("margin", reverse=True)
ds = ds.select(range(min(5000, len(ds))))
margin_max = ds[0]["margin"]
margin_min = ds[-1]["margin"]
avg_margin = sum(ds[i]["margin"] for i in range(len(ds))) / len(ds)
print(f"Top {len(ds)} — margin range: {margin_min:.1f}–{margin_max:.1f}, avg: {avg_margin:.2f}")

ds = ds.remove_columns(["margin"])

split = ds.train_test_split(test_size=0.1, seed=42)
train_data, eval_data = split["train"], split["test"]
print(f"Train: {len(train_data):,} | Eval: {len(eval_data):,}")



os.environ["WANDB_PROJECT"] = CONFIG["wandb_project"]
api_key = os.environ.get("WANDB_API_KEY", "")
if not api_key:
    os.environ["WANDB_MODE"] = "offline"
    print("[INFO] W&B offline (set WANDB_API_KEY for online)")

cfg = CPOConfig(
    
    loss_type="simpo",
    beta=2.5,
    simpo_gamma=0.8,

    
    output_dir=f"{BASE_DIR}/checkpoints/simpo",

    
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,

    
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    gradient_checkpointing=True,

    
    max_length=CONFIG["max_seq_len"],
    max_prompt_length=256,

    
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    report_to="wandb",
    eval_strategy="no",
    seed=42,
)

trainer = CPOTrainer(
    model=model, args=cfg,
    train_dataset=train_data, eval_dataset=eval_data,
    tokenizer=tokenizer,
)


print("\nStarting SimPO training...")
print("Signals to watch (wandb logs every 10 steps):")
print("  rewards/chosen   → should RISE (model prefers chosen more)")
print("  rewards/rejected → should FALL (model rejects rejected more)")
print("  rewards/margins  → should RISE (gap widening, target: 1.0-2.5)")
print("  loss             → should decrease and plateau around 0.3-0.4")
print("  (Eval disabled — training loss is the quality signal for 1 epoch)")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.cuda.empty_cache()
gc.collect()
vram("before train()")

t0 = time.time()
result = trainer.train(resume_from_checkpoint=False)
elapsed = (time.time() - t0) / 60
print(f"Done in {elapsed:.1f} min | loss: {result.training_loss:.4f}")

trainer.save_model(f"{BASE_DIR}/checkpoints/simpo/final_adapter")
tokenizer.save_pretrained(f"{BASE_DIR}/checkpoints/simpo/final_adapter")
print(f"Saved to {BASE_DIR}/checkpoints/simpo/final_adapter")



hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    hf_token = input("Paste your HuggingFace write token: ").strip()
print(f"Token length: {len(hf_token)} chars")
if hf_token:
    login(token=hf_token)
else:
    print("[ERROR] No token provided — cannot push to Hub.")

try:
    _ = model  
except NameError:
    print("[INFO] Loading adapter from disk for push...")
    import unsloth, torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from peft import PeftModel
    try:
        _ = CONFIG
    except NameError:
        CONFIG = {
            "simpo_hub": "hemanthrajelangovan/Qwen2.5-1.5B-SimPO-2026",
            "base_dir": "/content/PostTraining_2026",
            "max_seq_len": 1024,
        }
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=CONFIG["max_seq_len"],
        dtype=None, load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, f"{CONFIG['base_dir']}/checkpoints/simpo/final_adapter")
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

print(f"Pushing to {CONFIG['simpo_hub']}...")
print(f"Token length: {len(hf_token) if hf_token else 0} chars")
try:
    model.push_to_hub(CONFIG["simpo_hub"], commit_message="SimPO adapter", token=hf_token or None)
    tokenizer.push_to_hub(CONFIG["simpo_hub"], token=hf_token or None)
    print(f"[OK] Pushed to https://huggingface.co/{CONFIG['simpo_hub']}")
except Exception as e:
    print(f"[WARN] Push failed: {e}")
    print("")
    print("Possible fixes:")
    print("1. Go to https://huggingface.co/settings/tokens → create token with WRITE role")
    print("2. In Colab: 🔑 Secrets → add HF_TOKEN with the EXACT token (no spaces, no quotes)")
    print("3. If token was set before creating it, refresh: Runtime → Restart runtime, then re-run")

try:
    del trainer
except NameError:
    pass
del model
gc.collect()
torch.cuda.empty_cache()
vram("after cleanup")
print("\n[DONE] SimPO complete. Proceed to GRPO.")
