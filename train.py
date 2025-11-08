import os, csv, time, math, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch, torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data import get_dataloaders, set_seed
from src.model import TinyTransformerSeq2Seq

def print_env():
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        try: print("GPU:", torch.cuda.get_device_name(0))
        except: pass
    else:
        print("Using CPU only")

def perplexity(loss):
    try: return math.exp(loss)
    except OverflowError: return float("inf")

def train_one_epoch(model, loader, opt, device):
    model.train(); total=0.0; n=0
    crit = nn.CrossEntropyLoss()
    for src, dec_in, label in tqdm(loader, desc="train", leave=False):
        src, dec_in, label = src.to(device), dec_in.to(device), label.to(device)
        logits = model(src, dec_in)
        loss = crit(logits.view(-1, logits.size(-1)), label.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item(); n += 1
    return total / max(n,1)

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval(); total=0.0; n=0
    crit = nn.CrossEntropyLoss()
    for src, dec_in, label in tqdm(loader, desc="valid", leave=False):
        src, dec_in, label = src.to(device), dec_in.to(device), label.to(device)
        logits = model(src, dec_in)
        loss = crit(logits.view(-1, logits.size(-1)), label.view(-1))
        total += loss.item(); n += 1
    return total / max(n,1)

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_ckpt(path, model, opt, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt: opt.load_state_dict(ckpt["opt"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("best_val", float("inf"))
    return start_epoch, best_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", default="baseline", type=str)
    ap.add_argument("--epochs", default=2, type=int)
    ap.add_argument("--block_size", default=64, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--d_model", default=128, type=int)
    ap.add_argument("--n_head", default=4, type=int)
    ap.add_argument("--n_layer", default=2, type=int)
    ap.add_argument("--d_ff", default=512, type=int)
    ap.add_argument("--dropout", default=0.1, type=float)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--no_pe", action="store_true", help="消融：关闭位置编码")
    ap.add_argument("--resume", default="", type=str, help="恢复训练的 ckpt 路径")
    args = ap.parse_args()

    set_seed(args.seed)
    print_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dl, val_dl, meta = get_dataloaders(
        block_size=args.block_size, batch_size=args.batch_size, val_ratio=0.1, device_type=device.type
    )

    model = TinyTransformerSeq2Seq(
        vocab_size=meta["vocab_size"], d_model=args.d_model, n_head=args.n_head,
        n_layer=args.n_layer, d_ff=args.d_ff, block_size=args.block_size,
        dropout=args.dropout, use_pe=(not args.no_pe)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    exp_dir = os.path.join("results", args.exp_name); os.makedirs(exp_dir, exist_ok=True)
    metrics_csv = os.path.join(exp_dir, "metrics.csv")
    best_path   = os.path.join(exp_dir, "ckpt_best.pt")
    last_path   = os.path.join(exp_dir, "ckpt_last.pt")

    start_epoch = 1; best_val = float("inf")
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_val = load_ckpt(args.resume, model, opt, device)
        print(f"Resume from {args.resume}, start_epoch={start_epoch}, best_val={best_val:.4f}")

    tr_hist, va_hist = [], []
    t0 = time.time()
    for ep in range(start_epoch, args.epochs + 1):
        tr = train_one_epoch(model, train_dl, opt, device)
        va = eval_loss(model, val_dl, device)
        tr_hist.append(tr); va_hist.append(va)
        print(f"[Epoch {ep:02d}] train={tr:.4f} | valid={va:.4f} | train_ppl={math.exp(tr):.2f} | valid_ppl={math.exp(va):.2f}")

        save_ckpt({"model": model.state_dict(), "opt": opt.state_dict(),
                   "epoch": ep, "best_val": best_val, "args": vars(args)}, last_path)
        if va < best_val:
            best_val = va
            save_ckpt({"model": model.state_dict(), "opt": opt.state_dict(),
                       "epoch": ep, "best_val": best_val, "args": vars(args)}, best_path)

        scheduler.step()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(tr_hist, label="train"); plt.plot(va_hist, label="valid")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    fig_path = os.path.join(exp_dir, "loss_curve.png")
    plt.savefig(fig_path, dpi=160); print("曲线：", fig_path)

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_loss", "train_ppl", "valid_ppl"])
        for i, (tr, va) in enumerate(zip(tr_hist, va_hist), start=1):
            w.writerow([i, tr, va, math.exp(tr), math.exp(va)])
    print("指标：", metrics_csv)

    with torch.no_grad():
        src_example, _, _ = next(iter(val_dl))
        src_example = src_example[:1].to(device)
        gen_ids = model.generate(src_example, start_id=meta["bos_id"], max_new_tokens=400, temperature=0.9, top_k=50)
        itos = meta["itos"]; txt = "".join(itos[int(i)] for i in gen_ids[0].tolist()[1:])
        with open(os.path.join(exp_dir, "sample.txt"), "w", encoding="utf-8") as f:
            f.write(txt)
    print("生成样例：", os.path.join(exp_dir, "sample.txt"))

    print(f"⏱ Total time: {time.time()-t0:.1f}s")
    print("Best valid loss:", best_val)

if __name__ == "__main__":
    main()
