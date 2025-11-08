# src/data.py
import os, requests, torch
from torch.utils.data import Dataset, DataLoader

SPECIAL_TOKENS = ["<BOS>"]

def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def download_tiny_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache_dir = os.path.expanduser("~/.cache/tinyshakespeare")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "input.txt")
    if not os.path.exists(path):
        print("▶ 正在下载 Tiny Shakespeare …")
        r = requests.get(url, timeout=60); r.raise_for_status()
        with open(path, "wb") as f: f.write(r.content)
        print("✅ 下载完成 →", path)
    else:
        print("ℹ 已存在缓存文件 →", path)
    return path

def build_vocab(text: str):
    chars = sorted(list(set(text)))
    vocab = SPECIAL_TOKENS + chars
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

class CharSeqPairDataset(Dataset):
    def __init__(self, ids: torch.Tensor, block_size: int, bos_id: int):
        self.ids = ids; self.block = block_size; self.bos_id = bos_id

    def __len__(self): return len(self.ids) - self.block - 1

    def __getitem__(self, i):
        x = self.ids[i : i + self.block]
        y = self.ids[i + 1 : i + 1 + self.block]
        dec_in = torch.empty_like(x); dec_in[0] = self.bos_id; dec_in[1:] = y[:-1]
        return x, dec_in, y

def get_dataloaders(block_size=64, batch_size=64, val_ratio=0.1, device_type="cpu"):
    path = download_tiny_shakespeare()
    with open(path, "r", encoding="utf-8") as f: text = f.read()
    stoi, itos = build_vocab(text)
    ids = encode(text, stoi)

    n = int(len(ids) * (1 - val_ratio))
    train_ids, val_ids = ids[:n], ids[n:]
    bos_id = stoi["<BOS>"]

    train_ds = CharSeqPairDataset(train_ids, block_size, bos_id)
    val_ds   = CharSeqPairDataset(val_ids,   block_size, bos_id)

    use_cuda = (device_type == "cuda")
    num_workers = min(os.cpu_count() or 0, 4) if use_cuda else 0
    pin_mem = use_cuda

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True, num_workers=num_workers, pin_memory=pin_mem)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          drop_last=True, num_workers=num_workers, pin_memory=pin_mem)

    meta = {"vocab_size": len(stoi), "stoi": stoi, "itos": itos, "bos_id": bos_id}
    return train_dl, val_dl, meta
