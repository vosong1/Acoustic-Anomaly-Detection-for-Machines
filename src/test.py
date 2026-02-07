import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.load_data import collect_all
    from src.audio_utils import load_audio
except:
    from load_data import collect_all
    from audio_utils import load_audio


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_logmel(y, sr, n_fft, hop, n_mels):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)
    logmel = np.log(mel + 1e-8)
    return logmel.T.astype(np.float32)  # (T, n_mels)


def frames_to_stacked_vectors(logmel_TxM, stack=20 ):
    T, M = logmel_TxM.shape
    if T < stack:
        pad = np.zeros((stack - T, M), dtype=np.float32)
        logmel_TxM = np.vstack([logmel_TxM, pad])
        T = stack
    # sliding window stack frames
    X = []
    for i in range(T - stack + 1):
        X.append(logmel_TxM[i:i+stack].reshape(-1))  # stack*M
    return np.vstack(X).astype(np.float32)  # (N, stack*M)


class DenseAE(nn.Module):
    def __init__(self, in_dim=320, bottleneck=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, bottleneck),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, in_dim),
        )

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out


@torch.no_grad()
def score_clip(model, clip_vectors, device):
    # recon error per vector, then average over clip (paper averages over the segment) :contentReference[oaicite:2]{index=2}
    x = torch.from_numpy(clip_vectors).to(device)
    y = model(x)
    err = torch.mean((y - x) ** 2, dim=1)  # MSE per vector
    return float(err.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--machine", type=str, default=None)

    # logmel params (paper uses 64 mel; AE stacks 5 frames => 320D) :contentReference[oaicite:3]{index=3}
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--stack", type=int, default=5)

    # AE training
    ap.add_argument("--bottleneck", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = read_json(args.config)
        args.machine = cfg.get("machine", args.machine)
        lm = cfg.get("logmel", {})
        args.sr = lm.get("sr", args.sr)
        args.n_fft = lm.get("n_fft", args.n_fft)
        args.hop = lm.get("hop", args.hop)
        args.n_mels = lm.get("n_mels", args.n_mels)
        args.stack = lm.get("stack", args.stack)
        ae = cfg.get("ae", {})
        args.bottleneck = ae.get("bottleneck", args.bottleneck)
        args.epochs = ae.get("epochs", args.epochs)
        args.batch = ae.get("batch", args.batch)
        args.lr = ae.get("lr", args.lr)

    if not args.machine:
        raise SystemExit("Thiếu --machine (hoặc machine trong config.json).")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = collect_all(args.machine)
    train_files = data["train_normal"]
    test_normal = data["test_normal"]
    test_abn = data["test_abnormal"]

    # --------- Build training matrix (all stacked vectors from normal clips) ----------
    X_train = []
    for p in train_files:
        y, sr = load_audio(p, sr_target=args.sr)
        lm = compute_logmel(y, sr, args.n_fft, args.hop, args.n_mels)
        X = frames_to_stacked_vectors(lm, stack=args.stack)
        X_train.append(X)
    X_train = np.vstack(X_train)  # (N, 5*64)

    # --------- Normalize (paper: normalize log-mel by train mean/std) :contentReference[oaicite:4]{index=4} ----------
    mu = X_train.mean(axis=0, keepdims=True)
    sig = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mu) / sig

    ds = TensorDataset(torch.from_numpy(X_train))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

    model = DenseAE(in_dim=X_train.shape[1], bottleneck=args.bottleneck).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # --------- Train ----------
    model.train()
    for ep in range(1, args.epochs + 1):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            yb = model(xb)
            loss = loss_fn(yb, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"epoch {ep:02d} | loss {total/len(ds):.6f}")

    # --------- Score clips ----------
    model.eval()

    def clip_score(path):
        y, sr = load_audio(path, sr_target=args.sr)
        lm = compute_logmel(y, sr, args.n_fft, args.hop, args.n_mels)
        X = frames_to_stacked_vectors(lm, stack=args.stack)
        X = (X - mu) / sig
        return score_clip(model, X, device)

    scores = []
    labels = []

    for p in test_normal:
        scores.append(clip_score(p))
        labels.append(0)

    for p in test_abn:
        scores.append(clip_score(p))
        labels.append(1)

    auc = roc_auc_score(labels, scores)
    print(f"\n✅ AUC (logmel + dense AE) = {auc:.4f}")


if __name__ == "__main__":
    main()
