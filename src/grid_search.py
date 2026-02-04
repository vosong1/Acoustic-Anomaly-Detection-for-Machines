import argparse
import itertools
import json
import os
import subprocess
import pandas as pd

def run(cmd):
    print("[CMD]", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", type=str, default="fan")
    ap.add_argument("--out_csv", type=str, default="results_grid.csv")
    ap.add_argument("--python", type=str, default="python")
    args = ap.parse_args()

    # Grid mặc định (bạn chỉnh ở đây)
    mfcc_grid = {
        "n_mfcc": [11, 20],
        "frame_size": [512, 1024],
        "hop_size": [256, 512],
        "n_fft": [512, 1024],
    }
    svm_grid = {
        "nu": [0.05, 0.1, 0.15],
        "gamma": ["scale", "auto", 0.01],
        "pca": [0.95],
    }

    rows = []

    for n_mfcc, frame_size, hop_size, n_fft in itertools.product(
        mfcc_grid["n_mfcc"], mfcc_grid["frame_size"], mfcc_grid["hop_size"], mfcc_grid["n_fft"]
    ):
        feat = f"mfcc_n{n_mfcc}_fs{frame_size}_h{hop_size}_fft{n_fft}"

        cfg = {
            "machine": args.machine,
            "feature_mfcc": feat,
            "mfcc": {
                "sr": 16000,
                "n_mfcc": n_mfcc,
                "frame_size": frame_size,
                "hop_size": hop_size,
                "n_fft": n_fft,
                "max_plot": 0,
                "scale_in_extract": False
            },
            "svm": {
                "scale": True
            }
        }

        os.makedirs("tmp_configs", exist_ok=True)
        cfg_path = os.path.join("tmp_configs", f"{feat}.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        # Extract MFCC
        run([args.python, "-m", "src.extract_mfcc", "--config", cfg_path])

        for nu, gamma, pca in itertools.product(svm_grid["nu"], svm_grid["gamma"], svm_grid["pca"]):
            cfg["svm"].update({"nu": nu, "gamma": gamma, "pca": pca, "kernel": "rbf"})
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            out = run([args.python, "-m", "src.SVM", "--config", cfg_path, "--feature", feat])
            auc = None
            for line in (out.stdout or "").splitlines():
                if "AUC Score" in line:
                    try:
                        auc = float(line.split(":")[-1].strip())
                    except:
                        pass

            rows.append({
                "feature": feat,
                "n_mfcc": n_mfcc,
                "frame_size": frame_size,
                "hop_size": hop_size,
                "n_fft": n_fft,
                "nu": nu,
                "gamma": gamma,
                "pca": pca,
                "auc": auc
            })
            print(rows[-1])

    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print(df.head(10))

if __name__ == "__main__":
    main()
