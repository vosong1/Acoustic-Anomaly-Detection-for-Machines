from pathlib import Path

root = Path("data/raw/fan")
for snr in ["-6db","0db","6db"]:
    n = len(list((root/snr/"normal").glob("*.wav")))
    a = len(list((root/snr/"abnormal").glob("*.wav")))
    print(snr, "normal", n, "abnormal", a)