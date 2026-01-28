```
Acoustic-Anomaly-Detection-for-Machines/
│
├── data/
│   └── raw/
│       ├── fan/
│       └── valve/
│
├── extract/
│   └── features/
│       └── mfcc/
│           └── fan/
│               ├── train_normal.csv
│               ├── test_normal.csv
│               └── test_abnormal.csv
│
├── logs/
│   ├── mfcc.log
│   ├── chroma.log
│   └── data_check.log
│
├── results/
│   └── mfcc/
│       └── fan/
│           ├── train_normal/
│           ├── test_normal/
│           ├── test_abnormal/
│           └── audio_summary.csv
│
├── src/
│   ├── dsp/
│   │   ├── __init__.py
│   │   └── dsp_mfcc.py
│   │
│   ├── audio_utils.py
│   ├── check_data.py
│   ├── extract_mfcc.py
│   ├── extract_chroma.py
│   ├── extract_stft.py
│   ├── load_data.py
│   └── rename_arrange.py
│
├── load_data.py
├── .gitignore
└── README.md
```
