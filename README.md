```
Acoustic-Anomaly-Detection-for-Machines
│
├── data/                         # Dataset directory
│   └── raw/
│       ├── fan/                  # Fan machine audio
│       │   ├── -6db/
│       │   ├── 0db/
│       │   └── 6db/
│       │       ├── normal/
│       │       └── abnormal/
│       │
│       └── valve/                # Valve machine audio
│
├── extract/                      # Extracted intermediate data
│
├── features/                     # Feature datasets after extraction
│
├── models/                       # Saved trained models
│   ├── gmm/
│   ├── isolation_forest/
│   └── oneclass_svm/
│
├── results/                      # Model prediction results
│   ├── gmm/
│   ├── isolation_forest/
│   └── oneclass_svm/
│
├── src/                          # Source code
│
│   ├── dsp/                      # Digital signal processing modules
│   │   ├── dsp_mfcc.py
│   │   ├── dsp_logmel.py
│   │   ├── dsp_stft.py
│   │   └── handcraft.py
│   │
│   ├── extract_mfcc.py           # MFCC feature extraction
│   ├── extract_logmel.py         # Log-Mel feature extraction
│   ├── extract_stft.py           # STFT feature extraction
│   ├── extract_chroma.py         # Chroma feature extraction
│   │
│   ├── load_data.py              # Dataset loader
│   ├── check_data.py             # Dataset checking utilities
│   ├── audio_utils.py            # Audio processing utilities
│   ├── config_utils.py           # Configuration helper functions
│   │
│   ├── m_gmm.py                  # Gaussian Mixture Model anomaly detection
│   ├── m_iso.py                  # Isolation Forest anomaly detection
│   └── m_svm.py                  # One-Class SVM anomaly detection
│
├── config.json                   # Project configuration file
├── README.md                     # Project documentation
└── paper.pdf                     # Reference research paper
```
