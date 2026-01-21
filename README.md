```
Acoustic-Anomaly-Detection-for-Machines/
│
├── data/
│   ├── raw/
│   │   └── valve/
│   │       ├── train/
│   │       │   ├── normal/
│   │       │   ├── normal1/
│   │       │   ├── normal2/
│   │       │   └── normal3/
│   │       └── test/
│   │           ├── abnormal/
│   │           ├── abnormal1/
│   │           ├── abnormal2/
│   │           ├── abnormal3/
│   │           └── normal/
│   │
│   ├── processed/       
│   └── features/          
│       └── mfcc/
│           └── valve/
│
├── results/
│   ├── mfcc_plots/
│   │   └── valve/
│   ├── audio_summary.csv
│   └── metrics.csv
│
├── logs/
│   └── data_check.log
│
├── src/
│   ├── check_data.py
│   ├── extract_mfcc.py
│   ├── prepare_dataset.py
│   ├── train_ocsvm.py
│   ├── train_iforest.py
│   ├── train_gmm.py
│   └── utils.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── .gitignore            
├── requirements.txt        
└── README.md            
