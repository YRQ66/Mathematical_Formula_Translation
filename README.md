# MathMatch
수학 개념 매칭 프로그램 (스마일 게이트 희망스튜디오 퓨처랩 AI 부문 2기 지원 프로젝트)

# TO DO
- [x] validation bleu
- [x] validation cer
- [x] best model 저장
- [x] wandb → mlflow
- [x] I2L-140K 데이터셋 preprocessing
- [ ] test emr
- [ ] 새로 생성한 데이터셋 validation으로 최종 성능 확인

## Requiremets
- Apple Silicon (M1) version
```
pip install -r requirements_all.txt
```
torch는 Apple Silicon 버전으로 맞추어 있기 때문에 torch는 따로 install 
- Windows
```
pip install -r requirements.txt
&& 디바이스에 맞는 torch 버전 다운
```

## Directory Structure
[Dataset Source](https://zenodo.org/record/56198#.YtPD1-xBzze)
```
cd ./data
iconv -c -f ISO-8859-1 -t utf-8 im2latex_formulas.lst > im2latex_formulas_utf.lst
.
├── README.md
├── im2latex
│   ├── config
│   │   ├── config_eval_ex.yaml
│   │   └── config_train_ex.yaml
│   ├── create_dataset
│   ├── data
│   │   ├── 100K
│   │   │   ├── formulas.lst
│   │   │   ├── images
│   │   │   ├── readme.txt
│   │   │   ├── test.lst
│   │   │   ├── train.lst
│   │   │   └── valid.lst
│   │   ├── 140K
│   │   │   ├── formulas.txt
│   │   │   ├── images
│   │   │   ├── test.pkl
│   │   │   ├── train.pkl
│   │   │   ├── valid.pkl
│   │   │   └── tokenizer-wordlevel.json
│   ├── dataset.py
│   ├── evaluate.py
│   ├── metric.py
│   ├── preprocess.py
│   ├── requirements.txt
│   ├── requirements_all.txt
│   ├── tokenizer.py
│   ├── train.py
│   └── visualize.py
└── mathmatch
    └── data_building
        ├── crawler.py
        └── data
            └── crawling_data.json
```
## Quick Start
[Code Source](https://www.kaggle.com/code/younghoshin/finetuning-trocr)
1. python tokenizer.py -> ./data/tokenizer-wordlevel.json
2. python train.py
