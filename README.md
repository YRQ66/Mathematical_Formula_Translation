# MathMatch
수학 개념 매칭 프로그램 (스마일 게이트 희망스튜디오 퓨처랩 AI 부문 2기 지원 프로젝트)

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
[Dataset Source](https://anvilproject.org/guides/content/creating-links)
```
cd ./data
iconv -c -f ISO-8859-1 -t utf-8 im2latex_formulas.lst > im2latex_formulas_utf.lst

.
├── data
│   ├── formula_images_processed
│   ├── im2latex_formulas.lst 
│   ├── im2latex_formulas_utf.lst 
│   ├── im2latex_test.lst
│   ├── im2latex_train.lst
│   ├── im2latex_validate.lst
│   └── tokenizer-wordlevel.json
├── preprocess.py
├── tokenizer.py
└── visualize.py
```
## Quick Start
1. python tokenizer.py -> ./data/tokenizer-wordlevel.json
2. python train.py
