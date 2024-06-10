# PAPER

The official repository for the "PAPER: A Persona-Aware Chain-of-Thought Learning Framework for Personalized Dialogue Response Generation".

## Installation

```python
pip install -r requirements.txt
```

## Data
You can download the original dateset HLA-Chat++ on https://drive.google.com/drive/folders/1-8kwWCo6vfmDzGk8eJ1UeMMnuUlyx3mI?usp=sharing
### Instruction data construction
```python
python data/data_process/new_data.py
```

## Training
### Persona understanding stage
```
bash flow/sfts1.sh
```
### Persona perception stage
```
python src/persona_extraction.py
bash flow/predict.sh
```
### Response generation stage
```
bash flow/sfts2.sh
```
## Inference
```
bash flow/predict.sh
```
### LLM-based Evaluation
```
python evaluation/gpt_evaluation/compute_scores.py
```


