# Clip Playground

## General library: `load_from_clip.py`

Check its doc strings.

## Disease classification: `neural_covid_pneu.py`

This script uses the COVID-US dataset to test whether the embeddings made by various CLIP-based models are good enough for disease classification.

Simply run it to get accuracy results. Its code specifies which CLIP-based models are being tested. You can change the models at the top of the code of `neural_covid_pneu.py`.