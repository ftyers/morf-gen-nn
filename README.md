# morf-gen-nn

## Morphological generation with neural networks

* `train.py`: train a model for generating morphology
* `predict.py`: run the model on a test corpus

### Input and output format:

#### Vocabulary

```
2	c
3	e
4	а
5	б
```

#### Training data

студентов|||студент|||POS=NOUN|Animacy=Anim|Case=Gen|Gender=Masc|Number=Plur
студенты|||студент|||POS=NOUN|Animacy=Anim|Case=Nom|Gender=Masc|Number=Plur
мальчик|||мальчик|||POS=NOUN|Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing
студент|||студент|||POS=NOUN|Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing

### Usage

#### Train: 

```
python3 morf-gen-nn/train.py data/vocab.tsv data/training.txt 200 models/russian.200.model
```

#### Predict:

```
python3 morf-gen-nn/predict.py data/vocab.tsv data/test.txt 10 models/russian.200.model
```
