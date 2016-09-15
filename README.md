# morf-gen-nn

## Morphological generation with neural networks

* `train.py`: train a model for generating morphology
* `predict.py`: run the model on a test corpus

### Prerequisites

You need blocks and Theano.

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

Uses `training.txt` to train a model for `200` iterations and store the output in `russian.200.model`

```
python3 morf-gen-nn/train.py vocab.tsv training.txt 200 russian.200.model
```


#### Predict:

Uses `russian.200.model` to generate the forms in `training.txt` with an _n_-best list of `10`.

```
python3 morf-gen-nn/predict.py data/vocab.tsv data/test.txt 10 models/russian.200.model
```
