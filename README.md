# morf-gen-nn

## Morphological generation with neural networks

* `train.py`: train a model for generating morphology
* `predict.py`: run the model on a test corpus

### Prerequisites

You need blocks and Theano.

### Input and output format:

#### Vocabulary (symbol table)

This is a tab-separated file with all the symbols you're going to use in your 
input and a unique identifying integer. The `<S>`, `</S>` and `<UNK>` are 
reserved symbols.

```
2	c
3	e
4	а
5	б
37	<S>
38	</S>
39	<UNK>
40	Animacy=Anim
41	Animacy=Inan
```

#### Training data

The input format for forms and analyses is a three column table with either tab as a separator
or `|||`. The first column is surface form, the second column is lemma and the third column is 
morphological tags in `Feature=Value` pairs. Features are separated by a single `|` character.

```
студентов|||студент|||POS=NOUN|Animacy=Anim|Case=Gen|Gender=Masc|Number=Plur
студенты|||студент|||POS=NOUN|Animacy=Anim|Case=Nom|Gender=Masc|Number=Plur
мальчик|||мальчик|||POS=NOUN|Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing
студент|||студент|||POS=NOUN|Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing
```

### Usage

#### Train: 

Uses `training.txt` to train a model for `200` iterations and store the output in `russian.200.model`

```
python3 train.py vocab.tsv training.txt 200 russian.200.model
```


#### Predict:

Uses `russian.200.model` to generate the forms in `training.txt` with an _n_-best list of `10`.

```
python3 predict.py vocab.tsv test.txt 10 russian.200.model
```
