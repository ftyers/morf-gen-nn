# morf-gen-nn

## Morphological generation with neural networks

The whole thing is based on:

https://github.com/mila-udem/blocks-examples/tree/master/reverse_words

It has two scripts:

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

Uses `training.txt` to train a model for `20` epochs and store the output in `russian.20.model`

```
python3 train.py vocab.tsv training.txt 20 russian.20.model
```


#### Predict:

Uses `russian.20.model` to generate the forms in `training.txt` with an _n_-best list of `10`.

```
python3 predict.py vocab.tsv test.txt 10 russian.20.model
```

### Performance

Time for training one model running for 25 epochs:
```
real	74m10.590s
user	73m37.860s
sys	0m12.960s
```

Time for decoding the test set of 1,594 samples with a beam size of 10:

```
Accuracy: 98.49	1594.0	1570.0

real	2m45.731s
user	2m44.396s
sys	0m0.624s
```

Thats approximately 10 words/sec on a normal laptop.

### Results

On Task 1 of the SigMorPhon 2016 shared task:

| Language | Accuracy | Winning system  | Relative | 
-----------|----------|----------------------------|
| Arabic   | 00.00    | 95.47           | -0.00    |
| Finnish  | 00.00    | 96.80           | -0.00    |
| Georgian | 00.00    | 98.50           | -0.00    |
| German   | 00.00    | 95.80           | -0.00    |
| Hungarian| 00.00    | 99.30           | -0.00    |
| Maltese  | 00.00    | 88.99           | -0.00    |
| Navajo   | 00.00    | 91.48           | -0.00    |
| Russian  | 00.00    | 91.46           | -0.00    |
| Spanish  | 00.00    | 98.84           | -0.00    |
| Turkish  | 98.49    | 98.93           | -0.44    |

Ideas to improve performance:

* Use an ensemble of five models (like in the MED system)
* Use character embeddings
