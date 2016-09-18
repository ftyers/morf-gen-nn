# morf-gen-nn

## Morphological generation with neural networks

The whole thing is based on code from:

https://github.com/mila-udem/blocks-examples/tree/master/reverse_words

With inspiration from [1]

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

__Note:__ It seems like you need the vocabulary to be divisible by 10, so if you have some 
vocabulary size that is not, then you probably need to pad it with unique symbols. If you 
don't, you'll get some weird errors from Theano.

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

On Task 1 of the SIGMORPHON 2016 shared task:

| Language  | Accuracy | Winning system[1] | Relative | 
------------|----------|-------------------|----------|
| Arabic    | 00.00    | 95.47             | -0.00    |
| Finnish   | 00.00    | 96.80             | -0.00    |
| Georgian  | 97.51    | 98.50             | -0.99    |
| German    | 94.48    | 95.80             | -1.32    |
| Hungarian | 98.99    | 99.30             | -0.31    |
| Maltese   | 00.00    | 88.99             | -0.00    |
| Navajo    | 55.24    | 91.48             | -36.24   |
| Russian   | 88.74    | 91.46             | -2.72    |
| Spanish   | 98.50    | 98.84             | -0.34    |
| Turkish   | 98.49    | 98.93             | -0.44    |

* dev + train combined
* _n_ epochs = 30
* beam size = 10

Ideas to improve performance:

* Use an ensemble of five models (like in the MED system)
* Use character embeddings

## References 

1. Kann, K. and Schütze, H. (2016) "MED: The LMU system for the SIGMORPHON 2016 shared task on morphological reinflection". _Proceedings of the 2016 Meeting of SIGMORPHON_, Berlin, Germany. Association for Computational Linguistics

