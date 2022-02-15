# Research initiation project

As part of my studies at INSA, in my 4th year I chose to do a research initiation project with the IRISA laboratory in Rennes. The theme of this project was to analyze (statistically) the behaviors of hidden embeddings on a BERT transformers architecture. We focused on 2 models that were trained on French lexicons (flauBERT and camemBERT).

We wanted to compare the behavior of embeddings in two different situations:

- First, on simply pre-trained architectures.

- then on the same architectures but trained on a sentence classification task.


## Table of contents

1. [pre_trained](#pre_trained)
2. [fine_tuning](#fine_tuning)
3. [fine_tuned_models](#fine_tuned_models)



## pre_trained

In this section you will find a notebook which presents the analysis on the models called "Raw" that is to say not specified on a particular spot. These models are directly available on the Huggin face **transformers** library

## fine_tuning

In this section you will find the python scripts we used for the fine tuning of our models.
We did a fine tuning of the FLUE/CLS data. 

## fine_tuned_models


