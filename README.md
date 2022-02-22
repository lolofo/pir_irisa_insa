# Research initiation project

As part of my studies at INSA, in my 4th year I chose to do a research initiation project with the IRISA laboratory in Rennes. The theme of this project was to analyze (statistically) the behaviors of hidden embeddings on a BERT transformers architecture. We focused on 2 models that were trained on French lexicons (flauBERT and camemBERT).

We wanted to compare the behavior of embeddings in two different situations:

- First, on simply pre-trained architectures.

- then on the same architectures but trained on a sentence classification task.


## Table of contents

1. [fine_tuning](#fine_tuning)
2. [models_study](#models_study)


## fine_tuning

In this section you will find the python scripts we used for the fine tuning of our models, and to generate our data.

First of all go to the root of this directory. There is two scripts and here are the command line to :

- first generate the data. The following commands will generate two objects of type **datasets** which contain the data to specialize our models.

```
python flue-cls-prepare.py flaubert-tokens.datasets
python flue-cls-prepare.py camembert-tokens.datasets
```
- finally, fine tune the models. The next commands will allow to specialize the models, thanks to the data we just generated.

```
python flue-cls-finetune.py -a 5000 -b 1000 -n 1 -s ./flaubert_base_cased_1 -d flaubert-tokens.datasets -c flaubert/flaubert_base_cased
python flue-cls-finetune.py -a 5000 -b 1000 -n 1 -s ./camembert_base_1 -d camembert-tokens.datasets -c camembert-base
```

Thanks to all these commands we can generate all the data and models necessary for our study.

About the parser :

```
-n # number of epochs
-a # size of the training dataset
-b # size of the test dataset
-n # number of epoch(s) for the training
-s # savedir the folder where the trained model will be saved
-d # datadir the direction of the data for the training

-c # the checkpoint where we load the pretrain model (Huggin face)
   # -c flaubert/flaubert_base_cased // flaubert fine tuning
   # -c camembert-base               // camembert fine tuning
```



## models_study

Now that you have created the data and models for our study, you can come and run the two notebooks in this directory. There are two notebooks :

- **raw_study.ipynb** in this notebook you will find the study we conducted on the so-called pre-trained models.

- **fine_tuning_study.ipynb** this notebook contains a very similar study, but this time conducted on models that were specialized using the scripts written in the previous section.






