# Research initiation project

As part of my studies at INSA, in my 4th year I chose to do a research initiation project with the IRISA laboratory in
Rennes. The theme of this project was to analyze (statistically) the behaviors of hidden embeddings on a BERT
transformers architecture. We focused on 2 models that were trained on French lexicons (flauBERT and camemBERT).

We wanted to compare the behavior of embeddings in two different situations:

- First, on simply pre-trained architectures.

- then on the same architectures but trained on a sentence classification task. (simple)

## Table of contents

1. [fine tuning](#fine-tuning)
2. [models study](#models-study)
3. [convergence and performance](#convergence-and-performance)

## fine tuning

In this section you will find the python scripts we used for the fine tuning of our models, and to generate our data.

First of all go to the root of this directory. There is two scripts and here are the command line to :

- first generate the data. The following commands will generate two objects of type **datasets** which contain the data
  to specialize our models.

```commandline
python flue-cls-prepare.py -c flaubert/flaubert_base_cased flaubert-tokens.datasets 
python flue-cls-prepare.py -c camembert-base camembert-tokens.datasets
```

- finally, fine tune the models. The next commands will allow to specialize the models, thanks to the data we just
  generated.

```commandline
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

## models study

Now that you have created the data and models for our study, you can come and run the two following notebooks :

- **raw_study.ipynb** in this notebook you will find the study we conducted on the so-called pre-trained models.

- **fine_tuning_study.ipynb** this notebook contains a very similar study, but this time conducted on models that were
  specialized using the scripts written in the previous section.

## Convergence and Performance

After these two study we made some different observations about the convergence of the cosines through the layers.
The objective of this final part is now to see if there is any links between this convergence of the cosines and the
performance in the classification task.
To answer this question we will first take the embedding of the *[CLS]* tokens on different layers. To get these
embeddings you can execute the following command line :

```commandline
python data_cls.py
```

If you want to speed up this phase to get the data (which is still long),
you can change the batch size by adding the argument -b in the command line.
For example, if you want a batch of size 32, the command line would be as follows:

```commandline
python data_cls.py -b 32
```

With these command lines you will get all the files that are necessary to run the notebook **cls_study.ipynb**.




