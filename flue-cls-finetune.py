#!/usr/bin/env python

import os
import sys
import getopt
import logging

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('flue-cls-finetune')

checkpoint = 'flaubert/flaubert_base_cased'
ntrains = 500
ntests = 100
nepochs = 1
savedir = None

datadir = None


# ===================================================================
#
# define a number of constants we'll be using
#
def usage():
    print('flue-cls-finetune.py [options]')
    print('Synopsis:')
    print('  Finetune model on the FLUE/CLS task.')
    print('Options:')
    print(
        '   -d, --datasets=s        load tokenized datasets from directory d (default: load and tokenize datasets from hub)')
    print('   -a, --num-train=n       set number of training samples (default: all)')
    print('   -b, --num-test=n        set number of test samples (default: {})'.format(numtests))
    print('   -c, --checkpoint=s      set checkpoint to finetune (default: {})'.format(checkpoint))
    print('   -n, --num-epochs=n      number of epochs (default: {})'.format(nepochs))
    print("   -s, --save=s            save model to directory s (default: checkpoint's basename)")
    print('   -h, --help              print this help')


# ===================================================================
#
# parse command line
#
try:
    opts, args = getopt.getopt(sys.argv[1:], "ha:b:c:n:s:d:",
                               ["help", "num-train=", "num-test=", "checkpoint=", "num-epochs=", "save=", "dataset="])
except getopt.GetoptError as err:
    print(err)  # will print something like "option -a not recognized"
    sys.exit(1)

# print(opts)

for o, a in opts:
    if o in ("-h", "--help"):
        usage()
        sys.exit(0)
    elif o in ('-a', '--num-train'):
        ntrains = int(a)
    elif o in ('-b', '--num-test'):
        ntests = int(a)
    elif o in ('-c', '--checkpoint'):
        checkpoint = a
    elif o in ('-n', '--num-epochs'):
        nepochs = int(a)
    elif o in ('-s', '--save'):
        savedir = a
    elif o in ('-d', '--dataset'):
        datadir = a
    else:
        assert False, "unhandled option {}".format(o)

if savedir == None:
    savedir = os.path.basename(checkpoint)

# ===================================================================
#
# now do the job
#

#
# Faster and simpler way is to load directly from HuggingFace's dataset
#


from datasets import load_dataset, load_from_disk

if datadir == None:

    logging.info('loading dataset from HuggingFace hub')

    datasets = load_dataset("flue", "CLS")
    for split in datasets.keys():
        logging.info('loaded {} entries from {} split'.format(len(datasets[split]), split))

    #
    # Load tokenizer
    #

    logging.info('loading tokenizer for {}'.format(checkpoint))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    #
    # tokenize dataset
    #

    logging.info('tokenizing datasets')

    tokenized_datasets = datasets.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True),
                                      batched=True)

else:

    logging.info('loading dataset from local directory {}'.format(datadir))

    tokenized_datasets = load_from_disk(datadir)
    for split in tokenized_datasets.keys():
        logging.info('loaded {} entries from {} split'.format(len(tokenized_datasets[split]), split))

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(ntrains)) if ntrains != None else \
tokenized_datasets["train"].shuffle(seed=42)
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(ntests)) if ntests != None else \
tokenized_datasets["test"].shuffle(seed=42)

#
# load model
#

logging.info('loading model from {}'.format(checkpoint))

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

#
# run finetuning
#

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import TrainingArguments

training_args = TrainingArguments("flaubert-cls-finetune", num_train_epochs=nepochs)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained(savedir)

#
# evaluate finetuning
#

out = trainer.evaluate()
print(out)

logging.info("BYE!")
