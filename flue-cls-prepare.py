#!/usr/bin/env python

import os
import sys
import getopt
import logging

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('flue-cls-finetune')

checkpoint = 'flaubert/flaubert_base_cased'


# ===================================================================
#
# define a number of constants we'll be using
#
def usage():
    print('flue-cls-prepare.py [options] dir')
    print('Synopsis:')
    print('  Prepare data to be ready for finetuning given checkpoint and dump to dir')
    print('Options:')
    print('   -c, --checkpoint=s      set checkpoint to finetune (default: {})'.format(checkpoint))
    print('   -h, --help              print this help')


# ===================================================================
#
# parse command line
#
try:
    opts, args = getopt.getopt(sys.argv[1:], "hc:", ["help", "checkpoint="])
except getopt.GetoptError as err:
    print(err)  # will print something like "option -a not recognized"
    sys.exit(1)

for o, a in opts:
    if o in ("-h", "--help"):
        usage()
        sys.exit(0)
    elif o in ('-c', '--checkpoint'):
        checkpoint = a
    else:
        assert False, "unhandled option {}".format(o)

if len(args) != 1:
    logging.error('Invalid number of arguments (use --help for usage)')
    sys.exit(0)

out = args[0]

# ===================================================================
#
# now do the job
#

#
# Faster and simpler way is to load directly from HuggingFace's dataset
#

logging.info('loading dataset')

from datasets import load_dataset

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

tokenized_datasets = datasets.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True), batched=True)

#
# tokenize dataset
#

logging.info('saving datasets to file {}'.format(out))

tokenized_datasets.save_to_disk(out)

logging.info("BYE!")
