import torch
import datasets
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import argparse

from transformers import FlaubertForSequenceClassification
from transformers import CamembertForSequenceClassification

import tqdm

datadir_camembert = "camembert-tokens.datasets"
datadir_flaubert = "flaubert-tokens.datasets"

tokenized_datasets_camembert = datasets.load_from_disk(datadir_camembert)
tokenized_datasets_flaubert = datasets.load_from_disk(datadir_flaubert)

device = torch.device("cpu")

######################
### The batch size ###
######################

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int)
args = parser.parse_args()

batch = 4  # default batch : 4
if args.batch_size is not None:
    batch = args.batch_size


#############################################
### Transform the data into torch dataset ###
#############################################

class HFDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, idx):
        return torch.tensor(self.dset[idx]["input_ids"]), torch.tensor(self.dset[idx]["attention_mask"]), \
               torch.tensor(self.dset[idx]["label"])

    def __len__(self):
        return len(self.dset)


############################################
### function to get the embedding of CLS ###
############################################

def load_train_test_data(model_type: str = 'flaubert',
                         fine_tune: bool = False,
                         layer: int = 12):
    """
    :param model_type: flaubert or camembert model
    :param fine_tune: model fine tune or raw models
    :param layer: from which layer we take the token CLS
    :return: two numpy arrays one for the training one for the test part
    """

    model = None
    dataset = None

    # flaubert_model
    if model_type == "flaubert":
        dataset = tokenized_datasets_flaubert
        if fine_tune:
            # we use the model we fine tuned on 1 epoch
            model = FlaubertForSequenceClassification.from_pretrained('./flaubert_base_cased_1',
                                                                      num_labels=2,
                                                                      output_attentions=False,
                                                                      output_hidden_states=True)
        else:
            # we use the raw model from the huggin face library
            model = FlaubertForSequenceClassification.from_pretrained('flaubert/flaubert_base_cased',
                                                                      num_labels=2,
                                                                      output_attentions=False,
                                                                      output_hidden_states=True)
    # camembert model
    else:
        dataset = tokenized_datasets_camembert

        if fine_tune:
            # we use the model we fine tuned on 1 epoch
            model = FlaubertForSequenceClassification.from_pretrained('./camembert_base_1',
                                                                      num_labels=2,
                                                                      output_attentions=False,
                                                                      output_hidden_states=True)
        else:
            # we use the raw model from the huggin face library
            model = CamembertForSequenceClassification.from_pretrained('camembert-base',
                                                                       num_labels=2,
                                                                       output_attentions=False,
                                                                       output_hidden_states=True)

    # model to the cpu
    model.to(device)

    # creation of dataloader
    train_dataset = HFDataset(dataset["train"])
    test_dataset = HFDataset(dataset["test"])

    # with a better computer choose a batch_size much more larger !
    # It will be faster
    train_dl = DataLoader(train_dataset, batch_size=batch)
    test_dl = DataLoader(test_dataset, batch_size=batch)

    train_array = None
    train_label = None
    test_array = None
    test_label = None

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(train_dl)):
            input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            buff = outputs.hidden_states[layer - 1][:, 0, :].detach().numpy()
            if i == 0:
                train_array = buff
                train_label = labels
            else:
                train_array = np.concatenate((train_array, buff), axis=0)
                train_label = np.concatenate((train_label, labels), axis=0)

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(test_dl)):
            input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            buff = outputs.hidden_states[layer - 1][:, 0, :].detach().numpy()
            if i == 0:
                test_array = buff
                test_label = labels.detach().numpy()
            else:
                test_array = np.concatenate((test_array, buff), axis=0)
                test_label = np.concatenate((test_label, labels.detach().numpy()), axis=0)

    return train_array, train_label, test_array, test_label


##############################
### load and save the data ###
##############################

# first for FlauBERT

"""
train_fl_12, train_label_fl_12, test_fl_12, test_label_fl_12 = load_train_test_data()

with open('numpy_save/flaubert_raw_layer_12.npy', 'wb') as f:
    np.save(f, train_fl_12)
    np.save(f, train_label_fl_12)
    np.save(f, test_fl_12)
    np.save(f, test_label_fl_12)

train_fl_13, train_label_fl_13, test_fl_13, test_label_fl_13 = load_train_test_data(layer=13)

with open('numpy_save/flaubert_raw_layer_13.npy', 'wb') as f:
    np.save(f, train_fl_13)
    np.save(f, train_label_fl_13)
    np.save(f, test_fl_13)
    np.save(f, test_label_fl_13)

train_fl_ft, train_label_fl_ft, test_fl_ft, test_label_fl_ft = load_train_test_data(fine_tune=True,
                                                                                    layer=13)

with open('numpy_save/flaubert_ft.npy', 'wb') as f:
    np.save(f, train_fl_ft)
    np.save(f, train_label_fl_ft)
    np.save(f, test_fl_ft)
    np.save(f, test_label_fl_ft)
"""

# then for CamemBERT

train_cb_12, train_label_cb_12, test_cb_12, test_label_cb_12 = load_train_test_data(model_type="camembert")

with open('numpy_save/flaubert_raw_layer_12.npy', 'wb') as f:
    np.save(f, train_cb_12)
    np.save(f, train_label_cb_12)
    np.save(f, test_cb_12)
    np.save(f, test_label_cb_12)

train_cb_13, train_label_cb_13, test_cb_13, test_label_cb_13 = load_train_test_data(model_type="camembert",
                                                                                    layer=13)

with open('numpy_save/flaubert_raw_layer_13.npy', 'wb') as f:
    np.save(f, train_cb_13)
    np.save(f, train_label_cb_13)
    np.save(f, test_cb_13)
    np.save(f, test_label_cb_13)

train_cb_ft, train_label_cb_ft, test_cb_ft, test_label_cb_ft = load_train_test_data(model_type="camembert",
                                                                                    fine_tune=True,
                                                                                    layer=13)

with open('numpy_save/flaubert_ft.npy', 'wb') as f:
    np.save(f, train_cb_ft)
    np.save(f, train_label_cb_ft)
    np.save(f, test_cb_ft)
    np.save(f, test_label_cb_ft)
