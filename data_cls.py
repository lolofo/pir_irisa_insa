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
                         fine_tune: bool = False
                         ):
    """
    :param model_type: flaubert or camembert model
    :param fine_tune: model fine tune or raw models

    :return: tuple of numpy arrays
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
            model = CamembertForSequenceClassification.from_pretrained('./camembert_base_1',
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
    model.eval()

    # creation of dataloader
    train_dataset = HFDataset(dataset["train"])
    test_dataset = HFDataset(dataset["test"])

    # with a better computer choose a batch_size much more larger !
    # It will be faster
    train_dl = DataLoader(train_dataset, batch_size=batch)
    test_dl = DataLoader(test_dataset, batch_size=batch)

    train_array_5 = None
    train_array_7 = None
    train_array_9 = None
    train_array_11 = None
    train_array_12 = None
    train_label = None

    test_array_5 = None
    test_array_7 = None
    test_array_9 = None
    test_array_11 = None
    test_array_12 = None
    test_label = None

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(train_dl)):
            input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if i == 0:
                train_array_5 = outputs.hidden_states[5][:, 0, :].detach().numpy()
                train_array_7 = outputs.hidden_states[7][:, 0, :].detach().numpy()
                train_array_9 = outputs.hidden_states[9][:, 0, :].detach().numpy()
                train_array_11 = outputs.hidden_states[11][:, 0, :].detach().numpy()
                train_array_12 = outputs.hidden_states[12][:, 0, :].detach().numpy()
                train_label = labels
            else:
                train_array_5 = np.concatenate((train_array_5, outputs.hidden_states[5][:, 0, :].detach().numpy()),
                                               axis=0)
                train_array_7 = np.concatenate((train_array_7, outputs.hidden_states[7][:, 0, :].detach().numpy()),
                                               axis=0)
                train_array_9 = np.concatenate((train_array_9, outputs.hidden_states[9][:, 0, :].detach().numpy()),
                                               axis=0)
                train_array_11 = np.concatenate((train_array_11, outputs.hidden_states[11][:, 0, :].detach().numpy()),
                                                axis=0)
                train_array_12 = np.concatenate((train_array_12, outputs.hidden_states[12][:, 0, :].detach().numpy()),
                                                axis=0)
                train_label = np.concatenate((train_label, labels), axis=0)

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(test_dl)):
            input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if i == 0:
                test_array_5 = outputs.hidden_states[5][:, 0, :].detach().numpy()
                test_array_7 = outputs.hidden_states[7][:, 0, :].detach().numpy()
                test_array_9 = outputs.hidden_states[9][:, 0, :].detach().numpy()
                test_array_11 = outputs.hidden_states[11][:, 0, :].detach().numpy()
                test_array_12 = outputs.hidden_states[12][:, 0, :].detach().numpy()
                test_label = labels
            else:
                test_array_5 = np.concatenate((test_array_5, outputs.hidden_states[5][:, 0, :].detach().numpy()),
                                              axis=0)
                test_array_7 = np.concatenate((test_array_7, outputs.hidden_states[7][:, 0, :].detach().numpy()),
                                              axis=0)
                test_array_9 = np.concatenate((test_array_9, outputs.hidden_states[9][:, 0, :].detach().numpy()),
                                              axis=0)
                test_array_11 = np.concatenate((test_array_11, outputs.hidden_states[11][:, 0, :].detach().numpy()),
                                               axis=0)
                test_array_12 = np.concatenate((test_array_12, outputs.hidden_states[12][:, 0, :].detach().numpy()),
                                               axis=0)
                test_label = np.concatenate((test_label, labels), axis=0)

    return (train_array_5, train_array_7, train_array_9, train_array_11, train_array_12, test_array_5,
            test_array_7, test_array_9, test_array_11, test_array_12, train_label, test_label)


##############################
### load and save the data ###
##############################


# FlauBERT

(tr_array_5, tr_array_7, tr_array_9, tr_array_11, tr_array_12, te_array_5,
 te_array_7, te_array_9, te_array_11, te_array_12, tr_label, te_label) = load_train_test_data()

with open('numpy_save/flaubert_raw.npy', 'wb') as f:
    np.save(f, tr_array_5)
    np.save(f, tr_array_7)
    np.save(f, tr_array_9)
    np.save(f, tr_array_11)
    np.save(f, tr_array_12)

    np.save(f, te_array_5)
    np.save(f, te_array_7)
    np.save(f, te_array_9)
    np.save(f, te_array_11)
    np.save(f, te_array_12)

    np.save(f, tr_label)
    np.save(f, te_label)


(tr_array_5, tr_array_7, tr_array_9, tr_array_11, tr_array_12, te_array_5,
 te_array_7, te_array_9, te_array_11, te_array_12, tr_label, te_label) = load_train_test_data(fine_tune=True)

with open('numpy_save/flaubert_ft.npy', 'wb') as f:
    np.save(f, tr_array_5)
    np.save(f, tr_array_7)
    np.save(f, tr_array_9)
    np.save(f, tr_array_11)
    np.save(f, tr_array_12)

    np.save(f, te_array_5)
    np.save(f, te_array_7)
    np.save(f, te_array_9)
    np.save(f, te_array_11)
    np.save(f, te_array_12)

    np.save(f, tr_label)
    np.save(f, te_label)
    

# CamemBERT

(tr_array_5, tr_array_7, tr_array_9, tr_array_11, tr_array_12, te_array_5,
 te_array_7, te_array_9, te_array_11, te_array_12, tr_label, te_label) = load_train_test_data(model_type="camembert")

with open('numpy_save/camembert_raw.npy', 'wb') as f:
    np.save(f, tr_array_5)
    np.save(f, tr_array_7)
    np.save(f, tr_array_9)
    np.save(f, tr_array_11)
    np.save(f, tr_array_12)

    np.save(f, te_array_5)
    np.save(f, te_array_7)
    np.save(f, te_array_9)
    np.save(f, te_array_11)
    np.save(f, te_array_12)

    np.save(f, tr_label)
    np.save(f, te_label)

(tr_array_5, tr_array_7, tr_array_9, tr_array_11, tr_array_12, te_array_5,
 te_array_7, te_array_9, te_array_11, te_array_12, tr_label, te_label) = load_train_test_data(model_type="camembert",
                                                                                              fine_tune=True)

with open('numpy_save/camembert_ft.npy', 'wb') as f:
    np.save(f, tr_array_5)
    np.save(f, tr_array_7)
    np.save(f, tr_array_9)
    np.save(f, tr_array_11)
    np.save(f, tr_array_12)

    np.save(f, te_array_5)
    np.save(f, te_array_7)
    np.save(f, te_array_9)
    np.save(f, te_array_11)
    np.save(f, te_array_12)

    np.save(f, tr_label)
    np.save(f, te_label)