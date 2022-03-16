import warnings
warnings.filterwarnings("ignore")
import random
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import sys
import os
# Set up TWINT config
import twint
# Solve compatibility issues with notebooks and RunTime errors.
import nest_asyncio

# stopword
import nltk
from nltk.corpus import stopwords
nltk.download("vader_lexicon")
nltk.download('stopwords')
sw_nltk = stopwords.words('english')

import re
text_cleaning_regex = "@S+|https?:S+|http?:S|[^A-Za-z0-9]+"

# sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,roc_curve, auc, f1_score,cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# bert
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler,random_split
from transformers import BertModel,BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AdamW, get_linear_schedule_with_warmup



# Create a function to tokenize the input for encoder
# Load bert tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def preprocessing_for_bert(X, y, batch_size=32):
    '''
    :param X: array
    :param y: array
    :param batch_size:
    :return: dataloader for bert training
    '''
    input_ids = []
    attention_masks = []

    for sent in X:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=280,  # maximum twitter lenght 280
            pad_to_max_length=True,
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,
            truncation=True
        )

        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    labels = torch.tensor(y)
    # convert the tensors into a PyTorch Dataset=
    data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(data)
    # feed dataset to training loop
    dataloader = DataLoader(data,  # The training samples.
                            sampler=sampler,  # Select batches randomly
                            batch_size=batch_size)

    return dataloader


def initialize_model(epochs=4):
    '''
    :param epochs:
    :return:  Bert Classifier, the optimizer and the learning rate scheduler.
    '''

    bert_classifier = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                    num_labels=3,
                                                                    output_attentions=False,
                                                                    output_hidden_states=False)

    #     #Tell PyTorch to run the model on GPU
    #     bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(), lr=0.001, eps=1e-8)

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Warm up steps is a parameter which is used to lower the learning rate in order to reduce the impact of deviating the model from learning on sudden new data set exposure.

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=50,
                                                num_training_steps=total_steps)

    return bert_classifier, optimizer, scheduler


# evaluation metrics

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def kappa_score(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return cohen_kappa_score(preds_flat, labels_flat, labels=None, weights=None)


# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(model, train_dataloader, epochs):
    """
    Train the BertClassifier model
    """
    train_loss_set = []
    print("Start training...\n")

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()  # Measure how long the training epoch takes.
        total_train_loss = 0  # Reset the total loss for this epoch.
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
            loss = output.loss
            logits = output.logits

            total_train_loss += loss.item()
            # Perform a backward pass to compute gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update the model’s parameters
            optimizer.step()
            # Update the learning rate
            scheduler.step()
            # Calculate the average loss over all of the batches.

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

# save model
output_model = './sentiment_bert.pth'

def save(bert_classifier, optimizer):
    torch.save({
        'bert_classifier_state_dict': bert_classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

save(bert_classifier, optimizer)

# to load
# checkpoint = torch.load('./sentiment_bert.pth')
# bert_classifier.load_state_dict(checkpoint['bert_classifier_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])