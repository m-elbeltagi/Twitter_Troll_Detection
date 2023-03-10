import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import datetime
import sys 




save_path = r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets'
writer = SummaryWriter(r'runs/FintunedtinyBert')

## setting device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Device in use: {} \n'.format(device))


batch_size = 16
n_epochs = 1
num_labels = 2

## using 'engine = python' for the second file because it's large and runs into error without it
raw_troll_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets\russian_troll_tweets_200k_dataset.csv')
raw_nontroll_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets\non_troll_candidate_dataset.csv', engine='python')


## double square brackets to put in a dataframe instead of single to put in series
troll_dataset = raw_troll_data[['text']]
raw_nontroll_data = raw_nontroll_data[['tweet', 'country']]



## removing the tweets not from US, then setting the sizes to be equal, and creating the label column, 0 for non-troll, 1 for troll, and droping empty columns
nontroll_dataset = raw_nontroll_data[raw_nontroll_data['country']=='United States of America']
nontroll_dataset = nontroll_dataset[['tweet']]

nontroll_dataset.columns = ['text']
nontroll_dataset = nontroll_dataset.dropna()
troll_dataset = troll_dataset.dropna()
nontroll_dataset = nontroll_dataset.iloc[:100000]    

troll_dataset = troll_dataset.iloc[:nontroll_dataset.shape[0]]
troll_dataset['label'] = list(np.ones(nontroll_dataset.shape[0], dtype=int))
nontroll_dataset['label'] = list(0* np.ones(nontroll_dataset.shape[0], dtype=int))
combined_set = pd.concat([troll_dataset, nontroll_dataset], ignore_index=True)
combined_set = combined_set.sample(frac=1, random_state=666).reset_index(drop=True)       ##shuffling dataset



## to check if any of the text sizes exceeds the maximum context size of the model we choose (which we can't have), notice we're only looking at length in words here, but model context size is in tokens, so if it's close we need to be more careful (redo these plots after tokenizing), if it's much less than it's fine, as is the case here for the disaster tweet dataset
def plot_texts_lengths(train_dataset):
    train_dataset = Dataset.from_pandas(train_dataset)                      
    train_dataset.set_format(type='pandas')                                 
    train_dataframe = train_dataset[:]
    train_dataframe['words per text (per category)'] = train_dataframe['text'].str.split().apply(len)
    train_dataframe.boxplot('words per text (per category)', by='label', grid=False, showfliers=False, color='black')
    plt.suptitle("")
    plt.xlabel("")
    plt.show()


## note that the tokenizer used needs to match the one used for the imported pretrained model (when using hugging face pipeline it takes care of this step)
model_name = "cross-encoder/ms-marco-TinyBERT-L-2"               
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = tokenizer.model_max_length

## importing the pretrained model
transformer = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)).to(device)


def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_length)          ##padding fills to match the largest text size in the batch, truncation truncates anything longer than context size (which as we've checked we don't have any), list is to apply it to each entry in the batch, (but when just testing on one text entry, list affects the text)


combined_set = Dataset.from_pandas(combined_set)
tokenized_dataset = combined_set.map(tokenize, batched=True, batch_size=batch_size)
    

## using HF Dataset method "select" to split the data (already shuffled above) (in documentation this select takes a list, but apparently works with range() too)
train_set = tokenized_dataset.select(range(int(0.8*len(combined_set))))
test_set = tokenized_dataset.select(range(int(0.8*len(combined_set)), len(combined_set)))


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}


logging_steps = len(train_set) // batch_size
training_args  = TrainingArguments(output_dir= save_path + r'\FinetunedLLM',
                                    num_train_epochs=1,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    load_best_model_at_end=True,
                                    metric_for_best_model='f1',
                                    weight_decay=0.01,
                                    evaluation_strategy='steps',
                                    disable_tqdm=True,
                                    logging_steps=logging_steps,
                                    eval_steps=logging_steps,
                                    save_steps=logging_steps,
                                    log_level='error',
                                    report_to='tensorboard')





trainer = Trainer(model=transformer,
                    args=training_args,
                    train_dataset=train_set,
                    eval_dataset=test_set,
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])



print('start time is: {} \n'.format(datetime.datetime.now()))


# trainer.train()


print('Finsih time is: {} \n'.format(datetime.datetime.now()))



################################################
## Load trained model, using the unused dataset portion from above to double-check model accuracy

def test_accuracy():
    raw_troll_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets\russian_troll_tweets_200k_dataset.csv')
    raw_nontroll_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets\non_troll_candidate_dataset.csv', engine='python')

    troll_dataset = raw_troll_data[['text']]
    raw_nontroll_data = raw_nontroll_data[['tweet', 'country']]

    nontroll_dataset = raw_nontroll_data[raw_nontroll_data['country']=='United States of America']
    nontroll_dataset = nontroll_dataset[['tweet']]

    
    nontroll_dataset.columns = ['text']
    nontroll_dataset = nontroll_dataset.dropna()
    nontroll_dataset = nontroll_dataset.iloc[100000:]

    
    test_dataset = Dataset.from_pandas(nontroll_dataset)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=batch_size)
    
    
    
    
    model_path = save_path + r'\FinetunedLLM\checkpoint-10000'
    
    model = (AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)).to(device)
    
    test_trainer = Trainer(model)
    
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    
    y_pred = np.argmax(raw_pred, axis=1)
    
    ## all the outputs should be zero because these were leabelled zero in the training process
    
    accuracy = (len(test_dataset)-np.sum(y_pred))/len(test_dataset)
    
    return accuracy

