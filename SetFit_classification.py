import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
import datetime
import sys 



save_path = r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets'
writer = SummaryWriter(r'runs/setfit')

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
nontroll_train = nontroll_dataset.iloc[:8]            ## using 8 exdamples from each class to train SetFit


troll_train = troll_dataset.iloc[:nontroll_train.shape[0]]
troll_train['label'] = list(np.ones(nontroll_train.shape[0], dtype=int))
nontroll_train['label'] = list(0* np.ones(nontroll_train.shape[0], dtype=int))
train_set = pd.concat([troll_train, nontroll_train], ignore_index=True)
train_set = train_set.sample(frac=1, random_state=666).reset_index(drop=True)       ##shuffling dataset


## using the rest of the datasets for testing
nontroll_test = nontroll_dataset.iloc[nontroll_train.shape[0]:] 
troll_test = troll_dataset.iloc[nontroll_train.shape[0]:]
troll_test['label'] = list(np.ones(troll_test.shape[0], dtype=int))
nontroll_test['label'] = list(0* np.ones(nontroll_test.shape[0], dtype=int))
test_set = pd.concat([troll_test, nontroll_test], ignore_index=True)
test_set = train_set.sample(frac=1, random_state=666).reset_index(drop=True)       ##shuffling dataset


train_set = Dataset.from_pandas(train_set)
test_set = Dataset.from_pandas(test_set)

model_name = "distilbert-base-uncased"
model = ((SetFitModel.from_pretrained(model_name, use_differentiable_head=True, head_params={"out_features": num_labels}))).to(device)



trainer = SetFitTrainer(model=model,
                        train_dataset=train_set,
                        eval_dataset=test_set,
                        loss_class=CosineSimilarityLoss,
                        metric="accuracy",
                        batch_size=batch_size,
                        num_iterations=20, # Number of text pairs to generate for contrastive learning
                        num_epochs=1) # Number of epochs to use for contrastive learning
                        # column_mapping={"text": "text", "label": "label"}) # don't need this line in my case, but I just left this here to remember the syntax



trainer.train()

metrics = trainer.evaluate()

print (metrics)



# model._save_pretrained(save_path + r'\SetFit')





def test_accuracy():
    raw_nontroll_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets\non_troll_candidate_dataset.csv', engine='python')

    raw_nontroll_data = raw_nontroll_data[['tweet', 'country']]

    nontroll_dataset = raw_nontroll_data[raw_nontroll_data['country']=='United States of America']
    nontroll_dataset = nontroll_dataset[['tweet']]

    
    nontroll_dataset.columns = ['text']
    nontroll_dataset = nontroll_dataset.dropna()
    nontroll_dataset = nontroll_dataset.iloc[100000:]

    
    test_dataset = Dataset.from_pandas(nontroll_dataset)

    
    model = SetFitModel.from_pretrained(save_path + r'\SetFit')
    
    
    raw_preds = model(test_dataset)
    
    y_pred = np.argmax(raw_preds, axis=1)
    
    ## all the outputs should be zero because these were leabelled zero in the training process
    
    accuracy = (len(test_dataset)-np.sum(y_pred))/len(test_dataset)
    
    return accuracy

# print (test_accuracy())