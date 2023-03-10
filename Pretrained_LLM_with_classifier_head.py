import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import random_split
from transformers import pipeline
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModel
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import datetime
import sys 

## the first part of the file, along with the check function was just for exploring the dataset, when running just call the preprocess_dataset function which acts as a pipeline



save_path = r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets'


## setting device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Device in use: {} \n'.format(device))

torch.manual_seed(666) 
batch_size = 50
n_epochs = 1
learning_rate = 0.00001  
beta1 = 0.5         

## using 'engine = python' for the second file because it's large and runs into error without it
raw_troll_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets\russian_troll_tweets_200k_dataset.csv')
raw_nontroll_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets\non_troll_candidate_dataset.csv', engine='python')


## double square brackets to put in a dataframe instead of single to put in series
troll_dataset = raw_troll_data[['text']]
raw_nontroll_data = raw_nontroll_data[['tweet', 'country']]



## removing the tweets not from US, then setting the sizes to be equal, and creating the target column, 0 for non-troll, 1 for troll, and droping empty columns
nontroll_dataset = raw_nontroll_data[raw_nontroll_data['country']=='United States of America']
nontroll_dataset = nontroll_dataset[['tweet']]

nontroll_dataset.columns = ['text']
nontroll_dataset = nontroll_dataset.dropna()
troll_dataset = troll_dataset.dropna()
nontroll_dataset = nontroll_dataset.iloc[:100000]    ## decreasing size a bit so I can deal with it on this PC

troll_dataset = troll_dataset.iloc[:nontroll_dataset.shape[0]]
troll_dataset['target'] = list(np.ones(nontroll_dataset.shape[0], dtype=int))
nontroll_dataset['target'] = list(0* np.ones(nontroll_dataset.shape[0], dtype=int))
train_dataset = pd.concat([troll_dataset, nontroll_dataset], ignore_index=True)
train_dataset = train_dataset.sample(frac=1, random_state=666).reset_index(drop=True)       ##shuffling dataset


## to check if any of the text sizes exceeds the maximum context size of the model we choose (which we can't have), notice we're only looking at length in words here, but model context size is in tokens, so if it's close we need to be more careful (redo these plots after tokenizing), if it's much less than it's fine, as is the case here for the disaster tweet dataset
def plot_texts_lengths(train_dataset):
    train_dataset = Dataset.from_pandas(train_dataset)                      
    train_dataset.set_format(type='pandas')                                 
    train_dataframe = train_dataset[:]
    train_dataframe['words per text (per category)'] = train_dataframe['text'].str.split().apply(len)
    train_dataframe.boxplot('words per text (per category)', by='target', grid=False, showfliers=False, color='black')
    plt.suptitle("")
    plt.xlabel("")
    plt.show()


## note that the tokenizer used needs to match the one used for the imported pretrained model (when using hugging face pipeline it takes care of this step)
model_name = "distilbert-base-uncased"               ## max context size 512 tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)

## importing the pretrained model
transformer = AutoModel.from_pretrained(model_name).to(device)


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)          ##padding fills to match the largest text size in the batch, truncation truncates anything longer than context size (which as we've checked we don't have any)





## checking if forward pass of imported model works 
def check_model():
    transformer = AutoModel.from_pretrained(model_name).to(device)
    text = "test string"
    text_tensor = tokenizer.encode(text, return_tensors="pt").to(device)                                    ## shape = (batch_size, n_tokens), encode() method only applies the tokenizer to get the input_ids, not the attention masks as well
    print ('input text tensor size {} \n'.format(text_tensor.shape))
    
    with torch.no_grad():
        outputs = transformer(text_tensor)
    print ('output hidden state tensor size {} \n'.format(outputs.last_hidden_state.shape))                 ## shape = (batch_size, n_tokens, hidden_dim), so for each token returns a tensor of shape (batch_size, hidden_dim)
    
    print(outputs)
    # print (outputs.last_hidden_state[:,0])
    
    ## remove from GPU memory
    del transformer
    



## fucntion to extract hidden states to be applied to whole dataset
def extract_hidden_state(batch, transformer=transformer):
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).to(device)
    
    with torch.no_grad():
        last_hidden_state = transformer(input_ids, attention_mask).last_hidden_state[:,0]
        last_hidden_state = last_hidden_state.cpu().numpy()                         ## needs to be on cpu to use numpy, so we can use map() method
    
        
    ## free up memory, and I double checked it didn't affect the output hidden states, but does it actually free if I delete inside function?
    del input_ids
    del attention_mask
    
    return {'hidden_state': last_hidden_state}
       



def preprocess_dataset(dataset):
    arrow_dataset = Dataset.from_pandas(dataset)
    print ('---ENCODING---')
    arrow_dataset= arrow_dataset.map(tokenize, batched=True, batch_size=batch_size)                                  ## tokenizing the dataset, bacth_size = None applies it to the dataset as a whole (as a single batch), if not set to None, then needs to match the training batch size
    
    arrow_dataset.set_format(type='torch')
    print ('---PASSING THROUGH MODEL---')
    dataset_hidden = arrow_dataset.map(extract_hidden_state, batched=True, batch_size=batch_size)              ## the training dataset with text, target, input, ids, atternion_mask, and hidden_states output by the model, the loading this does when run shows it going through the dataset once (for e.g. train dataset len is 7612 examples, if batchsize = 128, then thats ~60 batches, so you'll see it loading 0/60), if no batch size set, default size is 1000 (None is 1 batch, uses whole dataset)

    return dataset_hidden


# train_dataset_hidden = preprocess_dataset(train_dataset)



## project the classes down to 2D (using UMAP, see paper) to visualise separation, note if separation not visible by eye, still doesn't mean they're not separable for sure
def dimReduction(train_dataset_hidden):
    X_train = np.array(train_dataset_hidden['hidden_state'])
    y_train = np.array(train_dataset_hidden['target'])

    ## UMAP works best with features scaled to [0,1]
    X_scaled = MinMaxScaler().fit_transform(X_train)
    
    mapper = UMAP(n_components=2, metric='cosine').fit(X_scaled)
    
    reduced_df = pd.DataFrame(mapper.embedding_, columns=['X','Y'])
    reduced_df['target'] = y_train
    print (reduced_df.head())
    
    fig, axes = plt.subplots(1, 2, figsize=(5,3))
    axes = axes.flatten()
    cmaps = ["Reds", "Blues"]
    labels = ['Troll', 'Non-Troll']
    
    
    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        reduced_df_sub = reduced_df.query(f"target == {i}")
        axes[i].hexbin(reduced_df_sub["X"], reduced_df_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
    plt.tight_layout()
    plt.show()
    

# dimReduction(train_dataset_hidden)


def prepare_torch_trainset(train_dataset_hidden):
    ## create the arrays that will be transformed into torch tensors (from the HF Dataset)
    train_hidden = np.array(train_dataset_hidden['hidden_state'])
    labels = np.array(train_dataset_hidden['target'])
    
    
    hidden_states_tensor = torch.from_numpy(train_hidden).float()
    # hidden_states_tensor = hidden_states_tensor.view(7613, 84*768)              ## need this only if using the full hidden state, if only CLS token hidden state (or avg of hidden states), then don't need this line, becasue alread in the appropriate shape of batch size and the number of input to the classifier network
    hidden_states_tensor.requires_grad = True
    
    labels = torch.from_numpy(labels).float()
    
    
    classifier_train_dataset = TensorDataset(hidden_states_tensor, labels)
    
    ## splitting into train and validation datasets
    train_size = int(0.8 * len(classifier_train_dataset))
    val_size = len(classifier_train_dataset) - train_size
    train_dataset, val_dataset = random_split(classifier_train_dataset, [train_size, val_size])           
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size) 
    return train_loader, val_loader


# train_loader, val_loader = prepare_torch_trainset(train_dataset_hidden)

## remove from GPU memory
del transformer


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__() 

        self.lin1 = torch.nn.Linear(in_features=768, out_features=100, bias=True)
        self.lin2 = torch.nn.Linear(in_features=100, out_features=1)




    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))             ## outputs probability of belonging to the y=1 class
        return x




def make_train_step(model, loss_func, optimizer):
    
    def train_step(x, y):
        model.train()           ## put model in train mode
        y_hat = model(x)
        y_hat = y_hat.squeeze()        ## model outputs y_hat shape: (batch_size,1), squeeze gets rid of the one, to match shape of y: (batch_size)
        
        loss = loss_func(input=y_hat, target=y)
        loss.backward()         ## calculate gradients
        optimizer.step()        ## update parameters
        optimizer.zero_grad()   ## zero gradients  (because they accumulate)
        
        return loss.item()
    return train_step 




model = Classifier()  
model.to(device)  
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(beta1, 0.999)) 

train_step = make_train_step(model, loss_func, optimizer)


## defning the training loop
losses = []
val_losses = []
val_accuracy = []


def start_train_loop():
    print('start time is: {} \n'.format(datetime.datetime.now()))
    for epoch in range(n_epochs):
        
        
        for counter, data in enumerate(train_loader, start=0):
            
            x_batch = data[0].to(device)
            y_batch = data[1].to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
    

            with torch.no_grad():                       ## turn off gradient computation
            
                correct = 0
                total = 0
                val_loss = 0
                
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    
                    model.eval()                        ## put model in evaluation mode
                    
    
                    y_hat = model(x_val)
                    y_hat = y_hat.squeeze()
                    val_loss += loss_func(input=y_hat, target=y_val)
                    
                    
                    predicted = [1 if i>=0.5 else 0 for i in y_hat]
                    predicted = torch.FloatTensor(predicted)
                    predicted = predicted.to(device)
                    total += len(y_val)
                    correct += (predicted == y_val).sum().item()
                    
                    
                accuracy = 100 * correct / total
                val_loss = val_loss / total
                    
                val_accuracy.append(accuracy)
                val_losses.append(val_loss.item())
                
                print ('[epoch: {}][batch: {}]  train_loss: {}, val_accuracy: {}, avg_val_loss: {} \n'.format(epoch+1, counter+1, loss, accuracy, val_loss))
                if (counter+1)%5 == 0:
                    torch.save(model.state_dict(), save_path + r'\temp_files\Classifier_checkpoint_weights_epoch{}_batch{}_accuracy{}.pt'.format(epoch+1, counter+1, accuracy))
                    print ('CHECKPOINT \n')
                    
    torch.save(model.state_dict(), save_path + r'\pretrainedLLM_classifier_weights.pt')
    print('model is saved \n')
    print('Finsih time is: {} \n'.format(datetime.datetime.now()))
    
   
    
    
def plot_losses():
    plt.plot(losses, label = 'train_loss')
    plt.legend(loc='upper left', frameon=True, edgecolor='k')
    plt.show()
    

    plt.plot(val_losses, label = 'validation_loss')
    plt.legend(loc='upper left', frameon=True, edgecolor='k')
    plt.show()
    
        
    plt.plot(val_accuracy, label = 'validation accuracy')
    plt.legend(loc='upper left', frameon=True, edgecolor='k')
    plt.show()
    

## call training loop
# start_train_loop()

## loss plots 
# plot_losses()

    
def classifier_eval():    
    transformer = AutoModel.from_pretrained(model_name).to(device)

    loaded_model = Classifier()
    loaded_model.load_state_dict(torch.load(save_path + r'\pretrainedLLM_classifier_weights.pt'))
    loaded_model = loaded_model.to(device) 
    
    test_hidden = preprocess_dataset(test_dataset)
    test_hidden = np.array(test_hidden['hidden_state'])
    test_hidden_tensor = torch.from_numpy(test_hidden).float().to(device)
    
    with torch.no_grad():
        loaded_model.eval() 
        output = loaded_model(test_hidden_tensor)
    
    preds = [1 if i>=0.5 else 0 for i in output]
    
    return preds



# predictions = classifier_eval()

# raw_test_data['target'] = predictions
# raw_test_data = raw_test_data.drop(['keyword', 'location', 'text'], axis=1)
# raw_test_data = raw_test_data.set_index('id')


# raw_test_data.to_csv(save_path + r'\classifier_submission.csv')
# print ('done')




def test_accuracy():
    raw_troll_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets\russian_troll_tweets_200k_dataset.csv')
    raw_nontroll_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\russian_troll_tweets\non_troll_candidate_dataset.csv', engine='python')

    troll_dataset = raw_troll_data[['text']]
    raw_nontroll_data = raw_nontroll_data[['tweet', 'country']]

    nontroll_dataset = raw_nontroll_data[raw_nontroll_data['country']=='United States of America']
    nontroll_dataset = nontroll_dataset[['tweet']]

    
    nontroll_dataset.columns = ['text']
    nontroll_dataset = nontroll_dataset.dropna()
    test_dataset = nontroll_dataset.iloc[100000:]
    
    
    loaded_model = Classifier()
    loaded_model.load_state_dict(torch.load(save_path + r'\pretrainedLLM_classifier_weights.pt'))
    loaded_model = loaded_model.to(device) 
    
    test_hidden = preprocess_dataset(test_dataset)
    test_hidden = np.array(test_hidden['hidden_state'])
    test_hidden_tensor = torch.from_numpy(test_hidden).float().to(device)
     
    with torch.no_grad():
        loaded_model.eval() 
        output = loaded_model(test_hidden_tensor)
    
    y_pred = [1 if i>=0.5 else 0 for i in output]
    
    print (y_pred)
    
    accuracy = (len(test_dataset)-np.sum(y_pred))/len(test_dataset)
    
    return accuracy


