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




## setting device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Device in use: {} \n'.format(device))

batch_size = 1000
num_labels = 2

save_path = r'.\Twitter_Troll_Detection'
data_path = r'.\cov19_tweets\COVID.csv'

cov19_data = pd.read_csv(data_path)
cov19_data = cov19_data[['Tweet Posted Time (UTC)', 'Tweet Content', 'Tweet Language']]
cov19_data = cov19_data[cov19_data['Tweet Language']=='English']
cov19_data = cov19_data.drop('Tweet Language', axis=1)

time = cov19_data['Tweet Posted Time (UTC)'].tolist()
cov19_data = cov19_data.drop('Tweet Posted Time (UTC)', axis=1)

cov19_data.columns = ['text']
cov19_data = cov19_data.reset_index(drop=True)



def tokenize(batch, **kwargs):
    if list(kwargs.items())[0][1] == 1:
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2")
        max_length = tokenizer.model_max_length
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_length)
    if list(kwargs.items())[0][1] == 2:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased" )
        return tokenizer(batch['text'], padding=True, truncation=True)
    



## Choice 1 is finetuned, choice 2 is pretrained with trained classifier head
def predict_class(data, model_choice):
    if model_choice == 1:
        data = Dataset.from_pandas(data)
        tokenized_dataset = data.map(tokenize, batched=True, batch_size=batch_size, fn_kwargs={'tokenizer_choice':1})
            
        model_path = save_path + r'\FinetunedLLM\checkpoint-10000'
        model = (AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)).to(device)
        test_trainer = Trainer(model)
        
        raw_pred, _, _ = test_trainer.predict(tokenized_dataset)
        y_pred = np.argmax(raw_pred, axis=1)
        

    if model_choice == 2:
        return
        ## ISSUE: Need to improve importing from the pretrained file before adding that stuff here, make config file with the common functions, and Classifier class
        
        # loaded_model = Classifier()
        # loaded_model.load_state_dict(torch.load(save_path + r'\pretrainedLLM_classifier_weights.pt'))
        # loaded_model = loaded_model.to(device) 
        
        # test_hidden = preprocess_dataset(test_dataset)
        # test_hidden = np.array(test_hidden['hidden_state'])
        # test_hidden_tensor = torch.from_numpy(test_hidden).float().to(device)
         
        # with torch.no_grad():
        #     loaded_model.eval() 
        #     output = loaded_model(test_hidden_tensor)
        
        # y_pred = [1 if i>=0.5 else 0 for i in output]
    
    
    return y_pred
    
    
preds = predict_class(cov19_data, 1)






def datefrmt(date):
    temp = datetime.datetime.strptime(date, '%d %b %Y %X')
    return datetime.datetime.date(temp)


formatted_time = list(map(datefrmt, time))

df = {'time': formatted_time, 'Troll Activity': preds}
df = pd.DataFrame(df)

troll_df = df[df['Troll Activity']==1]
nontroll_df = df[df['Troll Activity']==0]

troll_df = troll_df['time'].tolist()
nontroll_df = nontroll_df['time'].tolist()


plt.xticks(rotation=55)
plt.ylabel('Counts')
plt.hist(x=troll_df, alpha=0.5, label='Trolls')
plt.hist(nontroll_df, alpha=0.5, label='Non-Trolls')
plt.legend(loc='upper right')

