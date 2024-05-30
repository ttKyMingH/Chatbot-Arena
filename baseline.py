import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import re
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

os.system('python -m pip install --no-index --find-links=../input/lmsys-pip-wheels transformers')
os.system('python -m pip install --no-index --find-links=../input/lmsys-pip-wheels tokenizers')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import log_loss
from tokenizers import AddedToken
warnings.simplefilter('ignore')

# ====================================================
# Directory settings
# ====================================================
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

train = pd.read_csv('lmsys-chatbot-arena/train.csv')
test = pd.read_csv('lmsys-chatbot-arena/test.csv')
submission = pd.read_csv('lmsys-chatbot-arena/sample_submission.csv')

class CFG:
    n_splits = 5
    seed = 42
    max_length = 1539 # 512 x 3 + a
    lr = 1e-5
    train_batch_size = 8
    eval_batch_size = 4
    train_epochs = 4
    weight_decay = 0.01
    warmup_ratio = 0.1
    num_labels = 3
    debug=True
    model = "microsoft/deberta-v3-xsmall"
    target_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']

# https://www.kaggle.com/code/piantic/train-lmsys-deberta-v3-starter-code/notebook

def add_label(df):
    labels = np.zeros(len(df), dtype=np.int32)
    labels[df['winner_model_a'] == 1] = 0
    labels[df['winner_model_b'] == 1] = 1
    labels[df['winner_tie'] == 1] = 2
    df['label'] = labels
    return df


def add_stats(df):
    # Some stats
    df["prompt_words"] = df["prompt"].apply(lambda x: x.replace('\n', ' ').split(" "))
    df["total_prompt_words"] = df["prompt"].apply(lambda x: len(x.split(" ")))
    df["prompt_length"] = df["prompt"].apply(lambda x: len(x))

    df["response_a_words"] = df["response_a"].apply(lambda x: x.replace('\n', ' ').split(" "))
    df["total_response_a_words"] = df["response_a"].apply(lambda x: len(x.split(" ")))
    df["response_a_length"] = df["response_a"].apply(lambda x: len(x))

    df["response_b_words"] = df["response_b"].apply(lambda x: x.replace('\n', ' ').split(" "))
    df["total_response_b_words"] = df["response_b"].apply(lambda x: len(x.split(" ")))
    df["response_b_length"] = df["response_b"].apply(lambda x: len(x))
    
    return df

def truncate_text(df, column_name, max_length=512):
    df[f"{column_name}"] = df[column_name].str[:max_length]
    return df

train = add_label(train)
train = add_stats(train)
train = truncate_text(train, 'prompt')
train = truncate_text(train, 'response_a')
train = truncate_text(train, 'response_b')

class Tokenize(object):
    def __init__(self, train, valid):
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.model)
        self.train = train
        self.valid = valid
        
    def get_dataset(self, df):
        ds = Dataset.from_dict({
                'id': [e for e in df['id']],
                'prompt': [ft for ft in df['prompt']],
                'response_a': [ft for ft in df['response_a']],
                'response_b': [ft for ft in df['response_b']],
                'label': [s for s in df['label']],
            })
        return ds
    
    def tokenize_function(self, df):
        tokenized_inputs = self.tokenizer(
            df['prompt'], df['response_a'], df['response_b'],
            truncation=True, padding=True, max_length=CFG.max_length
        )
        return tokenized_inputs
    
    def __call__(self):
        train_ds = self.get_dataset(train)
        valid_ds = self.get_dataset(valid)
        
        tokenized_train = train_ds.map(
            self.tokenize_function, batched=True
        )
        tokenized_valid = valid_ds.map(
            self.tokenize_function, batched=True
        )
        
        return tokenized_train, tokenized_valid, self.tokenizer
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    score = log_loss(labels, predictions)
    results = {
        'score': score
    }
    return results

data = train.copy()
data["label"] = data["label"].astype('int32')
skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
for i, (_, val_index) in enumerate(skf.split(data, data["label"])):
    data.loc[val_index, "fold"] = i
data.head(5)

# if CFG.debug:
#     display(data.groupby('fold').size())
#     data = data.sample(n=1000, random_state=0).reset_index(drop=True)
#     display(data.groupby('fold').size())

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=True,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    num_train_epochs=CFG.train_epochs,
    weight_decay=CFG.weight_decay,
    evaluation_strategy='epoch',
    metric_for_best_model='score',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to='none',
    warmup_ratio=CFG.warmup_ratio,
    optim='adamw_torch'
)

for fold in range(len(data['fold'].unique())):
    train = data[data['fold'] != fold]
    valid = data[data['fold'] == fold]
    
    # ADD NEW TOKENS for ("\n") new paragraph and (" "*2) double space
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.add_tokens([AddedToken("\n", normalized=False)])
    tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])
    tokenize = Tokenize(train, valid)
    tokenized_train, tokenized_valid, tokenizer = tokenize()
    
    model = AutoModelForSequenceClassification.from_pretrained(CFG.model, num_labels=CFG.num_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    y_true = valid['label'].values
    predictions = trainer.predict(tokenized_valid).predictions
    predictions = predictions.argmax(axis=1)# + 1
    cm = confusion_matrix(y_true, predictions, labels=[x for x in range(0,3)])
    draw_cm = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[x for x in range(0,3)])
    draw_cm.plot()
    plt.show()
    
    trainer.save_model(f'deberta-v3-xsmall_fold_{fold}')
    tokenizer.save_pretrained(f'deberta-v3-xsmall_fold_{fold}')
    
    valid.to_csv(f'valid_df_fold_{fold}.csv', index=False)
    break # just fold 0 train for test