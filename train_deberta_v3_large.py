import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from typing import Optional, Union
import pandas as pd, numpy as np, torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

VER=5
# TRAIN WITH SUBSET OF 60K
NUM_TRAIN_SAMPLES = 1_024
USE_PEFT = False
FREEZE_LAYERS = 0
FREEZE_EMBEDDINGS = False
MAX_INPUT = 512


#数据路径
TRAIN_PATH = '/kaggle/input/gte-base-context/gte-base-public-13-context-322538.csv'
VALID_PATH = '/kaggle/input/gte-base-context/gte-base-valid-context-970.csv'
MODEL = 'microsoft/deberta-v3-large'

df_valid = pd.read_csv(VALID_PATH)
print('Validation data size:', df_valid.shape )
df_valid.head()

df_train = pd.read_csv(TRAIN_PATH)
df_train = df_train.drop(columns="source")
df_train = df_train.fillna('').sample(frac=1.0, random_state=10).reset_index(drop=True)
#df_train = df_train[df_train['A'].map(len)<250].reset_index(drop=True)
print('Train data size:', df_train.shape )
df_train.head()

option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k,v in option_to_index.items()}

def preprocess(example):
    first_sentence = [ "[CLS] " + example['context'] ] * 5
    second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first', 
                                  max_length=MAX_INPUT, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]
    
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset_valid = Dataset.from_pandas(df_valid)
dataset = Dataset.from_pandas(df_train)
dataset = dataset.remove_columns(["__index_level_0__"])
dataset

tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
tokenized_dataset = dataset.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
tokenized_dataset

model = AutoModelForMultipleChoice.from_pretrained(MODEL)

def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}

training_args = TrainingArguments(
    warmup_ratio=0.1, 
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    report_to='none',
    output_dir = f'./checkpoints_{VER}',
    overwrite_output_dir=True,
    fp16=True,
    gradient_accumulation_steps=8,
    logging_steps=250,
    evaluation_strategy='steps',
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    load_best_model_at_end=True,
    metric_for_best_model='map@3',
    lr_scheduler_type='cosine',
    weight_decay=0.012,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_valid,
    compute_metrics = compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
)

trainer.train()
trainer.save_model(f'model_v{VER}')
