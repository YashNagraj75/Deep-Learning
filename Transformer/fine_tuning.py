
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)

model_name = "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype = torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

index = 200
dialogue = dataset['train'][index]['dialogue']
summary = dataset['train'][index]['summary']

prompt = f"""
Summarize the following conversation:
{dialogue}

Summary:
"""

input_ids = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(input_ids['input_ids'])
print(dialogue)
print(tokenizer.decode(outputs[0]))

def preprocess(example):
    start_prompt = "Summarize the following conversation.\n\n"
    end_prompt = "\n\nSummary"
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
    example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True, return_tensors='pt').input_ids
    example['labels'] = tokenizer(example['summary'], padding="max_length", truncation=True, return_tensors='pt').input_ids
    
    return example
    
tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['id','topic','dialogue','summary',])

print(tokenized_dataset['train'][0])
print(tokenized_dataset)


# # # Full FineTune the model 
# # output_dir = f'/kaggle/working/output/{str(int(time.time()))}'
# # log_dir = f'/kaggle/working/log/{str(int(time.time()))}'
# # training_args = TrainingArguments(
# #     output_dir = output_dir,
# #     learning_rate=3e-5,                  # learning rate
# #     per_device_train_batch_size=8,       # batch size for training
# #     per_device_eval_batch_size=8,        # batch size for evaluation
# #     num_train_epochs=3,                  # number of training epochs
# #     weight_decay=0.01,                   # strength of weight decay
# #     logging_dir=log_dir,                # directory for storing logs
# #     logging_steps=500,                   # log every 500 steps
# #     save_steps=500
# # )

# # trainer = Trainer(
# #     model=model,                         # the instantiated 🤗 Transformers model to be trained
# #     args=training_args,                  # training arguments, defined above
# #     train_dataset=tokenized_dataset['train'],      # training dataset
# #     eval_dataset=tokenized_dataset['validation'],  # evaluation dataset
# # )

# # %% [code] {"execution":{"iopub.status.busy":"2024-06-29T11:03:02.359613Z","iopub.execute_input":"2024-06-29T11:03:02.359900Z","iopub.status.idle":"2024-06-29T11:18:52.528600Z","shell.execute_reply.started":"2024-06-29T11:03:02.359858Z","shell.execute_reply":"2024-06-29T11:18:52.526986Z"}}
# trainer.train()

# Now lets do peft 
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(
    r = 32,
    lora_alpha=32,
    target_modules = ["q","v"],
    lora_dropout=0.05,
    bias="none",
    task_type = TaskType.SEQ_2_SEQ_LM
)

# Add adapter layers to LLM 
peft_model = get_peft_model(model,lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

output_dir = f'./peft/output/{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir = output_dir,
    auto_find_batch_size = True,
    learning_rate = 1e-3,
    num_train_epochs = 3,
    logging_steps = 100,
)

peft_trainer = Trainer(
    model = peft_model,
    args = peft_training_args,
    train_dataset = tokenized_dataset['train']
)



peft_trainer.train()

peft_model_path="/peft-dialogue-summary-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

