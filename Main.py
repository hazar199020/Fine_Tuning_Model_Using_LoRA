#PEFT is parameter efficient fine tuning (Option for Paramteter training)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer,AutoConfig,AutoModelForSequenceClassification, DataCollatorWithPadding,TrainingArguments,Trainer

# choose base Model it has only 67M params
model_checkpoint = 'distilbert-base-uncased'

# create tokenizer to convert Text to numerical form
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# create tokenize function : How to convert text (DB) to numbers
def tokenize_function(examples):
    # extract text
    text = examples["text"]
    #tokenize and truncate text so all input has same length
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        #return numpy tensors
        return_tensors="np",
        #use truncate
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# define an evaluation function to pass into trainer later
# #p is model outcome
def compute_metrics(p):
    # import accuracy evaluation metric
    accuracy = evaluate.load("accuracy")
    predictions, labels = p
    # convert logits to label
    predictions = np.argmax(predictions, axis=1)
    #compare model outcome with the actual label
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

def main():

   # load dataset
   dataset = load_dataset('shawhin/imdb-truncated')

   #choose base Model it has only 67M params
   model_checkpoint = 'distilbert-base-uncased'

   # define label maps
   id2label = {0: "Negative", 1: "Positive"}
   label2id = {"Negative":0, "Positive":1}

   # generate classification model from model_checkpoint
   model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

   # create tokenizer to convert Text to numerical form
   tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

   # add pad token if none exists so all input have the same length
   if tokenizer.pad_token is None:
      tokenizer.add_special_tokens({'pad_token': '[PAD]'})
      model.resize_token_embeddings(len(tokenizer))

   # tokenize training and validation datasets
   tokenized_dataset = dataset.map(tokenize_function, batched=True)

   # create data collator
   #pad the shorter sequence in one batch to match the longest sequence
   #it is alot computationally efficient
   data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

   #apply text examples on untrained Model
   #define list of examples
   text_list = ["It was good.", "Not a fan, don't recommened.", "Better than the first one.", "This is not worth watching even once.", "This one is a pass."]

   print("Untrained model predictions:")
   print("----------------------------")
   for text in text_list:
     # tokenize text
     inputs = tokenizer.encode(text, return_tensors="pt")
     # compute logits
     logits = model(inputs).logits
     # convert logits to label
     predictions = torch.argmax(logits)
     print(text + " - " + id2label[predictions.tolist()])

   #Lora Adaption (Low Rank Adaption) which is augment the model with additional trainable  - Fewer than Actual Model parameters
   #Task is sequence classification
   peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        #like learning rate
                        lora_alpha=32,
                        #probabilty of dropout
                        lora_dropout=0.01,
                        #which module we want to apply LoRa to  (Query Layers)
                        #Query Layers : Control how each token “asks” for information from others.
                        target_modules = ['q_lin'])

   #add the new parameters to the model
   model = get_peft_model(model, peft_config)
   model.print_trainable_parameters()

   # hyperparameters
   #learning rate
   lr = 1e-3
   batch_size = 4
   num_epochs = 3
   # define training arguments
   training_args = TrainingArguments(
    #where to save the model
    output_dir= model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    #every epoch will complete the evalution startegy
    #evaluation_strategy="epoch",
    #save_strategy="epoch",
    #return best version of the model
    #load_best_model_at_end=True,
    report_to=[]
     )

   #creater trainer object
   trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_dataset["train"],
     eval_dataset=tokenized_dataset["validation"],
     tokenizer=tokenizer,
     data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
     compute_metrics=compute_metrics)

   # train model
   trainer.train()

   print("Trained model predictions:")
   print("--------------------------")
   for text in text_list:
     #When feeding a model → use PyTorch tensors
     #Because the model is PyTorch.
     inputs = (tokenizer.encode(text, return_tensors="pt"))
     logits = model(inputs).logits
     predictions = torch.max(logits,1).indices
     print(text + " - " + id2label[predictions.tolist()[0]])

if __name__ == '__main__':
    main()