from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import sys

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset

def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator

def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          gradient_accumulation_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
        
    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.save_pretrained(output_dir)
        
    training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            gradient_accumulation_steps = gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="adafactor"
        )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
    )
    #train
    trainer.train()
    trainer.save_model()


def main():
    train_file_path = 'C:/Users/afj/Desktop/exerciseAI/train.txt'
    model_name = 'gpt2'
    output_dir = 'C:/Users/afj/Desktop/exerciseAI/Models/mymodel2'
    overwrite_output_dir = True
    per_device_train_batch_size = 2
    num_train_epochs = 3
    save_steps = 50000
    gradient_accumulation_steps = 4

    # Train
    train(
        train_file_path=train_file_path,
        model_name=model_name,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
if __name__ == "__main__":
    main()