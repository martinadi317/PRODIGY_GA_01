# PRODIGY_GA_01
!pip install transformers torch pandas
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import pandas as pd

sample_text = """
Artificial intelligence is transforming industries worldwide.
Machine learning models can generate creative writing based on prompts.
Quantum computing is the future of technology.
Renewable energy sources are critical for sustainability.
The future of AI involves ethics and responsible usage.
Exploring space requires advanced technology and collaboration.
Robotics is reshaping the manufacturing sector.
Advanced algorithms enable personalized recommendations online.
Climate change can be mitigated through technology.
Data science helps businesses make data-driven decisions.
"""

with open("sample_dataset.txt", "w") as f:
    f.write(sample_text.strip())

from transformers import TextDataset, DataCollatorForLanguageModeling

def load_dataset(file_path, tokenizer, block_size=128):
    # Load and preprocess the dataset
    with open(file_path, encoding="utf-8") as f:  # Ensure correct encoding
        text = f.read()
    # Check if the text is shorter than block_size, adjust block_size if necessary
    block_size = min(block_size, len(tokenizer.encode(text))) 
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
        overwrite_cache=True,  # Overwrite cache to reflect changes in text
    )
    print(f"Dataset length: {len(dataset)}")
    return dataset

def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.resize_token_embeddings(len(tokenizer)) 

train_dataset = load_dataset("sample_dataset.txt", tokenizer, block_size=64) # Reduced block_size
data_collator = create_data_collator(tokenizer)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
# Save the fine-tuned model and tokenizer
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

# Function to generate text using the fine-tuned model
def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = inputs.to(model.device)  # This line ensures inputs are on the same device as the model
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=0.7,  # Controls randomness
        top_p=0.9,       # Nucleus sampling
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model with a prompt
prompt = "The future of AI is"
generated_text = generate_text(prompt, model, tokenizer)
print(generated_text)
