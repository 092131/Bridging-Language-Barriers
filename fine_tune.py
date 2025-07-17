import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load dataset
dataset = load_dataset("Aarif1430/english-to-hindi", split="train")

# Split the dataset into train and validation
train_dataset = dataset.train_test_split(test_size=0.2, seed=42)["train"]  # 80% for training
eval_dataset = dataset.train_test_split(test_size=0.2, speft_config = LoraConfig( lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"  # Using CAUSAL_LM for this task
)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    max_seq_length=512,  
    tokenizer=tokenizer,
    args=training_args,  
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    fp16=True,
    logging_steps=10,
    learning_rate=2e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    optim="adamw_torch",
    report_to="none



"
""
""
"

Dhruva sai Neela
19:23
# Load your dataset
dataset = load_dataset("Hemanth-thunder/english-to-malayalam-mt")
print(dataset)
print(dataset["train"][0])
from datasets import load_dataset
You
19:32
Aarif1430/english-to-hindi
Hover over a message to pin it
keep
Dhruva sai Neela
19:41
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load dataset
dataset = load_dataset("Aarif1430/english-to-hindi", split="train")

# Split the dataset into train and validation
train_dataset = dataset.train_test_split(test_size=0.2, seed=42)["train"]  # 80% for training
eval_dataset = dataset.train_test_split(test_size=0.2, s
# Model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 

# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
Dhruva sai Neela
19:42
# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)  # Removed use_auth_token=True
model.resize_token_embeddings(len(tokenizer))


# PEFT configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16, 
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
# Apply PEFT to the model
model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=2,  
    per_device_eval_batch_size=1,  
    num_train_epochs=0.5,  
    bf16=True, 
    logging_steps=10,
    learning_rate=1e-4,  
    evaluation_strategy="epoch",
    save_strategy="epoch",
    optim="adamw_torch",
    report_to="none"
)
# Data formatting function
def formatting_func(example):
    text = f"Translate English to Hindi: {example['english_sentence']} | {example['hindi_sentence']}<eos>"
    # Return as a list even if processing a single example
    return [text]

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    tokenizer=tokenizer,
    args=training_args,
)
# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model()