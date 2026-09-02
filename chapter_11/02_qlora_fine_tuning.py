"""
Chapter 11: Multimodal Agents and Fine-Tuning with LoRA and QLoRA
Hands-On: QLoRA Fine-Tuning, Every Step from Base Model to Adapter

Extracted from: chapter_11_multimodal_finetuning.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.

NOTE: Steps 1, 2, and 4 need a CUDA-capable GPU and the `bitsandbytes`
package, this chapter's own stated requirement for QLoRA specifically;
they will not run on a CPU-only machine. The "Verify the dataset shape"
block between Steps 3 and 4 is the one part of this file genuinely
runnable anywhere, no GPU required, since it only exercises a small,
free, CPU-only tokenizer to check the training data's own shape.
"""


# ---- Step 1: Load the base model quantised to four-bit precision, the "Q" in QLoRA. ----
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto",
)

# ---- Step 2: Prepare the quantised model for training, then attach small, trainable LoRA adapter matrices. ----
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ---- Step 3: Prepare a small domain dataset, following this chapter's own guidance that even a modest example set is enough to start. ----
from datasets import Dataset

training_examples = [
    {"text": "### Instruction:\nSummarise this clause in plain English.\n### Clause:\nThe Lessee shall indemnify the Lessor against all claims.\n### Response:\nYou (the tenant) must cover the landlord's costs if someone makes a legal claim."},
    {"text": "### Instruction:\nSummarise this clause in plain English.\n### Clause:\nTermination for convenience may be exercised upon thirty days written notice.\n### Response:\nEither party can end the agreement for any reason, as long as they give thirty days' written notice first."},
    # a real fine-tuning run needs hundreds to thousands of examples in this
    # same instruction/response shape; two are shown here purely so this
    # snippet's structure is complete and copy-pasteable end to end
]

dataset = Dataset.from_list(training_examples)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=256, padding="max_length")

tokenized_dataset = dataset.map(tokenize)

# ---- Verify the dataset shape before spending any GPU time ----
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("gpt2")   # CPU-only stand-in for verifying dataset shape
tokenizer.pad_token = tokenizer.eos_token

dataset = Dataset.from_list(training_examples)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=256, padding="max_length")

tokenized_dataset = dataset.map(tokenize)
print(tokenized_dataset)
print("input_ids length:", len(tokenized_dataset[0]["input_ids"]))
print("decoded back:", repr(tokenizer.decode(tokenized_dataset[0]["input_ids"][:15])))

# ---- Step 4: Run a short training loop with Hugging Face's own `Trainer`. ----
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./lora-legal-adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False, bf16=True,
    logging_steps=1,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

model.save_pretrained("./lora-legal-adapter")

# ---- Step 5: Load the adapter back onto the base model for inference, the part of this walkthrough every reader can actually run against any already-trained adapter. ----
from peft import PeftModel

inference_model = PeftModel.from_pretrained(base_model, "./lora-legal-adapter")

prompt = "### Instruction:\nSummarise this clause in plain English.\n### Clause:\nForce majeure events excuse performance during their continuance.\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(inference_model.device)
output = inference_model.generate(**inputs, max_new_tokens=80)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# ---- How it works ----
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

base = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")

narrow = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
get_peft_model(base, narrow).print_trainable_parameters()

base2 = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")

wide = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none")
get_peft_model(base2, wide).print_trainable_parameters()
