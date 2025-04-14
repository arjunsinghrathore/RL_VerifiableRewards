from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import re
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig, GRPOTrainer
import time
from huggingface_hub import login
import wandb
from datetime import datetime

# Huggingface Login
login(token="YOUR_KEY")

# Wandb Login 
wandb.login(key='YOUR_KEY')
run = wandb.init(
    # Set the project where this run will be logged
    project="rl_course_proj",
)

#####################################################
# LOAD DATASET
dataset_id = 'AI-MO/NuminaMath-TIR'
train_dataset, test_dataset = load_dataset(dataset_id, split=['train[:5%]', 'test[:5%]'])

#####################################################
# STRUCTURING THE DATASET
print("\nStructure of the dataset: ",train_dataset)

# One sample from the dataset
print("\nOne sample from the dataset: ",train_dataset[0])

# We will be following the DeepseekR1 way for prompting the model to reason and follow a structure in its outputs <think></think> and <answer></answer>
# So for this we will modify the dataset accordingly to add this additional system prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

# We can now remove the problem and the messages column as we dont need them
train_dataset = train_dataset.remove_columns(['messages', 'problem'])
print("\nUpdated Structure of the dataset: ",train_dataset)

# Now lets take a look at one of the examples prompt field
print("\nTrain Dataset Prompt Field: ",train_dataset[0]['prompt'])

#####################################################
# LOADING THE MODEL
model_id = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda", #"auto",
)

# Configuring LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

print(f"\n{model_id} Trainabale Parameters --> ",model.print_trainable_parameters())

#####################################################
# SETTING THE REWARD FUNCTIONS

# Format Reward
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

# Accuracy Reward
def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

#####################################################
# SETTING THE GRPO TRAINING HYPER-PARAMETERS
training_args = GRPOConfig(
    output_dir="/scratch/ax2119/rl_course/project/baseline_checkpoints/",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=10,
    bf16=True,
    do_eval=True,
    run_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}",

    # Parameters that control de data preprocessing (Need to understand these better)
    max_completion_length=256, # default: 256
    num_generations=6, # default: 8
    max_prompt_length=256, # default: 512

    # Parameters related to reporting and saving (Update this to use W&B)
    report_to="wandb",
    logging_steps=10,
    push_to_hub=False,

    # Checkpointing
    save_strategy="steps",
    save_steps=100, # Consider increasing this if saving every 10 steps is too frequent
    save_total_limit=5, # Keep only the top 3 checkpoints
    load_best_model_at_end=True, # Load the best model when training finishes
    metric_for_best_model="reward", # Or "eval_reward", "loss", "eval_loss" etc. Needs to match a logged metric.
    greater_is_better=True, # True for reward/accuracy, False for loss
)

#####################################################
# TRAINING THE MODEL
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

#####################################################
#
