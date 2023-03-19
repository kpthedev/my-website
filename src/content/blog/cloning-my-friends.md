---
title: "Cloning my friends with Transformers"
description: "Using Hugging Face Transformers to fine-tune a GPT model on my friends and me."
pubDate: "Mar 19 2023"
---

As the resident AI-obsessed programmer in my friend group, it is my duty to explore using machine learning models to impress/scare my friends. I've used [Dreambooth](https://dreambooth.github.io/) to generate images of us and cloned our voices with [Tortoise-TTS](https://github.com/neonbjb/tortoise-tts), which ultimately led us to a new goal--completely replacing ourselves with AI clones. In this post, I'll explain the first step in my quest to clone my friends with generative AI. Using Hugging Face's *Transformers*, I've fine-tuned a GPT model on thousands of messages my friends and I have sent in our various group chats, and can now generate terrifyingly lifelike conversations between us.

## Training data

Getting the training data--the group chat messages--differs between messaging apps, so your mileage may vary. Some apps, like WhatsApp and Telegram, have very easy ways of exporting the chat history as a `.txt` or even a `.json` file. However, other apps, like iMessage, Discord, and Signal, require some tinkering to export messages. Once you get the messages, you'll need to organize them all in one large text file with the following format:

```
SPEAKER1:
A message from the first speaker.

SPEAKER2:
A message sent from the second speaker.
```

This is the format I chose for the training data, mainly for simplicity. Notice that the speaker names are all capitalized and followed by a colon, and then the next line is the message. Since each messaging app exports to different formats, I'll leave the data cleaning for you to implement in your favorite scripting language.

If you don't have a specific dataset, then you can follow along using Andrej Karpathy's [tinyshakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). It concatenates parts of Shakespeare's plays in the same format I specified above, and training with it is quick since it's small.

## Tools for training on a single GPU

```bash
pip install torch transformers datasets bitsandbytes
```

The workhorse of this project is Hugging Face [Transformers](https://huggingface.co/docs/transformers/index), an amazing Python library that provides several APIs for downloading, training, and using state-of-the-art transformer models. We'll be using the 774 million parameter GPT-2 model hosted on Hugging Face's [models hub](https://huggingface.co/gpt2-large) for fine-tuning. *Transformers* will take care of downloading the weights and configs of the model, and we'll use another Hugging Face library, [datasets](https://huggingface.co/docs/datasets/index), for loading and processing our training data.

If you've ever trained or fine-tuned large deep learning models, you'll know that one of the biggest limitations is how much memory you have available on your graphics card. All the code in this blog post fits into my *Nvidia A4000 16GB* graphics card using some tricks I'll talk about that cut down on memory usage. If you don't have a 16GB graphics card, you'll probably have to train on a smaller model or reduce the batch/block size. For a small model to train, you could look into Hugging Face's [DistilGPT2](https://huggingface.co/distilgpt2)--just replace every mention of `gpt2-large` in this post with `distilgpt2` and you're set (the output quality won't be as good though).

Another memory optimization we'll be using is from the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library. Specifically, we'll be using an 8-bit optimizer to calculate gradients. This means that instead of doing math with floating-point numbers represented with 32 or 16 bits of precision, they'll be cut down to 8 bits, which greatly reduces the amount of memory usage on the graphics card. We'll make use of this and similar techniques later when training the model.

## Writing the training script

Now with the prerequisites ready, we can start implementing our training script, `train.py`. The full code for everything in this post can be found in this [Github repo](https://github.com/kpthedev/cloning-my-friends).

Before we go any further, we'll need to import the following:

```python
import math
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
```

### Preparing datasets

We'll start with preprocessing our data. The basic operations will be: splitting into training/validation sets, tokenizing, and then chunking into fixed blocks.

For the first step, we use the datasets library's `load_data()` function to read the data. We specify that it's a text file, split the first 5% for the validation set, and the remaining 95% for training. The validation set will let us calculate the loss of our model so we can see how well it's learning. We also keep line breaks since they're critical to the structure of the input file. 

```python
training_input_file = "input.txt"

# Load and split dataset
data = load_dataset(
    "text",
    data_files=training_input_file,
    split=["train[5%:]", "train[:5%]"],
    cache_dir="./cache",
    keep_linebreaks=True,
)
train_data, eval_data = data[0], data[1]
```
The next step is to tokenize our data, which means converting the English words into a vocabulary of numbers that the model can understand. Luckily, that vocabulary is already prepared for us and we can use the `from_pretrained()` function to pull it from the Hugging Face hub. The tokenizer we'll be using is the `GPT2Tokenizer` which was used to train the `gpt2-large` model that we'll be fine-tuning. So let's write a function that we can use to tokenize our training and validation datasets.

```python
def tokenize_data(data, tokenizer):
    """Tokenizes the input data using the specified tokenizer."""
    tokenized_data = data.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        remove_columns=data.column_names,
    )
    return tokenized_data
```

Then we'll initialize the tokenizer and run each of our datasets through the function:

```python
pretrained_model = "gpt2-large"

# Tokenize the data
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
train_data = tokenize_data(train_data, tokenizer)
eval_data = tokenize_data(eval_data, tokenizer)
```

The final step for preprocessing the data is grouping it into properly sized chunks. The GPT-2 model we're using has a maximum context size of 1024, meaning it can process blocks of tokens that are 1024 tokens long. For this reason, we need to chunk our data into blocks of 1024 to feed into the model while training. If your graphics card is running out of memory during training, you can decrease this to 512 or even 256 (you'll have to generate smaller sequences as a consequence).

```python
def group_text_blocks(data, block_size):
    """Groups the text in the dataset into blocks of length <= block_size."""

    def group_texts(example):
        combined = {
            k: [item for sublist in v for item in sublist] for k, v in example.items()
        }
        combined_length = len(list(combined.values())[0])

        if combined_length >= block_size:
            combined_length = (combined_length // block_size) * block_size

        result = {}
        for k, t in combined.items():
            result[k] = [
                t[i : i + block_size] for i in range(0, combined_length, block_size)
            ]
        result["labels"] = result["input_ids"].copy()
        return result

    return data.map(group_texts, batched=True)
```

The grouping function is a bit convoluted, but essentially we're taking the input data, which is a list of lists of tokens, and grouping them into one combined list with a length that's a multiple of the block size--some excess will be trimmed if the sequence is too long (note that GPT-2 doesn't have padding tokens to solve this issue, but other large language models do). Let's group our datasets so we can move on to training.

```python
# Group the data
block_size = 1024
train_data = group_text_blocks(train_data, block_size)
eval_data = group_text_blocks(eval_data, block_size)
```

### Picking a pre-trained model

In this post, I'll be fine-tuning the `gpt2-large` model. It's a 774 million parameter pre-trained model available on the model hub and it's the biggest model that I could fit in my 16GB graphics card. Some graphics cards have more memory, letting you train larger models like `gpt-xl`. If you have less memory, you could try using `gpt2-medium` or the much smaller `distilgpt2`, but keep in mind that doing so will reduce the quality of outputs. You could also try fine-tuning other large language models like EleutherAI's GPT-neo or Facebook's OPT models. The beauty of using the *Transformers* library is easy access to Hugging Face's model hub that hosts all these pre-trained models. The code in this post could easily be adapted to use these models by simply changing the `gpt2-large` string to the name of another model.

### Setting training arguments

To start training, we'll need to set our `TrainingArguments`. There are a lot of important parts here, and I'll explain them after showing you the code.

```python
training_args = TrainingArguments(
        output_dir=f"./models/finetuned-gpt2-large",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        optim="adamw_bnb_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
    )
```

Here is a quick rundown of the more standard parameters:

* `output_dir` - This is the output directory of the final pre-trained model. Change this as you please.
* `num_train_epochs` - This is the total number of epochs that the model will train for. While more epochs usually yields better results, each epoch adds to the training time.
* `per_device_train_batch_size` - This parameter controls how many chunks of data the model sees per batch. This *greatly* influences the memory usage on your graphics card, so you might have to play around with this number to avoid out-of-memory errors.

### Optimizations

These parameters are used specifically for cutting down our graphics card memory requirements:

* `optim` - This is the optimizer that lets our model actually learn. Notice that it uses the *bitsandbytes* library's more memory-efficient 8-bit optimizer. You could change this to the default `"adamw_torch"`, but this will greatly increase the memory usage on the graphics card.
* `fp16`/`bf16` - This is the precision of the model, which we reduce from 32 bits to 16 bits to save memory. If you have an Nvidia graphics card that is from the [Ampere](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)) generation or newer, use `bf16`. Otherwise, change it to `fp16`.
* `tf32` - This is another precision reducing optimization. This is **only** available for Ampere generation and newer Nvidia graphics cards.
* `gradient_checkpointing` - This is a major memory saver that pushes calculation of layer activations to the backward pass.
* `gradient_accumulation_steps` - The batch size we set earlier is very limited by the graphics card's memory, but this allows us to accumulate batches of gradients without updating any variables. Effectively, it increases our batch size with no increase in memory usage (the final *logical* batch size will be the batch size multiplied by the gradient accumulation steps. So, in this specific example, it would be `4 * 4 = 16`).

### Fine-tuning the GPT model

Equipped with our datasets, tokenizer, and training arguments, we can start the training. We'll define the following function to do so:

```python
def train_model(training_args, train_data, eval_data, tokenizer, model_name_or_path):
    """Trains a causal language model using the specified training and evaluation data."""
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer
```

Here, we load the configuration and then the pre-trained weights of the `gpt2-large` model using the ubiquitous `from_pretrained()` function. After that, it's just a matter of giving all the components to the function to actually perform the fine-tuning.


```python
# Set seed
set_seed(42)

# Train the model
trainer = train_model(
    training_args,
    train_data,
    eval_data,
    tokenizer,
    pretrained_model,
)

# Save the model
trainer.save_model()

# Evaluate the model
eval_loss = trainer.evaluate()["eval_loss"]
perplexity = math.exp(eval_loss)
print(f"Perplexity: {perplexity:.2f}")
```

It's finally time to save and run the training script. This might take a while depending on the size of your dataset, the parameters you pick, and the speed of your graphics card. On my A4000 graphics card, the `tinyshakespeare` dataset takes about 15 minutes to train. However, on the much larger dataset of me and my friends' group chat, it takes a couple of hours. Once the training is done, the `save_model()` function will save the model, and then we'll call `evaluate()` to get some metrics. Here, we calculate the perplexity, a common metric for large language models (lower is better).

## Generating new text

Now it's finally time to generate some text. We'll make a new script `gen.py` and in it, we'll load the tokenizer and the newly fine-tuned model. Notice that we use the path to the fine-tuned model in the `from_pretrained()` function.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
model = AutoModelForCausalLM.from_pretrained("./models/finetuned-gpt2-large")

# Put model on GPU
device = torch.device("cuda")
model.to(device)
```

Next, we'll set a prompt that will be used as the starting point for the model's generation, set a seed so that our output is reproducible, and finally print out the result.

```python
# Tokenize prompt
prompt = "JULIET:"
input = tokenizer(prompt, return_tensors="pt")
input.to(device)

# Set seed
set_seed(137)

# Generate
model_out = model.generate(
    input_ids=input.input_ids,
    max_length=1024,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    do_sample=True,
    num_return_sequences=1,
)

# Print result
output = tokenizer.batch_decode(
    model_out, skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]
print("\n-------\n")
print(output)
```
If you're using `tinyshakespeare`, enjoy a never-before-seen act from a Shakespearean play. Or if you trained on your friends, go ahead and scare them with how well your AI can model your conversations.

## Conclusion and future improvements

There you have it, fine-tuning a GPT model with Hugging Face *Transformers*. This is only the first step in my quest to clone my friends using machine learning, but it was quite successful in impressing (and maybe scaring) them with the power of AI. The next step is to use [reinforcement learning from human feedback](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) (RLHF), the mechanism that ChatGPT uses, to actually have a natural conversation with my AI clones. Stay tuned!