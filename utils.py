import json
import numpy as np
import random
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm  

def load_config(path):
    with open(path, 'r') as f:
        cfg = json.load(f)
    return cfg

def load_checkpoint(model_dir, optimizer=None):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    if optimizer is not None:
        optimizer_state = torch.load(os.path.join(model_dir, 'optimizer.pt'))
        optimizer.load_state_dict(optimizer_state)
    
    training_state = torch.load(os.path.join(model_dir, 'training_state.pt'))
    epoch = training_state.get('epoch', 0)
    step = training_state.get('step', 0)
    updates = training_state.get('updates', 0)
    
    return model, tokenizer, updates, epoch, step

def save_checkpoint(model, tokenizer, optimizer, updates, model_dir, epoch, step):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer.pt'))
    torch.save({
        'updates': updates,
        'epoch': epoch,
        'step': step
    }, os.path.join(model_dir, 'training_state.pt'))

def test_language_modeling(model_dir, prompt, device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=1000,
        num_beams=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Text:\n" + "-" * 80)
    print(generated_text)

def estimate_loss(model, tokenizer, valid_loader, device='cuda'):
    model.eval() 
    losses = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Evaluating", leave=False):
            inputs = tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}  
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            losses.append(loss.item())
            if len(losses) >= 40:
                break
    model.train()
    return np.mean(losses)

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
