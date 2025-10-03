# src/preprocess.py

import json
import os
import pickle

from transformers import AutoTokenizer as TOK
from collections import defaultdict

def run(config):
    print("Preprocessing data...")

    # read input path from config
    batch_size = config['run']['batch_size']
    model = config['run']['model']
    model_config_path  = config['run']['model_config']
    input_path = config['data']['raw']
    final_path = f'{config['data']['final']}/{model}'
    output_path = f'{final_path}/{config['data']['processed']}'
    num_senteces = config['data']['num_sentences']
    
    os.makedirs(final_path,exist_ok=True)
    #load model tokenizer
    with open(model_config_path, 'r') as f:
        model_path = json.load(f)[model]
    
    tokenizer = TOK.from_pretrained(model_path)
    
    # DATA
    # read data, sort by tokenized length, write to file
    
    #load data, will be sorted by sequence length
    with open(input_path, "r") as file:
        data = [line.strip() for line in file]

    if model=='fast':
        num_sentences = 500

    if num_senteces is not None:
        data = data[:num_senteces]
        
    
    # tokenize and compute sequence lengths
    length_groups = defaultdict(list)
    for sentence in data:
        tokenized = tokenizer(sentence, add_special_tokens=True)
        seq_length = len(tokenized["input_ids"])  # Get the length of the sequence
        length_groups[seq_length].append(sentence)

    # batch sequences of the same length
    batches = []
    for length, sentences in length_groups.items():
        # Divide sentences into batches of batch_size
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batches.append(batch)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for batch in batches:
            for seq in batch:
                output_file.write(seq + "\n")

    print(f"Data saved to {output_path}")
    return batches
