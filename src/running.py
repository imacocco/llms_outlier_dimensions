# src/running.py

import os
import torch
from tqdm import tqdm
import json
import os
import pickle

from transformers import AutoModelForCausalLM as MOD
from transformers import AutoTokenizer as TOK
from collections import defaultdict

# Mapping function
## sequence tokens' index doesn't correspond to LM tokens' index
## -> solve offset by word to LM token mapping

def tokenize_map(input_text, tokenizer):
    # Tokenize with offsets
    tokenized = tokenizer(input_text, return_offsets_mapping=True, add_special_tokens=False)

    # Extract tokens and offsets
    tokens = tokenized['input_ids']
    offsets = tokenized['offset_mapping']

    # Split the input text into words
    input_words = input_text.split()

    # Initialize the mapping structure and tracking variables
    word_to_token_mapping = {}
    current_word_index = 0
    current_word_tokens = []

    for i, (start, end) in enumerate(offsets):
        # If the current token starts beyond the end of the current word
        if start >= len(" ".join(input_words[:current_word_index + 1])):
            # Map the current word index to its tokens and update the mapping
            word_to_token_mapping[current_word_index] = (current_word_tokens,  i, start) ## Triple, 
            current_word_index += 1

            # Reset current word tokens if there are more words to process
            if current_word_index < len(input_words):
                current_word_tokens = []

        # Add the current token ID to the list of tokens for the current word
        current_word_tokens.append(tokens[i])

    # Add the last word to the mapping (if any tokens remain)
    if current_word_tokens:
        word_to_token_mapping[current_word_index] = (current_word_tokens, len(offsets), len(input_text))

    return word_to_token_mapping
    
def run(batches, config):
    print("Loading & running model...")

    # get parameters from config
    batch_size = config['run']['batch_size']
    final_path = config['data']['final']
    model_name = config['run']['model']
    model_config_path  = config['run']['model_config']
    only_last_layer  = config['run']['only_last']
    
    #load model 
    with open(model_config_path, 'r') as f:
        model_path = json.load(f)[model_name]

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    tokenizer = TOK.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = MOD.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatically map layers to available devices
        trust_remote_code=True)  # For custom models
    
    
    # CODE
    # Initialize main structure of hidden_representations
    ## layer
    ### sequence
    #### last 100+ tokens' (wiki, not LM tokens!) vectors
    hr = defaultdict(lambda: defaultdict(list))

    # Generated prediction of s[:99] (first 100 tokens)
    ## sequence ID
    ### Quartuple: (100th token, 101th token gold, pred, pred's logits)
    gen = defaultdict()

    # Automatically create layers and sequences
    num_layers = model.config.num_hidden_layers
    c_id = 0
    for batch in tqdm(batches, desc="Processing"):
        inputs = tokenizer(batch, return_tensors="pt")#.to(device)
        for k in inputs:
            inputs[k] = inputs[k].to(model.device)
        temp_output = model(**inputs, output_hidden_states=True)

        for seq_id in range(len(batch)):
            s = batch[seq_id]
            #print(s)
            i_token = s.split()
            # mapping
            temp_map = tokenize_map(s, tokenizer)

            # Works with 100th WIKI Token !
            num_tokens = temp_map[len(i_token)-1][1] - temp_map[99][1] # amount of LM tokens - index LM token of (last) 100th wiki_token
            l_index_token = list(range(1,num_tokens + 2))
            # [1,2,3,...n][-n] = 1, though we need 0, thus +1, +1 bc of range
            l_index_token.reverse() #!
        
            for token in l_index_token: # iterate over last tokens, starting from LM token of (last) 100th wiki_token
                if only_last_layer:
                    hr[f"l_{num_layers}"][f"id_{c_id + seq_id}"] = hr[f"l_{num_layers}"][f"id_{c_id + seq_id}"] + [(temp_output['hidden_states'][num_layers][seq_id,-token,:].cpu().detach().numpy())]
                else:    
                    for layer in range(1, num_layers+1): #not layer 0, num_layers+1
                        hr[f"l_{layer}"][f"id_{c_id + seq_id}"] = hr[f"l_{layer}"][f"id_{c_id + seq_id}"] + [(temp_output['hidden_states'][layer][seq_id,-token,:].cpu().detach().numpy())]

            i = tokenizer(s[:temp_map[99][2]], return_tensors="pt")#.to(device) # until last character index of 100th token
            i = i.to(model.device)
            pred = model.generate(
                **i,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,  # Return a dict with additional outputs
                output_scores=True,             # Include scores (logits) in the output
                pad_token_id=tokenizer.eos_token_id
            )

            gen[f"id_{c_id +seq_id}"] = (i_token[99],i_token[100],tokenizer.decode(pred.sequences[0][-1]),pred.scores[0].cpu().detach().numpy()) 
            # token 100, gold, pred, logits
        c_id = c_id + batch_size

    del model

    # SAVING HRs and PREDs in directory named after model
    final_path = f'{final_path}/{model_name}'
    os.makedirs(f'{final_path}/reps',exist_ok=True)
    for l in hr.keys():
        with open(os.path.join(f'{final_path}/reps/{l}_layer.pickle'),'wb') as f:
            pickle.dump(hr[l],f)
    del hr

    with open(os.path.join(final_path, 'pred.pickle'),'wb') as g:
        pickle.dump(gen,g)
    del gen

    print("Run complete.")
