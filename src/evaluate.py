# src/evaluate.py

import numpy as np
rng = np.random.default_rng(0)
import torch
import random
from tqdm import tqdm
import json
import os
import pickle

import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM as MOD
from transformers import AutoTokenizer as TOK
from collections import defaultdict, Counter


def outlier_dims_quantile(X, loc = 0.5, glo = 0.99, sort=True, outfile=None):
	X = abs(X)
	qt = np.quantile(X,q=loc,axis=0)
	threshold = np.quantile(X,q=glo)
	idx = np.where(qt>threshold)[0]

	if sort:
		idx = idx[np.argsort(qt[idx])[::-1]]

	if outfile is not None:
		fig, ax = plt.subplots()

		if sort:
			qt = np.sort(qt)
			ext = '_sorted' if sort else ''

		plt.scatter(np.arange(0,len(qt)),qt,alpha=0.999,s=10, label='median of dimension',edgecolors='lightblue', lw=0.15)
		plt.hlines(threshold,0,len(qt),color='tab:orange',label='global quantile = 0.99',linestyles='--', lw=1.5)

		plt.grid(linestyle = '--', alpha=0.5)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel('dimension', fontsize=16)
		plt.ylabel(f'dimension quantile={loc}', fontsize=16)
		
		plt.legend()#bbox_to_anchor=(1.0, 1))
		plt.tight_layout()
		plt.legend(fontsize=13.5)
		plt.savefig(f'{outfile}{ext}.pdf')
		plt.close()
		
	return idx


def run(config):
	print("Evaluating model representations...")

	# get parameters from config
	final_path = config['data']['final']
	only_last_layer  = config['run']['only_last']
	model_name = config['run']['model']
	model_config_path  = config['run']['model_config']
	only_last_layer  = config['run']['only_last']
	qt = config['run']['qt']
	threshold = config['run']['threshold']
	plot_ods = config['run']['plot_ods']

	final_path = f'{final_path}/{model_name}'
	pred_dir = final_path+'/pred.pickle'
	if plot_ods:
		os.makedirs(final_path+'/plots',exist_ok=True)

	#load model 
	with open(model_config_path, 'r') as f:
		model_path = json.load(f)[model_name]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = TOK.from_pretrained(model_path)
	llm = MOD.from_pretrained(
		model_path,
		torch_dtype=torch.float16,
		device_map="auto",  # Automatically map layers to available devices
		trust_remote_code=True)  # For custom models

	layer_dir = final_path
	num_layers = llm.config.num_hidden_layers
	lastlayer_dir = layer_dir + '/reps/l_'+str(num_layers)+'_layer.pickle'

	# LOADING & SAVING HRs
	print("#1:Calculating outliers layerwise...")
	outis=defaultdict(defaultdict(int).copy)
	for layer in tqdm(range(1,num_layers+1), desc="Processing"): #33

		if only_last_layer: #process only last layer
			layer = num_layers
		
		with open(layer_dir + '/reps/l_'+str(layer)+'_layer.pickle', 'rb') as g:
			h = pickle.load(g) #UnpicklingError: pickle data was truncated
	
		dimensions = defaultdict(list)
		# Initialize dimensions dictionary with empty lists
		num_dimensions = len(h[next(iter(h))][0])  # Determine number of dimensions
		dimensions = {d: [] for d in range(num_dimensions)}
		# Convert h to a structured array for efficient indexing
		keys = list(h.keys())
		# print(len(h[keys[1]]))
		# print(len(h[keys[1]][0]))
		#try:
		h_array = np.array([h[k][0] for k in keys])  # Extract first element of the list which corresponds to the first token after the 100th!
		
		# Append values to corresponding dimensions
		for d in range(num_dimensions):
			#dimensions[d] = filtered_h[:, d].tolist()  # Extract all values for dimension `d`
			dimensions[d] = h_array[:, d].tolist()
		h, h_array = None, None

		reps = np.array([dimensions[i] for i in dimensions.keys()], dtype=np.float16).T
		dimensions = None
		if plot_ods:
			outis[layer] = outlier_dims_quantile(reps, loc=qt, glo=threshold, outfile=f'{final_path}/plots/ods_l_{layer}')
		else:
			outis[layer] = outlier_dims_quantile(reps, loc=qt, glo=threshold)

		if only_last_layer:
			break

	outis_transformed = {k: v.tolist() for k, v in outis.items()}  

	f_path = f"{layer_dir}/OD.json" #writes calculated outliers in layer to file
	with open(f_path, 'w') as fp:
		json.dump(outis_transformed, fp, indent=4)

	# read representations
	print("#2:Fetching representations...")
	with open(lastlayer_dir, 'rb') as g:  # Open in binary read mode
		h = pickle.load(g)
	keys = list(h.keys())

	# previous bug: Ensure enumeration of keys is continuous and starts from 0
	p_copy = defaultdict()
	for i,k in enumerate(list(h.keys())):
		p_copy["id_"+str(i)] = h[k]
	h = p_copy
	p_copy = None
	dimensions = defaultdict(list)
	# Initialize dimensions dictionary with empty lists
	num_dimensions = len(h[next(iter(h))][0])  # Determine number of dimensions
	dimensions = {d: [] for d in range(num_dimensions)}
	# Convert h to a structured array for efficient indexing
	keys = list(h.keys())
	#try:
	h_array = np.array([h[k][0] for k in keys])  # Extract only first token after the 100-th!

	# Append values to corresponding dimensions
	for d in range(num_dimensions):
		#dimensions[d] = filtered_h[:, d].tolist()  # Extract all values for dimension d
		dimensions[d] = h_array[:, d].tolist()

	# load vocabulary and its inverse
	X_voc = tokenizer.get_vocab() 
	X_inv = {v: k for k, v in X_voc.items()}

	# load unembedding matrix
	if model_name in ["opt", "llama", "mistral", "olmo", "qwen", "stable","gemma"]:
		X_out = llm.lm_head.weight 
	else:
		X_out = llm.embed_out.weight
	del llm

	X_out = X_out[:len(X_voc)]
	D = X_out.shape[1]
	obs_keys = ['entropy', 'kl_div', 'surprisal', 'probas', 'idx']

	# find outlier dimensions and its complementary
	try:
		rd_out = outlier_dims_quantile(X_out.cpu().detach().numpy())
	except NotImplementedError:
		rd_out = outlier_dims_quantile(X_out.numpy())
	not_out = np.setdiff1d(np.arange(D), rd_out)

	reps = np.array([dimensions[i] for i in dimensions.keys()], dtype=np.float16).T
	del dimensions

	print("#3:Collecting gold-standard...")
	with open(pred_dir, 'rb') as g:  # Open in binary read mode
		next_token = pickle.load(g)

	p_copy = defaultdict() #ORDER!
	for i,k in enumerate(list(next_token.keys())):
		p_copy["id_"+str(i)] = next_token[k]

	next_token = p_copy
	del p_copy #print(len(next_token))

	#GOld: account for different tokenizer outputs
	if model_name =="mistral" and type(next_token["id_0"][1]) != tuple:
		next_token = np.array([tokenizer(next_token[k][1]).input_ids[1] for k in next_token.keys()]) 
	elif model_name =="opt" and type(next_token["id_0"][1]) != tuple:
		next_token = np.array([tokenizer(" "+next_token[k][1]).input_ids[1] for k in next_token.keys()])
	else:
		if type(next_token["id_0"][1]) ==tuple:
			n =[]
			for k in next_token.keys():
				n.append(next_token[k][1][-1].cpu())
			next_token = np.array(n) #Tensor
			n = None
		else:
			next_token = np.array([tokenizer(" "+next_token[k][1]).input_ids[0] for k in next_token.keys()])

	rd_last = outis_transformed[num_layers]

	not_last = np.setdiff1d(np.arange(D), rd_last)
	# put together the outliers of the last layer and the output layer
	not_rd = list(set.intersection(*[set(not_out),set(not_last)]))

	# group all subsets of features together
	ablation_modalities = {'only_ODs':rd_last, 'only_out':rd_out, 'ablate_ODs':not_last, 'ablate_out':not_out}

	for i in range(5):
		ablation_modalities['only_random'+str(i)] = rng.choice(not_rd,len(rd_last),replace=False)
		ablation_modalities['ablate_random'+str(i)] = np.setdiff1d(np.arange(D), ablation_modalities['only_random'+str(i)])

	# prepare results dictionary
	results = {kk:{k: [] for k in obs_keys} for kk in ablation_modalities.keys()}   # lists for each observable
	for k in list(results.keys()):# store the selected features
		results[k]['features'] = ablation_modalities[k]
		results['next_token'] = next_token  # store ground truth next token
		results['full_model'] = {k: [] for k in obs_keys}   # store lists for full model

	# debugging dictionaries #print(results.keys())

	print("#4:Final calculations...")
	reps = torch.tensor(reps).to(device)
	#print('GPU usage\t',torch.cuda.mem_get_info())
	N = len(reps) 
	batch_size = 100 # make it smaller if it goes out of memory
	print(f'Batch size for observables computation set to {batch_size}. Reduce if GPU memory is not sufficient')
	n_batches = N//batch_size+1

	X_outT= X_out.T.to(device)
	X_out= X_out.to(device)

	for n in range(n_batches):
		begin = n*batch_size
		end = min(N,(n+1)*batch_size)
		batch_size = end-begin
		
		my_reps = reps[begin:end] 
		P = torch.log_softmax(my_reps@X_outT,dim=1)
		results['full_model']['entropy'].extend(torch.sum(-P.exp()*P,dim=1).cpu().detach().numpy())
		idx = torch.argsort(P,dim=1,descending=True)[:,:20] # find indices for largest P
		results['full_model']['idx'].extend(idx.cpu().detach().numpy().tolist()) # save top 20 probabilities and associated tokens
		results['full_model']['probas'].extend(torch.take_along_dim(P.exp(),idx,dim=1).cpu().detach().numpy().tolist())    # save probabilities
		results['full_model']['surprisal'].extend( [-P[i,ground_truth].cpu().detach().numpy() for i,ground_truth in enumerate(next_token[begin:end])] )
		
		# cycle over different ablations modalities
		for k in ablation_modalities.keys():
			Q = torch.log_softmax(my_reps[:,ablation_modalities[k]]@X_out[:,ablation_modalities[k]].T,dim=1)
			results[k]['entropy'].extend(torch.sum(-Q.exp()*Q,dim=1).cpu().detach().numpy())
			idx = torch.argsort(Q,dim=1,descending=True)[:,:20]
			results[k]['idx'].extend(idx.cpu().detach().numpy().tolist())
			results[k]['probas'].extend(torch.take_along_dim(Q.exp(),idx,dim=1).cpu().detach().numpy().tolist())
			results[k]['surprisal'].extend( [-Q[i,ground_truth].cpu().detach().numpy() for i,ground_truth in enumerate(next_token[begin:end])] )
			results[k]['kl_div'].extend(torch.sum(P.exp() * (P - Q),axis=1).cpu().detach().numpy())
			
	del reps, P, Q
	torch.cuda.empty_cache()

	with open(f'{layer_dir}/results.pickle','wb') as f:
		pickle.dump(results,f)

	print("Results for the surprisal:")
	print('Ablation modality\tSurprisal')
	print('full_model', "\t"+str(np.mean(np.array(results['full_model']['surprisal']))))
	for i in ['ablate_ODs','ablate_random0','only_ODs','only_random0']:
		print(i, "\t"+str(np.mean(np.array(results[i]['surprisal']))))    

	print("Evaluation complete.")
