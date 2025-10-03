import pickle 
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns   # type: ignore
import os
from collections import Counter
import pandas as pd
from scipy.stats import linregress as lr
from scipy.stats import spearmanr as sp
from joblib import Parallel, delayed
import gc
import re
import json
import glob
import tqdm
import torch
rng = np.random.default_rng(0)

from transformers import AutoModelForCausalLM as MOD
from transformers import AutoTokenizer as TOK
os.environ["TOKENIZERS_PARALLELISM"] = "false"

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def boxplots(data, keys, template):
    """Boxplots of perplexity, entropy, kl divergence"""

    fig, axs = plt.subplots(1, 3, figsize=(9, 3.5))
    colors = sns.color_palette("tab10", len(keys))

    # Perplexity boxplot
    sns.boxplot([data[method]['surprisal'] for method in keys],patch_artist=True,ax=axs[0])
    axs[0].set_ylabel('Surprisal')
    axs[0].set_xticks(range(len(keys)))
    axs[0].set_xticklabels(keys, rotation=45, ha='right')

    # Entropy boxplot
    sns.boxplot([data[method]['entropy'] for method in keys],patch_artist=True,ax=axs[1])
    axs[1].set_ylabel('Entropy')
    axs[1].set_xticks(range(len(keys)))
    axs[1].set_xticklabels(keys, rotation=45, ha='right')

    # KL Divergence boxplot
    sns.boxplot([data[method]['kl_div'] for method in keys[1:]],patch_artist=True,ax=axs[2],
               palette=[colors[i] for i in range(1,len(keys))])
    axs[2].set_ylabel('KL Divergence')
    axs[2].set_yscale('log')
    axs[2].set_xticks(range(len(keys[1:])))
    axs[2].set_xticklabels(keys[1:], rotation=45, ha='right')

    # Perplexity difference boxplot
    # sns.boxplot([np.array(data[method]['ppl'])-np.array(data['full_model']['ppl']) for method in keys[1:]],patch_artist=True,ax=axs[3])
    # axs[3].set_title('Perplexity difference')
    # axs[3].set_xticks(range(len(keys[1:])))
    # axs[3].set_xticklabels(keys[1:], rotation=45, ha='right')
    # axs[3].set_yscale('log')
    # axs[3].set_ylim(1e-5,10)

    #plt.suptitle((template.split('/')[-1]).split('_')[1])
    plt.tight_layout()
    plt.savefig(f'{template}/boxplot.pdf')
    plt.close()

# -----------------------------------------------------------------------------------------------------------

def lin_coeff(data, keys, freqs, template, plot=False):
    """Compute linear fit between the frequencies of ground truth tokens against the frequency of predicted tokens"""

    # frequency empirical vs frequency models
    next_tk_freq = Counter(data['next_token'])
    tot = freqs[:,1].sum()
    freqs = dict(zip(freqs[:,0],freqs[:,1]/tot))
    slopes, spears, n_tokens, accuracy = [],[],[],[]
    # frequencies for next token model prediction
    for k in keys:
        tmp = [i[0] for i in data[k]['idx']]
        c = Counter(tmp)
        n_tokens.append(len(c))
        accuracy.append(np.sum(data['next_token']==np.array(tmp))/len(tmp))
        # vocab = set.union(*[set(next_tk_freq.keys()),set(c.keys())])
        # vocab = set.intersection(*[set(next_tk_freq.keys()),set(c.keys())])
        # ref = np.array([next_tk_freq[kk] for kk in vocab])/len(tmp)
        vocab = set.intersection(*[set(c.keys()),set(freqs.keys())])
        ref = np.array([freqs[kk] for kk in vocab])
        test = np.array([c[kk] for kk in vocab])/len(tmp)
        try:
            spear = sp(ref,test)
            line = lr(ref,test)
            spears.append(spear[0])
            slopes.append(line.slope)
            if plot:   
                plt.figure(figsize=(5,5))
                mx = max(ref.max(),test.max())
                mn = min(ref[ref>np.finfo(np.float32).eps].min(),test[test>np.finfo(np.float32).eps].min())
                x = np.logspace(np.log10(mn),np.log10(mx),100)
                plt.plot(x,x,'k--',alpha=0.75,label='y=x')
                plt.scatter(ref,test,s=20,alpha=0.5)
                plt.plot(x,line.slope*x+line.intercept, label=f'm={line.slope:.2f}',color='tab:orange',alpha=0.8)
                plt.plot([],[], color='none', label=fr'$\rho={spear[0]:.2f}$')
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.xlabel('corpus frequency', fontsize=14)
                plt.ylabel('prediction frequency', fontsize=14)
                plt.xscale('log')
                plt.yscale('log')
                plt.title(k.replace('_',' '),fontsize=15)
#                plt.xlim(0.65*mn,)
#                plt.ylim(0.65*mn,)
                plt.tight_layout()
                plt.legend(frameon=True,fontsize=14)
                plt.savefig(f'{template}/{k}.pdf')
                plt.close()
        except:
            spears.append(0)
            slopes.append(0)
        
    df = pd.DataFrame([keys,slopes,spears,n_tokens,accuracy]).T
    df.columns=['ablation','slopes','spearman','n_tokens','accuracy']
    df.to_csv(f'{template}/summary.csv',sep='\t', decimal='.', float_format='%.3f')

    # plot difference of freqs FM and abl ODs vs corpus freq
    tmp_FM = [i[0] for i in data['full_model']['idx']]
    c_FM = Counter(tmp_FM)
    tmp_abl_ods = [i[0] for i in data['ablate_ODs']['idx']]
    c_abl_ods = Counter(tmp_abl_ods)

    vocab = set.intersection(*[set(c_FM.keys()),set(c_abl_ods.keys()),set(freqs.keys())])
    ref = np.array([freqs[kk] for kk in vocab])
    FM = np.array([c_FM[kk] for kk in vocab])/len(tmp_FM)
    nODs = np.array([c_abl_ods[kk] for kk in vocab])/len(tmp_abl_ods)

    plt.figure()
    plt.scatter(ref,FM-nODs,s=20,alpha=0.5)
#    plt.hlines(0,min(ref),max(ref),color='k',linestyle='dashed',alpha=0.5)
    plt.xlabel('corpus frequency', fontsize=14)
    plt.ylabel('f(pref_FM) - f(pred_abl)', fontsize=14)
    plt.xscale('log')
    #plt.yscale('log')
    plt.savefig(f'{template}/diff.pdf')
    plt.tight_layout()
    plt.close()

    plt.figure()
    plt.scatter(ref,(FM-nODs)/FM,s=20,alpha=0.5)
#    plt.hlines(0,min(ref),max(ref),color='k',linestyle='dashed',alpha=0.5)
    plt.xlabel('corpus frequency', fontsize=14)
    plt.ylabel('f(pref_FM) - f(pred_abl)', fontsize=14)
    plt.xscale('log')
    #plt.yscale('log')
    plt.savefig(f'{template}/rel_diff.pdf')
    plt.tight_layout()
    plt.close()


# -----------------------------------------------------------------------------------------

def freq_vals(data, X_out, X, freqs, template):

    D = X.shape[1]
    nODs = data['ablate_ODs']['features']
    ODs = data['only_ODs']['features']

    # compute correlation val unembedding matrix vs corpus frequency
    sp_corr_vc = np.array(Parallel(4)(delayed(sp)(X_out[freqs[:,0],d],freqs[:,1]) for d in range(X_out.shape[1])))[:,0]

    # compute correlation values of activation vs freq(prediction)
    freqs = dict(zip(freqs[:,0],freqs[:,1]))
    preds = np.array([i[0] for i in data['full_model']['idx']])
    #c = Counter(preds)
    mask = []
    freqs_eff = []
    for i,p in enumerate(preds):
        try:
            freqs_eff.append(freqs[p])
            #freqs_eff.append(c[p])
            mask.append(i)
        except:
            continue
    mask = np.array(mask)
    freqs_eff = np.array(freqs_eff)
    sp_corr_full = np.array(Parallel(4)(delayed(sp)(X[mask,d],freqs_eff) for d in range(D)))[:,0]

    # plot
    labels = ['freq(predicted item)\nvs\nactivation','freq(voc item)\nvs\nunembedding value']
    plt.figure(figsize=(4,4))
    sns.boxplot([sp_corr_full[nODs],sp_corr_vc[nODs]], patch_artist=True, label='non-ODs', zorder=1, color='tab:blue')
    x = rng.uniform(-0.1,0.1,size=len(ODs))
    plt.scatter(x,sp_corr_full[ODs],label='ODs',color='tab:orange',marker='o',edgecolors='k',zorder=2)
    plt.scatter(x+1,sp_corr_vc[ODs],color='tab:orange',marker='o',edgecolors='k',zorder=3)

    plt.xticks(np.arange(len(labels)),labels = labels,fontsize=11)#, rotation=45, ha='right');
    plt.ylabel('spearman correlation',fontsize=11)

    t = 'correlation frequency against values'
    plt.title(t)
    plt.legend(loc='lower center',fontsize=10,frameon=True)
    plt.tight_layout()
    plt.savefig(f'{template}/freq_vals.pdf')
    plt.close()

    # plot unembedding matrix norm vs frequency
    norms = np.linalg.norm(X_out,axis=1)
    s = np.sum(list(freqs.values()))
    x = np.array(list(freqs.values()))/s
    y = norms[np.array(list(freqs.keys()))]
    um_norm_freq = sp(x,y)[0]
    plt.scatter(x,y,s=20,alpha=0.5)
    plt.xlabel('token corpus frequency',fontsize=14)
    plt.ylabel('token norm',fontsize=14)
    plt.title(rf'spearman $\rho={um_norm_freq:.3f}$')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{template}/freq_norm.pdf')
    plt.close()


# -----------------------------------------------------------------------------------------

def occurrencies(data, keys, template, X_inv):
    """Store occurrencies of gt and predicted tokens in a .csv file"""
    # compute frequencies for the different models
    next_tk_freq = Counter(data['next_token'])
    cc = {k:Counter([i[0] for i in data[k]['idx']]) for k in keys}
    # create vocabulary of all tokens predicted
    vocab = set.union(*[set(next_tk_freq.keys()),*[set(cc[k].keys()) for k in keys]])

    ks = ['full_model','ablate_ODs']
    preds = {kk:np.array([i[0] for i in data[kk]['idx']]) for kk in ks}
    freqs_data = {kk:Counter(preds[kk]) for kk in ks}
    idx_pred = {kk:{k:np.where(preds[kk]==k)[0] for k in freqs_data[kk].keys()} for kk in ks}

    ll = []
    for i in (list(vocab)):
        tmp = [i,str(X_inv[i]), next_tk_freq[i], cc['full_model'][i]]
        # for ablated model compute the ratio between frequencies
        #div = 1 if cc['full_model'][i] < 1 else cc['full_model'][i]
        #tmp.extend(cc[k][i]/div for k in keys[1:])
        tmp.extend(cc[k][i] for k in keys[1:])
        # intersection full with ablate
        try:
            inters = set(idx_pred['full_model'][i].tolist()).intersection(set(idx_pred['ablate_ODs'][i].tolist()))
        except:
            inters = []
        tmp.append(len(inters))
        ll.append(tmp) 

    cols = ['token_id','token','ground_truth']
    cols.extend(keys)
    cols.append('fm_int_ablate_ODs')
    df = pd.DataFrame(ll,columns=cols)
    df['token'] = df['token'].replace('\\r','\\return')
    df['token'] = df['token'].replace('\\n','\\newline')
    df['token'] = df['token'].replace('\\t','\\tab')
    df.to_csv(f'{template}/occurrencies.csv',sep='\t')

    return df

# -----------------------------------------------------------------------------------------

def compute_logits(data, X, X_out, X_inv, template, abs_val=False, annotate=False):

    if abs_val:
        X = abs(X)
        X_out = abs(X_out)
    t = (template.split('/')[-1]).split('_')[1]

    ods = np.array(data['only_ODs']['features'])
    nods = np.array(data['ablate_ODs']['features'])
    ks = ['full_model','only_ODs']#,'only_last']
    n_above = len(ods)//2
    preds = {kk:np.array([i[0] for i in data[kk]['idx']]) for kk in ks}
    freqs_data = {k:Counter(preds[k]) for k in ks}
    idx_pred = {kk:{k:np.where(preds[kk]==k)[0] for k in freqs_data[kk].keys()} for kk in ks}
    #tmp_idx = set.intersection(*[set(freqs_data['full_model'].keys()),set(freqs_data['ablate_ODs'].keys())])

    threshold = np.quantile(abs(X),0.99)

    df = pd.read_csv(f'{template}/od-only-tokens.csv',sep='\t')
    df = df[df['model']==t]
    od_favored = {k: idx_pred['only_ODs'][k] for k in df['token_id']}
    #print([(k,len(v)) for k,v in od_favored.items()])

    tmp = np.loadtxt(f'{template}/od-neutrals/{t}-neutral.txt',usecols=1,dtype=int)
    od_neutral = {k: idx_pred['full_model'][k] for k in tmp}
    #print([(k,len(v)) for k,v in od_neutral.items()])
    # # extract OD favored
    # od_favored = {}
    # thres = 0.25
    # for kk in list(tmp_idx): 
    #     if freqs_data['full_model'][kk]<30:
    #         continue
    #     if freqs_data['ablate_ODs'][kk]/freqs_data['full_model'][kk]>thres:
    #         continue
    #     od_favored[kk]=idx_pred['full_model'][kk]

    # # extract OD-neutral
    # bottom = 25
    # up = 50
    # thres_1 = 0.9
    # od_neutral = {}
    # for kk in list(tmp_idx):
    #     a = set(idx_pred['full_model'][kk].tolist())
    #     if len(a)<bottom or len(a)>up: 
    #         continue
    #     b = set(idx_pred['ablate_ODs'][kk].tolist())
    #     c = a.intersection(b)
    #     if len(c)<thres_1*len(a):
    #         continue
    #     od_neutral[kk] = list(c)
    
    flag = False
    if len(od_favored)==0:
        print('no od-favored tokens found')
        flag=True
    if len(od_neutral)==0:
        print('no od-neutral tokens found')
        flag=True
    if flag:
        return

    def group(source, target=None):

        if target is None:
            pred_od, pred_nod = [],[]
            for ii,id_source in enumerate(source.keys()):
                x_sub = X[source[id_source]]
                #print('total num elements: ',len(x_sub))
                idx_temp = np.where(np.sum(abs(x_sub[:,ods])>threshold,axis=1) > n_above)[0]
                #print(f'elements that have at least {n_above} ods over {threshold}: {len(idx_temp)}')
                x_sub = x_sub[idx_temp]
                logits = x_sub[:,ods]@X_out[id_source,ods]
                nlogits = x_sub[:,nods]@X_out[id_source,nods]
                pred_od.append(logits)
                pred_nod.append(nlogits)

            return pred_od, pred_nod
            
        else:
            pred_od, pred_nod, other_od, other_nod = [],[],[],[]
            for ii,id_source in enumerate(source.keys()):#[0:2]:
                #print(id_source, X_inv[id_source])
                x_sub = X[source[id_source]]
                #print('total num elements: ',len(x_sub))
                idx_temp = np.where(np.sum(abs(x_sub[:,ods])>threshold,axis=1) > n_above)[0]
                #print(f'elements that have at least {n_above} ods over {threshold}: {len(idx_temp)}')
                x_sub = x_sub[idx_temp]
                logits = x_sub[:,ods]@X_out[id_source,ods]
                nlogits = x_sub[:,nods]@X_out[id_source,nods]
                pred_od.append(logits)
                pred_nod.append(nlogits)
                tmp1, tmp2 = [],[]
                for id_target in target:
                    if id_target == id_source:
                        continue
                    tmp1.append(x_sub[:,ods]@X_out[id_target,ods])
                    tmp2.append(x_sub[:,nods]@X_out[id_target,nods])
    
                other_od.append(tmp1)
                other_nod.append(tmp2)
            return pred_od, pred_nod, other_od, other_nod
    
    def plot_single(pred_od, pred_nod, other_od, other_nod, x, y, filename):

        plt.figure()
        plt.scatter(pred_nod,other_nod, label = 'non-ODs', marker='s',edgecolors='k')
        plt.scatter(pred_od,other_od, label = 'ODs', marker='o',edgecolors='k')
        plt.xlabel(f"contribution towards predicted token '{x}'",fontsize=13)
        plt.ylabel(f"contribution towards non-predicted token '{y}'",fontsize=13)
        plt.title((filename.split('/')[-1]).split('_')[1],fontsize=16,y=1.0, pad=-14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=13,frameon=True, loc='lower left')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_averages(source, pred_od, pred_nod, other_od, other_nod, X_inv, filename, annotate=annotate):

        offset=0.25
        fig, ax = plt.subplots()
        for i,k in enumerate(list(source.keys())):  
            if i == 0:
                od_legend = 'ODs'
                nod_legend = 'non-ODs'
            else:
                od_legend = None
                nod_legend = None
                
            src_od_mean = np.mean(pred_od[i])
            src_od_std = np.std(pred_od[i])
            src_nod_mean = np.mean(pred_nod[i])
            src_nod_std = np.std(pred_nod[i])
            trg_od_mean = np.mean(other_od[i])
            trg_od_std = np.std(other_od[i])
            trg_nod_mean = np.mean(other_nod[i])
            trg_nod_std = np.std(other_nod[i])
            
            plt.errorbar(src_nod_mean, trg_nod_mean, xerr=src_nod_std, yerr=trg_nod_std,capsize=2.5,elinewidth=1,linestyle='dashed',ecolor='gray',zorder=-1,alpha=0.4)
            plt.scatter(src_nod_mean, trg_nod_mean, s=18, marker='s',c='tab:blue', edgecolor='k', label=nod_legend)
            plt.errorbar(src_od_mean, trg_od_mean, xerr=src_od_std, yerr=trg_od_std,capsize=2.5,elinewidth=1,linestyle='dashed',ecolor='gray',zorder=-1,alpha=0.4)
            plt.scatter(src_od_mean, trg_od_mean, s=18, marker='o',c='tab:orange', edgecolor='k', label=od_legend)

            if annotate:
                note = X_inv[k].lstrip('Ġ').lstrip('_').lstrip('▁')
                if note in ['before', 'States', 'bridge','command','mm']:
                    ax.annotate(X_inv[k].lstrip('Ġ'), (src_od_mean+offset, trg_od_mean-0.015),fontsize=13, alpha=0.8)
                    ax.annotate(X_inv[k].lstrip('Ġ'), (src_nod_mean+offset, trg_nod_mean-0.015),fontsize=13, alpha=0.8)

        plt.xticks(fontsize=13)
        plt.xlabel('contribution towards predicted token logit', fontsize=13)
        plt.yticks(fontsize=13)
        plt.ylabel('contribution towards OD-favored tokens', fontsize=13)
        t = (filename.split('/')[-1]).split('_')[1]
        plt.title(t,fontsize=16,y=1.0, pad=-14)

        plt.tight_layout()
        if t=='pythia-12b':
            plt.legend(fontsize=13, frameon=True, loc='lower left')    
        else:
            plt.legend(fontsize=13, frameon=True)#, loc='lower left')
        plt.savefig(filename) 
        plt.close()

    def boxes(pred_od, pred_nod, filename):

        colors = sns.color_palette("tab10", 2)
        plt.figure(figsize=(3.,4))
        sns.boxplot([pred_od,pred_nod], patch_artist=True,palette=[colors[1],colors[0]])
        plt.xticks([0,1],['ODs','non-ODs'],fontsize=16)#, rotation=45, ha='right')
        plt.yticks(fontsize=14)
        plt.ylabel('logit', fontsize=16)
        plt.title((filename.split('/')[-1]).split('_')[1],fontsize=17)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    pred_od, pred_nod, other_od, other_nod = group(od_neutral,od_favored)
    # find largest pop among the od_neutral
    k = ''
    i = 0
    j = 0
    tmp = 0
    for kk,vv in od_neutral.items():
        if len(vv) > tmp:
            k = kk
            tmp = len(vv)
            j = i
        i+=1
    # plot single combination
    k_other = list(od_favored.keys())[2]
    plot_single(pred_od[j], pred_nod[j], other_od[j][2], other_nod[j][2], X_inv[k].lstrip('Ġ').lstrip('▁'), X_inv[k_other].lstrip('Ġ').lstrip('▁'), f'{template}_single.pdf')
    # plot averages
    plot_averages(od_neutral, pred_od, pred_nod, other_od, other_nod, X_inv, f'{template}_averages.pdf', annotate=annotate)
    # plot boxplots for od favored
    od_fav, nod_fav = group(od_favored)
    ood = [z for y in od_fav for z in y]
    onod = [z for y in nod_fav for z in y]
    boxes(ood, onod, f'{template}_boxes.pdf' )

# -----------------------------------------------------------------------------------------

def check_for_massive(X,t=1000):
    idx = np.where( abs(X) > 1000*np.mean(abs(X)) )
    print('number of massive activations: ', len(idx[0]))
    return idx

# -----------------------------------------------------------------------------------------

def run(config):

    final_path = config['data']['final']
    model_name = config['run']['model']
    model_config_path  = config['run']['model_config']

    final_path = f'{final_path}/{model_name}'

    #load model 
    with open(model_config_path, 'r') as f:
        model_path = json.load(f)[model_name]

    tokenizer = TOK.from_pretrained(model_path)
    llm = MOD.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True)    
    
    # store dictionary
    X_voc = tokenizer.get_vocab() 
    X_inv = {v: k for k, v in X_voc.items()}
    del tokenizer

    # store unembedding matrix
    if model_name in ["opt", "llama", "mistral", "olmo", "qwen", "stable","gemma"]:
        X_out = llm.lm_head.weight.detach().numpy()
    else:
        X_out = llm.embed_out.weight.detach().numpy()
    del llm

    gc.collect()

    # load ablations data--------------------------------------------------------------------------------------
    with open(f'{final_path}/results.pickle', 'rb') as f:
        data = pickle.load(f)

    # group together the random experiments
    data['only_random'] = {}
    data['ablate_random'] = {}
    for k in ['entropy','surprisal','kl_div']:
        data['only_random'][k] = np.array([data['only_random'+str(i)][k] for i in range(5)]).mean(axis=0)
        data['ablate_random'][k] = np.array([data['ablate_random'+str(i)][k] for i in range(5)]).mean(axis=0)

    # load last layer representations
    reps_file = sorted(glob.glob(f'{final_path}/reps/l_*.pickle'), key=natsort)[-1]
    print(f'last layer filename: {reps_file}')
    with open(reps_file, 'rb') as f:
        Y = pickle.load(f) 
    X = np.array([ np.float32(v[0]) for k,v in Y.items() ])
    del Y 

    # corpus frequencies
    freqs = np.loadtxt(f'data/wikitext_{model_name.replace('fast','pythia')}_counts.txt',dtype=int)

    # RUN ANALYSES---------------------------------------------------------------------------------------------
    # check for the presence of massive activations
    _ = check_for_massive(X)

    # # box plot of surprisal/entropy/kl with every ablation method----------------------------------------------
    path = f'{final_path}/boxplots/'
    os.makedirs(path,exist_ok=True)
    keys = ['full_model','ablate_ODs','ablate_random','only_ODs','only_random']#'ablate_out','only_out'
    boxplots(data, keys, path)

    # relationship between ground truth next token frequencies and predictions---------------------------------
    path = f'{final_path}/freqs/'
    os.makedirs(path,exist_ok=True)
    keys = ['full_model','ablate_ODs','ablate_random0','only_ODs','only_random0','only_random1','only_random2','only_random3','only_random4']#'ablate_out','only_out',
    lin_coeff(data, keys, freqs=freqs, template=path, plot=True)

    # correlation freq vs values ------------------------------------------------------------------------------
    path = f'{final_path}/freq_vals/'
    os.makedirs(path,exist_ok=True)
    freq_vals(data, X=X, X_out=X_out, freqs = freqs, template = path)

    # # save csv of n most frequent next tokens in corpus and their frequency for each ablation method----------
    path = f'{final_path}/occurrence/'
    os.makedirs(path,exist_ok=True)
    keys = ['full_model','ablate_ODs','ablate_random0','only_ODs','only_random0','only_random1','only_random2','only_random3','only_random4']#'ablate_out','only_out',
    occurrencies(data, keys, path, X_inv)

    # # compute logits contributions-----------------------------------------------------------------------------
    # this can be done after extracting OD-favored and OD-neutral tokens from the previous csv files
    # procedure yet to be automatised
    # path = f'{final_path}/logits/'
    # os.makedirs(path, exist_ok=True)
    # annotate = True if model=='pythia-12b' else False
    # compute_logits(data, X, X_out, X_inv, f'{path}{corpus}_{model}', annotate=annotate)


