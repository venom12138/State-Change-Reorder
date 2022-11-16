import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np

csv_path = '/home/venom/.exp/1105_reordering_softsort/D0236_freeze=0,repr_type=Segmentation,cos_lr=True,bs=8,lr=1e-5/D0236_freeze=0,repr_type=Segmentation,cos_lr=True,bs=8,lr=1e-5_per_video_results.csv'
repr_type = 'Segmentation'

plt.rcParams['font.size'] = '10'
plt.rcParams['font.family'] = 'Times New Roman'

with open('../val_data/EPIC100_state_positive_val.yaml', 'r') as f:
    val_data_info = yaml.safe_load(f)

csv_data = pd.read_csv(csv_path)

verb_df = pd.read_csv('/home/venom/data/EPIC_100_verb_classes.csv')
noun_df = pd.read_csv('/home/venom/data/EPIC_100_noun_classes.csv')

verb_performance = {}
for index, row in csv_data.iterrows():
    key = row['key']
    verb = verb_df['key'][val_data_info[key]['verb_class']]
    
    if verb not in verb_performance.keys():
        verb_performance[verb] = {'spearman':[row['spearman']], 
                                'ab_distance':[row['ab_distance']], 
                                'pairwise_acc':[row['pairwise_acc']]}
    else:
        verb_performance[verb]['spearman'].append(row['spearman']) 
        verb_performance[verb]['ab_distance'].append(row['ab_distance']) 
        verb_performance[verb]['pairwise_acc'].append(row['pairwise_acc']) 

spearman_list = []
ab_distance_list = []
pairwise_acc_list = []
for verb in verb_performance.keys():
    spearman_list.append(np.mean(verb_performance[verb]['spearman']))
    ab_distance_list.append(np.mean(verb_performance[verb]['ab_distance']))
    pairwise_acc_list.append(np.mean(verb_performance[verb]['pairwise_acc']))

################################################

### spearman
baselen = 5
fig, ax = plt.subplots(1, 1,
                        figsize=(baselen, baselen),
                        constrained_layout=True
                        )
ax.grid(alpha=0.4)

ax.bar(list(verb_performance.keys()), spearman_list, align='center', alpha=0.5, ecolor='black', capsize=8)
# ax.set_title(f'Spearman Correlation repr_type={repr_type}')
ax.set_ylabel('Spearman Correlation')
# ax.set_xlabel('Verb Class')
# plt.legend()
save_path = '/'.join(csv_path.split('/')[:-1]) + '/spearman.pdf'
fig.savefig(save_path)

################################################

### ab_distance
baselen = 5
fig, ax = plt.subplots(1, 1,
                        figsize=(baselen, baselen),
                        constrained_layout=True
                        )
ax.grid(alpha=0.4)

ax.bar(list(verb_performance.keys()), ab_distance_list, align='center', alpha=0.5, ecolor='black', capsize=8)
# ax.set_title(f'Abs Distance repr_type={repr_type}')
ax.set_ylabel('Abs Distance')
# ax.set_xlabel('Verb Class')
# plt.legend()
save_path = '/'.join(csv_path.split('/')[:-1]) + '/abs_distance.pdf'
fig.savefig(save_path)

################################################

### pairwise_acc_list
baselen = 5
fig, ax = plt.subplots(1, 1,
                figsize=(baselen, baselen),
                constrained_layout=True
                )
ax.grid(alpha=0.4)

ax.bar(list(verb_performance.keys()), pairwise_acc_list, align='center', alpha=0.5, ecolor='black', capsize=8)
# ax.set_title(f'Pairwise Accuracy repr_type={repr_type}')
ax.set_ylabel('Pairwise Accuracy')
# ax.set_xlabel('Verb Class')
# plt.legend()
save_path = '/'.join(csv_path.split('/')[:-1]) + '/pairwise_acc.pdf'
fig.savefig(save_path)