import matplotlib.pyplot as plt
import pandas as pd
import yaml

with open('/u/ryanxli/venom/State-Change-Reorder/val_data/EPIC100_state_positive_val.yaml', 'r') as f:
    val_data_info = yaml.safe_load(f)

csv_path = '/u/ryanxli/.exp/0925_state_change_segm/1105_reordering_softsort/D0212_freeze=0,repr_type=Clip,cos_lr=True,bs=8,lr=1e-5/per_video_results.csv'
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

plt.bar(range(len(verb_performance.keys())), spearman_list, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.title('Spearman Correlation repr_type=Clip')
plt.legend()
plt.show()