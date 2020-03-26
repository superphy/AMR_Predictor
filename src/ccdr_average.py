import numpy as np
import pandas as pd
import glob
import os, sys

import seaborn as sns
import matplotlib.pyplot as plt

#drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
drugs= ['AMC','AMP','AZM','CHL','CIP','CRO','FIS','FOX','SXT','TET','TIO','GEN','NAL']
trials=[i for i in range(1,11,1)]
models = ['XGB','SVM', 'ANN']

results = glob.glob('ccdr_out/*')
cols = ['drug','trial','model','acc','nme','me','vme']

df = pd.DataFrame(columns = cols, data = np.zeros((len(results), 7),dtype=object))

#[ccdr_1dacc,c_nmer/c_total,c_mer/c_total,c_vme/c_total])

for i, res in enumerate(results):
    if res[-4:] == '.npy':
        run = np.load(res)
        acc = run[0]
        nme = run[1]
        me = run[2]
        vme = run[3]
        model, drug, trial = res[:-4].split('/')[-1].split('_')

    else:
        run = pd.read_pickle(res)
        acc = run['1D Acc'][run.index[0]]

        nme = 0
        me = 0
        vme = 0

        model, drug, trial = res[:-4].split('/')[-1].split('_')

    df.at[i,'drug'] = drug
    df.at[i,'trial'] = trial
    df.at[i,'model'] = model
    df.at[i,'acc'] = acc
    df.at[i,'nme'] = nme
    df.at[i,'me'] = me
    df.at[i,'vme'] = vme




sns.set(style="darkgrid", rc= {"xtick.bottom": True, "ytick.left": True, "axes.facecolor": ".9"})

"""
save_df = pd.DataFrame(data=np.zeros((13,3),dtype='object'), index=drugs, columns = models)
for i, row in enumerate(summ_df.values):
    summ_drug = row[5]
    summ_model = row[1]
    sum_1Dacc = row[6]
    save_df.at[summ_drug,summ_model] = sum_1Dacc
save_df.to_csv("fig1_df.csv")
"""

names = ['Co-Amoxiclav','Ampicillin','Azithromycin','Chloramphenicol','Ciprofloxacin','Ceftriaxone','Sulfisoxazole',
'Cefoxitin','Co-Trimoxazole','Tetracycline','Ceftiofur','Gentamicin','Nalidixic Acid']

bar = sns.catplot(
x='drug', y ='acc', hue ='model', kind='bar', data=df.sort_values(by=['acc']),
hue_order = ["XGB", "SVM", "ANN"], ci='sd', capsize=0.1, errwidth=0.75
)
plt.xlabel("Antimicrobial", fontsize=14)
plt.ylabel("Accuracy (Within 1 Dilution)", fontsize=14)

for ax in bar.axes.flat:
    labels = ax.get_xticklabels()
    labels = [names[drugs.index(i.get_text())] for i in labels]
bar.set_xticklabels(labels, rotation=-30, ha='left')


import matplotlib.transforms as mtrans
for ax in bar.axes.flat:
    trans = mtrans.Affine2D().translate(-20, 0)
    for i, t in enumerate(ax.get_xticklabels()):
        if False:
            trans = mtrans.Affine2D().translate(-20, 20)
            t.set_transform(t.get_transform()+trans)
        else:
            trans = mtrans.Affine2D().translate(-10, 0)
            t.set_transform(t.get_transform()+trans)

    import matplotlib.ticker as plticker
    ax.set(ylim=(0, 1))
    loc = plticker.MultipleLocator(base=0.1)
    ax.yaxis.set_major_locator(loc)
    ax.set_yticks([i/40 for i in range(40)], minor=True)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)

fig = plt.gcf()
fig.set_size_inches(7, 5)
fig.subplots_adjust(bottom=0.25)
plt.show()
"""

result_df = pd.DataFrame(index = drugs, columns = models, data=np.zeros((len(drugs),len(models))))

for model in models:
    accs = []
    for drug in ['AMC','AMP','AZM','CHL','FIS','FOX','TET','TIO','GEN','NAL']:
        #sub_df = df[df['drug']==drug & df['model']==model]
        drug_df = df['drug'] == drug
        model_df = df['model'] == model
        sub_df = df[drug_df & model_df]
        for acc1 in sub_df['nme']:
            accs.append(acc1)
        #result_df.at[drug,model] = np.sum(sub_df['acc'])/len(sub_df['acc'])
        #result_df.at[drug,model] = np.std(sub_df['acc'])
        #print(drug, model)
        #print(np.sum(sub_df['acc'])/len(sub_df['acc']), np.std(sub_df['acc']))
    print(model)
    accs = np.array(accs)
    print(len(accs))
    print(np.sum(accs)/len(accs))
    print(np.std(accs))

#print(result_df)
"""
