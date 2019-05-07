import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os, sys
import glob

def usage():
    print("\nusage: python all_data_figures {0-1}\n\nOptions:")
    print(
    "0 - Multiface accuracy vs features, all drugs all models",
    "1 - Multibar graph, 13 sets of 7 bars detailing 13 drug accuracies in each of 7 dataset comparisons",
    "2 - Simpsons Diversity Plots",
    sep = '\n')
    return

def label_points(x, y, labels, ax):
    for i, label in enumerate(labels):
        if label == 'CHL':
            ax.text(x[i]+0.005, y[i]-0.005, label, fontsize = 7)
        else:
            ax.text(x[i]+0.005, y[i]+0.005, label, fontsize = 7)

if __name__ == "__main__":
    figure = 0
    try:
        figure = sys.argv[1]
    except:
        usage()
        sys.exit(2)

    path = "results/"
    num_samples = sum([len(files) for r, d, files in os.walk(path)])

    print("found {} samples in {}".format(num_samples, path))

    master_df = pd.DataFrame(data = np.zeros((num_samples, 7),dtype = 'object'),index =[i for i in range(num_samples)], columns = ['acc','model','feats','train','test','drug', '1Dacc'])
    title_string = ''
    counter = 0
    directs = [[root,files] for root, dirs, files in os.walk(path) if not dirs]
    for i in directs:
        direc = i[0]
        for k, file in enumerate(i[1]):
            filepath = direc+'/'+file

            file = file.split('.')[0]
            drug, num_feats, train, test = file.split('_')
            num_feats = int(num_feats[:-5])
            model = train[:3]
            train = train[12:]
            test = test[8:]
            #print(drug, num_feats, model, train, test)
            acc_data = pd.read_pickle(filepath)
            acc = 0
            total = np.sum(acc_data.values[:,3])
            for row in acc_data.values:
                acc += row[1]*row[3]/total
            for j, stat in enumerate([acc,model,num_feats,train,test,drug]):
                #print(j,stat)
                #print(master_df.values.shape)
                master_df.values[counter,j] = stat
                master_df.values[counter,6] = acc_data.values[0,4]
            if(i==0):
                title_string = (("{} predictor trained on {}, tested on {}".format(drug, train, test)))
            counter+=1
    print(counter)
    master_df['feats'] = pd.to_numeric(master_df['feats'])
    master_df['acc'] = pd.to_numeric(master_df['acc'])
    master_df['1Dacc'] = pd.to_numeric(master_df['1Dacc'])

    if(figure == '0'):
        drop_counter = 0
        for i, row in enumerate(master_df.values):
            #if(row[1]!='XGB' or row[3]!='public' or row[4]!='aCrossValidation'):
            if(row[3]!='public' or row[4]!='aCrossValidation'):
                master_df = master_df.drop([i])
                drop_counter+=1
        print("{} rows removed from master dataframe".format(drop_counter))
        master_df = master_df.sort_values(by=['feats'])

        sns.set(style="ticks")
        #trying to print them all as one
        grid = sns.FacetGrid(master_df, col = 'drug',hue ="model",hue_order=["XGB","SVM","ANN"], col_wrap = 4, margin_titles=False, legend_out=True)
        #grid = sns.FacetGrid(master_df, col="model", hue ="model", row ="drug",hue_order=["XGB","SVM","ANN"],margin_titles=True)
        grid = (grid.map(plt.plot, "feats", "1Dacc", alpha=1).add_legend())
        #plt.ylim(0.8,1)
        grid.set_ylabels('Accuracy')
        grid.set_xlabels('Number of Features')
        plt.xlim(100,3000)

        #plt.legend(loc='lower right')
        plt.savefig('figures/model_finder_multiface.png')
        plt.show()

    if(figure == '1'):
        print(master_df)
        sys.exit()
        #group = sns.catplot(x='')

    if(figure == '2'):
        simp_df = pd.read_pickle('data/simpsons_diversity_dataframe.pkl')
        simp_df = simp_df.drop(['FIS'])

        # GRDI VS PUBLIC
        ax = sns.lmplot('public', 'grdi', data = simp_df, fit_reg = False)
        plt.xlabel('Simpsons Diversity In The Public Data Set')
        plt.ylabel("Simpsons Diversity In The GRDI Data Set")
        plt.xlim(0,0.75)
        plt.ylim(0,0.75)
        label_points(simp_df.public.values, simp_df.grdi.values, simp_df.index.values, plt.gca())
        plt.savefig('figures/public_grdi_diversity.png', dpi=300)

        # GRDI VS KH
        ax = sns.lmplot('kh', 'grdi', data = simp_df, fit_reg = False)
        plt.xlabel('Simpsons Diversity In The kh Data Set')
        plt.ylabel("Simpsons Diversity In The GRDI Data Set")
        plt.xlim(0,0.75)
        plt.ylim(0,0.75)
        label_points(simp_df.kh.values, simp_df.grdi.values, simp_df.index.values, plt.gca())
        plt.savefig('figures/kh_grdi_diversity.png', dpi=300)

    else:
        print("Did not pass a valid argument")
        usage()
        sys.exit(2)
