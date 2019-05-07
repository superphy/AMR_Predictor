import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os, sys
import glob
import pickle

def usage():
    print("\nusage: python all_data_figures all\n\nOptions:")
    print(
    "Replace 'all' with a specific number to generate just that set",
    "0 - Multiface accuracy vs features, all drugs all models",
    "1 - Multibar graph, 13 sets of 7 bars detailing 13 drug accuracies in each of 7 dataset comparisons",
    "2 - Simpsons Diversity Plots",
    "3 - MIC frequency & accuracy bar graphs",
    sep = '\n')
    return

def strip_tail(mic_str):
    # Removing tailing zeros from MIC values and the . if its the last char
    mic_str = mic_str.rstrip('0')
    if(mic_str[-1] =='.'):
        return mic_str[:-1]
    return mic_str

def label_points(x, y, labels, ax):
    for i, label in enumerate(labels):
        if label == 'CHL':
            ax.text(x[i]+0.005, y[i]-0.005, label, fontsize = 7)
        else:
            ax.text(x[i]+0.005, y[i]+0.005, label, fontsize = 7)

drugs = ['AMC', 'AMP', 'AZM', 'CHL', 'CIP', 'CRO', 'FIS', 'FOX', 'SXT', 'TET', 'TIO', 'GEN', 'NAL']

if __name__ == "__main__":
    if not os.path.exists(os.path.abspath(os.path.curdir)+'/figures'):
        os.mkdir(os.path.abspath(os.path.curdir)+'/figures')
    figure = 0
    try:
        figure = sys.argv[1]
    except:
        usage()
        sys.exit(2)

    if not os.path.exists(os.path.abspath(os.path.curdir)+'/results'):
        raise Exception("You need to execute run_tests.smk first")
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

    if(figure == '0' or figure =='all'):
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
        plt.clf()

    elif(figure == '1' or figure =='all'):
        print(master_df)
        sys.exit()
        #group = sns.catplot(x='')

    elif(figure == '2' or figure =='all'):
        if not os.path.exists(os.path.abspath(os.path.curdir)+'/figures/diversity_plots/'):
            os.mkdir(os.path.abspath(os.path.curdir)+'/figures/diversity_plots/')
        if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/simpsons_diversity_dataframe.pkl'):
            raise Exception("Please run diversity.py before calling this graph")

        simp_df = pd.read_pickle('data/simpsons_diversity_dataframe.pkl')
        simp_df = simp_df.drop(['FIS'])

        for dataset in ['public', 'kh']:
            ax = sns.lmplot(dataset, 'grdi', data = simp_df, fit_reg = False)
            plt.xlabel('Simpsons Diversity In The '+dataset+' Data Set')
            plt.ylabel("Simpsons Diversity In The grdi Data Set")
            plt.xlim(0,0.75)
            plt.ylim(0,0.75)
            if(dataset=='public'):
                label_points(simp_df.public.values, simp_df.grdi.values, simp_df.index.values, plt.gca())
            else:
                label_points(simp_df.kh.values, simp_df.grdi.values, simp_df.index.values, plt.gca())
            plt.savefig('figures/diversity_plots/'+dataset+'_grdi_diversity.png', dpi=300)
            plt.clf()


    elif(figure == '3' or figure =='all'):
        if not os.path.exists(os.path.abspath(os.path.curdir)+'/figures/accuracies_and_frequencies/'):
            os.mkdir(os.path.abspath(os.path.curdir)+'/figures/accuracies_and_frequencies/')

        #model_df = master_df['model'] == 'XGB'
        #feats_df = master_df['feats'] == 1000
        #test_df = master_df['test'] == 'aCrossValidation'
        #current_df = master_df[ model_df & feats_df & test_df]
        #print(current_df)
        #sys.exit()

        for set_number, dataset in enumerate(['public', 'grdi', 'kh']):
            if dataset =='grdi':
                with open('data/grdi_mic_class_order_dict.pkl','rb') as fh:
                    class_dict = pickle.load(fh)
            else:
                with open('data/public_mic_class_order_dict.pkl','rb') as fh:
                    class_dict = pickle.load(fh)

            for drug in drugs:
                if(dataset=='grdi' and drug == 'FIS'):
                    continue
                classes = [strip_tail(i) for i in class_dict[drug]]
                with open('results/'+dataset+str(set_number+1)+'_'+drug+'/'+drug+'_1000feats_XGBtrainedOn'+dataset+'_testedOnaCrossValidation.pkl','rb') as fh:
                    single_df = pickle.load(fh)
                ax = sns.barplot(y='Supports', x=classes, data = single_df)
                plt.xlabel('Minimum Inhibitory Concentration (mg/L)')
                plt.ylabel('No. Genomes')

                # Append the accuracies to the tops of the bars
                for i, p in enumerate(ax.patches):
                    if i >= len(single_df.values[:,0]+1):
                        break
                    height = p.get_height()
                    acc1d = float('{:1.2f}'.format(single_df.values[i,1]))
                    if height == 0:
                        continue
                    ax.text(p.get_x() + p.get_width()/2., p.get_height() + 3, str(int(acc1d*100))+'%' , ha="center")
                ax.set_title(dataset+' '+drug+' MIC Frequencies')
                plt.savefig('figures/accuracies_and_frequencies/'+dataset+'_'+drug+'.png',dpi=300)
                plt.clf()


    else:
        print("Did not pass a valid argument")
        usage()
        sys.exit(2)
