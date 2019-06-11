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
    "4 - MIC frequency graphs"
    "5 - Individual tests: acc vs feat size for each model, each dataset matchup",
    "6 - Serovar Frequency in the public dataset",
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
        if label in ['CHL', 'NAL']:
            ax.text(x[i]+0.005, y[i]-0.009, label, fontsize = 7)
        else:
            ax.text(x[i]+0.005, y[i]+0.005, label, fontsize = 7)

drugs = ['AMC', 'AMP', 'AZM', 'CHL', 'CIP', 'CRO', 'FIS', 'FOX', 'SXT', 'TET', 'TIO', 'GEN', 'NAL']
names = ['Amoxicillin','Ampicillin','Azithromycin','Chloramphenicol','Ciprofloxacin','Ceftriaxone','Sulfisoxazole',
'Cefoxitin','Trimethoprim/Sulfamethoxazole','Tetracycline','Ceftiofur','Gentamicin','Nalidixic Acid']

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
        multi_df = master_df
        drop_counter = 0
        for i, row in enumerate(multi_df.values):
            #if(row[1]!='XGB' or row[3]!='public' or row[4]!='aCrossValidation'):
            if(row[3]!='public' or row[4]!='aCrossValidation'):
                multi_df = multi_df.drop([i])
                drop_counter+=1
        print("{} rows removed from master dataframe".format(drop_counter))
        multi_df = multi_df.sort_values(by=['feats'])

        multi_df["drug"] = multi_df["drug"].map(dict(zip(drugs,names)))

        sns.set(style="ticks")
        #trying to print them all as one
        grid = sns.FacetGrid(multi_df, col = 'drug',col_order = names,hue ="model",hue_order=["XGB","SVM","ANN"], col_wrap = 4, margin_titles=False, legend_out=True)
        grid = (grid.map(plt.plot, "feats", "1Dacc", alpha=1).add_legend().set_titles("{col_name}"))
        #plt.ylim(0.8,1)
        grid.fig.get_children()[-1].set_bbox_to_anchor((0.9,0.1,0,0))
        plt.setp(grid._legend.get_texts(), fontsize='18')
        plt.setp(grid._legend.get_title(), fontsize='20')
        grid.set_ylabels('Accuracy')
        grid.set_xlabels('Number of Features')
        plt.xlim(100,3000)
        plt.ylim(0.8,1)

        #plt.legend(loc='lower right')
        plt.savefig('figures/model_finder_multiface.png', dpi=600)
        plt.clf()

    if(figure == '1' or figure =='all'):
        # add a new column that contains both the train and test in one column
        sns.set(style="darkgrid")
        master_df['train&test'] = [master_df['train'][i]+'-->'+master_df['test'][i] for i in range(len(master_df.index))]

        # use only XGB models of 1000 features
        XGB_df = master_df['model']=='XGB'
        thousand_df = master_df['feats']==1000
        dataset_df = master_df[XGB_df & thousand_df]

        if not os.path.exists(os.path.abspath(os.path.curdir)+'/figures/dataset_comparisons/'):
            os.mkdir(os.path.abspath(os.path.curdir)+'/figures/dataset_comparisons/')

        group = sns.catplot(x='train&test', y = '1Dacc',hue = "drug",hue_order=drugs, data = dataset_df, kind = 'bar', legend_out=True)
        plt.xlabel('Dataset Comparison')
        plt.ylabel("Accuracy")
        group._legend.set_title('Antimicrobial')
        bar_titles = [
        'Trained on GRDI,\nTested on Public', '\n\n\nTrained on kh,\nTested on GRDI',
        'Trained on GRDI,\nTested on kh','\n\n\nTrained on kh\nwith 5-Fold\nCross Validation',
        'Trained on Public,\nTested on GRDI','\n\n\nTrained on Public\nwith 5-Fold\nCross Validation',
        'Trained on GRDI with\n5-Fold Cross Validation']
        for text, label in zip(group._legend.texts, names):
            text.set_text(label)
        group.set_xticklabels(bar_titles)
        plt.xticks(rotation=0, fontsize = 7, horizontalalignment='center')
        group.fig.get_children()[-1].set_bbox_to_anchor((1.325,0.6,0,0))
        plt.setp(group._legend.get_title(), fontsize='18')
        plt.savefig('figures/dataset_comparisons/dataset_clusters.png', dpi=300, bbox_inches='tight')
        plt.clf()

        cust_pal = ['fire engine red','water blue', 'bright lime','vibrant purple','cyan','strong pink','dark grass green']
        sns.set_palette(sns.xkcd_palette(cust_pal))
        group = sns.catplot(x='drug', y = '1Dacc',hue = "train&test", order=drugs, data = dataset_df, kind = 'bar', legend_out =True)
        plt.xlabel('Antimicrobial')
        plt.ylabel("Accuracy")
        group._legend.set_title('Dataset Comparison')
        bar_titles = [
        'Trained on GRDI,\nTested on Public', 'Trained on kh,\nTested on GRDI',
        'Trained on GRDI,\nTested on kh','Trained on kh with\n5-Fold Cross Validation',
        'Trained on Public,\nTested on GRDI','Trained on Public with\n5-Fold Cross Validation',
        'Trained on GRDI with\n5-Fold Cross Validation']
        for text, label in zip(group._legend.texts, bar_titles):
            text.set_text(label)
        plt.xticks(rotation=-30)
        plt.savefig('figures/dataset_comparisons/drug_clusters.png', dpi=300)
        plt.clf()

    if(figure == '2' or figure =='all'):
        if not os.path.exists(os.path.abspath(os.path.curdir)+'/figures/diversity_plots/'):
            os.mkdir(os.path.abspath(os.path.curdir)+'/figures/diversity_plots/')
        if not os.path.exists(os.path.abspath(os.path.curdir)+'/data/simpsons_diversity_dataframe.pkl'):
            raise Exception("Please run diversity.py before calling this graph")

        simp_df = pd.read_pickle('data/simpsons_diversity_dataframe.pkl')
        simp_df = simp_df.drop(['FIS'])

        for dataset in ['public', 'kh']:
            ax = sns.lmplot(dataset, 'grdi', data = simp_df, fit_reg = False)
            plt.xlabel('Simpsons Diversity In The '+dataset.capitalize()+' Data Set')
            plt.ylabel("Simpsons Diversity In The GRDI Data Set")
            plt.xlim(0,0.75)
            plt.ylim(0,0.75)
            if(dataset=='public'):
                label_points(simp_df.public.values, simp_df.grdi.values, simp_df.index.values, plt.gca())
            else:
                label_points(simp_df.kh.values, simp_df.grdi.values, simp_df.index.values, plt.gca())
            plt.savefig('figures/diversity_plots/'+dataset+'_grdi_diversity.png', dpi=300)
            plt.clf()


    if(figure in ['3','4','all']):
        if(figure != '4'):
            if not os.path.exists(os.path.abspath(os.path.curdir)+'/figures/accuracies_and_frequencies/'):
                os.mkdir(os.path.abspath(os.path.curdir)+'/figures/accuracies_and_frequencies/')
        if(figure != '3'):
            if not os.path.exists(os.path.abspath(os.path.curdir)+'/figures/frequencies/'):
                os.mkdir(os.path.abspath(os.path.curdir)+'/figures/frequencies/')

        if(figure == 'all'):
            figures = ['3','4']
        else:
            figures = [figure]

        for figure in figures:
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
                    plt.ylabel('Number of Genomes')

                    if(figure == '3'):
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
                    if(figure == '4'):
                        # Append the frequencies to the tops of the bars
                        for i, p in enumerate(ax.patches):
                            if i >= len(single_df.values[:,0]+1):
                                break
                            height = p.get_height()
                            count = single_df.values[i,3]
                            if height == 0:
                                continue
                            ax.text(p.get_x() + p.get_width()/2., p.get_height() + 3, str(int(count)) , ha="center")
                        ax.set_title(dataset+' '+drug+' MIC Frequencies')
                        plt.savefig('figures/frequencies/'+dataset+'_'+drug+'.png',dpi=300)
                        plt.clf()

    if(figure == '5' or figure =='all'):
        if not os.path.exists(os.path.abspath(os.path.curdir)+'/figures/individual_tests/'):
            os.mkdir(os.path.abspath(os.path.curdir)+'/figures/individual_tests/')
        for direct in ([dirs for r,dirs,f in os.walk("results")][0]):
            splits = direct.split('_')

            # 2 splits means we are cross validating
            if(len(splits)==2):
                train = splits[0][0:-1]
                drug = splits[1]
                test = 'aCrossValidation'

            # 3 splits means we are testing across datasets
            else:
                train = splits[0][0:-1]
                test = splits[1]
                drug = splits[2]

            if(drug == 'FIS' and (train == 'grdi' or test == 'grdi')):
                continue

            # filter the relevant parts from the dataframe
            train_df = master_df['train']==train
            test_df = master_df['test']==test
            drug_df = master_df['drug']==drug
            split_df = master_df[train_df & test_df & drug_df]

            title_string = (("{} predictor trained on {}, tested on {}".format(drug, train, test)))
            sns.set(style="whitegrid")

            base = sns.relplot(x="feats", y="acc", hue="model",hue_order=["XGB","SVM","ANN"], kind="line", data=split_df)
            plt.rcParams["axes.titlesize"] = 8
            plt.title(title_string + ' (Direct)')
            plt.ylim(0,1)
            plt.xlim(0,10000)
            plt.xlabel('Number of Features')
            plt.ylabel('Accuracy')
            plt.savefig('figures/individual_tests/0D_'+(title_string.replace(" ",""))+'.png',dpi=300)
            plt.clf()


            window = sns.relplot(x="feats", y="1Dacc", hue="model",hue_order=["XGB","SVM","ANN"], kind="line", data=split_df)
            plt.rcParams["axes.titlesize"] = 8
            plt.title(title_string + ' (1-Dilution)')
            plt.ylim(0,1)
            plt.xlim(0,10000)
            plt.xlabel('Number of Features')
            plt.ylabel('Accuracy')
            plt.savefig('figures/individual_tests/1D_'+(title_string.replace(" ",""))+'.png',dpi=300)
            plt.clf()

    if(figure == '6' or figure == 'all'):
        amr_master_df = pd.read_excel("data/no_ecoli_GenotypicAMR_Master.xlsx")
        serovar_counts = amr_master_df['serovar'].value_counts()
        serovar_counts = serovar_counts[:50,]
        #spt = sns.countplot(x='serovar', data=amr_master_df, order = amr_master_df['serovar'].value_counts().index)
        i1_indx = 100
        for i, val in enumerate(serovar_counts.index):
            if val == 'I 1':
                i1_indx = i
        assert(i1_indx!=100)
        spt = sns.barplot([x for i,x in enumerate(serovar_counts.index) if i!=i1_indx], [x for i,x in enumerate(serovar_counts.values) if i!=i1_indx])
        plt.xticks(rotation=90, fontsize=10)
        plt.tight_layout()
        plt.savefig('figures/serovar_frequencies.png', dpi = 300)
        plt.clf()

    if(figure not in ['0','1','2','3','4','5', '6','all']):
        print("Did not pass a valid argument")
        usage()
        sys.exit(2)
