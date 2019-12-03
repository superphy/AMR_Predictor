ids, = glob_wildcards("genomes{id}.fasta")
drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FOX","GEN","NAL","SXT","TET","TIO"]

def find_nearest(mic, drug_arr):
    """
    Takes in a mic and if it doesnt match a predefined value, it will find
    the nearest mic (within a certain range)
    Can be useful for different rounding schemes
    """
    mic = float(mic)
    # removing prepending symbols, then converting to float
    ns_drug_arr = [i for i in drug_arr]
    ns_drug_arr[0] = float(ns_drug_arr[0].split('=')[1])
    ns_drug_arr[-1] = float(ns_drug_arr[-1].split('=')[1])

    for i,val in enumerate(ns_drug_arr):
        val = float(val)
        if abs(mic/val-1)<0.1:
            return drug_arr[i]
    if(mic<ns_drug_arr[0]):
        return drug_arr[0]
    elif(mic>ns_drug_arr[-1]):
        return drug_arr[-1]
    else:
        raise Exception("{} not found in range {}".format(mic,drug_arr))

def off_by_one(pred,act,drug_arr):
    """
    Are these 2 MIC within 1 dilution?
    """
    p_loc = drug_arr.index(pred)
    try:
        a_loc = drug_arr.index(act)
    except:
        print('value: {} is of type: {}'.format(act,type(act)))
        raise

    if(p_loc == a_loc):
        raise Exception("Predicted value equal to actual, should not be classified as an error")

    if abs(p_loc-a_loc) == 1:
        return True
    else:
        return False

def patric_to_superphy(mic):
    """
    Changes 3 charecter drug codes to match current implementation
    """
    if mic == 'AUG':
        return 'AMC'
    elif mic == 'AXO':
        return 'CRO'
    elif mic == 'AZI':
        return 'AZM'
    elif mic == 'COT':
        return 'SXT'
    else:
        return mic

def patric_supp_conv(df):
    """
    Converts the supplemental table provided into one
    compatable with our binning script
    """
    our_mic = ['AMP','AMC','CRO','AZM','CHL','CIP','SXT','FIS','FOX','GEN','NAL','TET','TIO']
    pat_mic = ['AMP','AUG','AXO','AZI','CHL','CIP','COT','FIS','FOX','GEN','NAL','TET','TIO']

    new_df = pd.DataFrame()

    for drug_num in range(len(our_mic)):
        new_df['MIC_'+our_mic[drug_num]] = df[df['Antibiotic']==pat_mic[drug_num]].set_index('SRA Run Accession')['Laboratory-derived MIC']

    return df

def compare_mic_df(ldf, rdf):
    """
    checks for data in the left(patric) and right(ncbi) that are different
    """
    correct = 0
    wrong = 0
    has_nan = 0
    invalid = 0

    errors = []

    for drug in drugs:
        ldfd = ldf[drug]
        rdfd = rdf[drug]
        for sample in ldfd.index.values:
            if sample not in rdfd.index.values:
                continue
            if pd.isnull(ldfd[sample]) or pd.isnull(rdfd[sample]):
                has_nan +=1
            elif(ldfd[sample] == rdfd[sample]):
                correct +=1
            else:
                if(rdfd[sample]=='invalid'):
                    invalid+=1
                    continue
                wrong +=1
                #print("{} {} are not equal ({} {})".format(ldfd[sample], rdfd[sample], drug, sample))
                errors.append([sample,drug,ldfd[sample], rdfd[sample]])

    df = pd.DataFrame(data = errors, columns=['id','antimicrobial','patric','NCBI'])

    print("Correct: {} Wrong: {} NaN/Invalid: {}".format(correct,wrong,has_nan+invalid))

    return df

def find_major(pred, act, drug):
	pred = (str(pred).split("=")[-1])
	pred = ((pred).split(">")[-1])
	pred = ((pred).split("<")[-1])
	pred = int(round(float(pred)))
	act = (str(act).split("=")[-1])
	act = ((act).split(">")[-1])
	act = ((act).split("<")[-1])
	act = int(round(float(act)))

	if(drug =='AMC' or drug == 'AMP' or drug =='CHL' or drug =='FOX'):
		susc = 8
		resist = 32
	if(drug == 'AZM' or drug == 'NAL'):
		susc = 16
	if(drug == 'CIP'):
		susc = 0.06
		resist = 1
	if(drug == 'CRO'):
		susc = 1
	if(drug == 'FIS'):
		susc = 256
		resist = 512
	if(drug == 'GEN' or drug =='TET'):
		susc = 4
		resist = 16
	if(drug == 'SXT' or drug =='TIO'):
		susc = 2

	if(drug == 'AZM' or drug == 'NAL'):
		resist = 32
	if(drug == 'CRO' or drug == 'SXT'):
		resist = 4
	if(drug == 'TIO'):
		resist = 8

	if(pred <= susc and act >= resist):
		return "VeryMajorError"
	if(pred >= resist and act <= susc):
		return "MajorError"
	return "NonMajor"

rule all:
    input:
        "results.csv"

# make predictions
rule predict:
    input:
        "genomes/{id}.fasta"
    output:
        "predictions/{id}.tsv"
        #temp("temps/{id}/{id}.fasta.kmc.stdout")
    threads:
        1
    shell:
        "mkdir temps/{wildcards.id}/ && bash mic_prediction_fasta.sh {input} temps/{wildcards.id} data_files/SALM.model.pkl {threads} {output} data_files/SALM.ArrInds data_files/antibioticsList_SALM.uniq data_files/SALM_MICMethods data_files/all_kmrs"

# print to same output style as current evaluator
rule format:
    input:
        expand("predictions/{id}.tsv", id = ids)
    output:
        "prediction_errors.txt"
    run:
        import pandas as pd
        import numpy as np
        import os, sys
        import yaml
        import math
        from sklearn.externals import joblib

        act_df = joblib.load("data_files/public_mic_class_dataframe.pkl")

        with open("data_files/class_ranges.yaml",'r') as infh:
            mic_class_labels = yaml.safe_load(infh)

        for i in input:
            df = pd.read_csv(i, delimiter='\t',skiprows=2)
            run_id = i.split("/")[-1].split('.')[0]

            # filter to keep only MIC's we have the data to validate
            df['Antibiotic'] = pd.Series([patric_to_superphy(i) for i in df['Antibiotic']])
            df = df.loc[df['Antibiotic'].isin(drugs)]

            # loop through predictions and save their results
            for i in df.index.values:
                drug = df['Antibiotic'][i]
                prediction = find_nearest(df['Prediction'][i],mic_class_labels[drug])
                actual = act_df[drug][run_id]

                # skip predictions that we dont have a value for
                if actual in ['invalid','nan']:
                    continue
                if isinstance(actual,float):
                    if math.isnan(actual):
                        continue

                if(prediction == actual):
                    with open('prediction_errors.txt', 'a') as myfile:
                        myfile.write("\nCorrect:"+drug)
                    continue

                obo = off_by_one(prediction, actual, mic_class_labels[drug])
                major = find_major(prediction, actual, drug)

                #print("\nDrug:{} Genome:{} Predicted:{} Actual:{} OBO:{} Major?:{}".format(
                #drug, run_id, prediction, actual, obo, major))

                with open('prediction_errors.txt', 'a') as myfile:
                    myfile.write("\nDrug:{} Genome:{} Predicted:{} Actual:{} OBO:{} Major?:{}".format(
                    drug, run_id, prediction, actual, obo, major))

rule evaluate:
    input:
        "prediction_errors.txt"
    output:
        'results.csv'
    run:
        import numpy as np
        import pandas as pd
        # create a dataframe for the results to be stored in as we read them
        results_df = pd.DataFrame(data = np.zeros((len(drugs),6)),index = drugs,columns=['Accuracy (1D)','Accuracy (Direct)',
        'Total Predictions','Non-Major Error Rate','Major Error Rate','Very Major Error Rate'])

        # we are going to walk line by line classifying results
        with open("prediction_errors.txt") as file:
            for line_num, line in enumerate(file):

                # looking at the first 5 characters we can see if the prediction was correct or if it was an error
                first_word = line[:5]

                # if it was an error we need to collect what type it was
                if(first_word=='Drug:'):

                    # this is the ordering of how the errors are written to the file
                    drug, genome, pred, act, off_by_one, major = line.split(' ')

                    # we need to split off all the english to just get the keyvalue
                    drug = drug.split(':')[1]
                    off_by_one = off_by_one.split(':')[1]
                    major = major.split(':')[1]

                    # lines end with a \n that we need to remove
                    major = major.rstrip()

                    # based on what we find we need to increment the correct value in the dataframe, we will convert to % later
                    results_df.at[drug,'Total Predictions']+=1
                    if(off_by_one=='True'):
                        results_df.at[drug,'Accuracy (1D)']+=1
                    if(major=='NonMajor'):
                        results_df.at[drug,'Non-Major Error Rate']+=1
                    elif(major=='MajorError'):
                        results_df.at[drug,'Major Error Rate']+=1
                    elif(major=='VeryMajorError'):
                        results_df.at[drug,'Very Major Error Rate']+=1
                    else:
                        raise Exception("Major rate: {} not able to be properly classified".format(major))

                # if the prediction was correct we simply increment both accuracies
                elif(first_word=='Corre'):
                    drug = line.split(':')[1]
                    drug = drug.rstrip()
                    results_df.at[drug,'Total Predictions']+=1
                    results_df.at[drug,'Accuracy (1D)']+=1
                    results_df.at[drug,'Accuracy (Direct)']+=1

                elif(first_word=='\n'):
                    # this is intentionally left empty to skip newline chars
                    continue

                else:
                    raise Exception("An unexpected {} was found in prediction_errors on line {}".format(first_word,line_num+1))

        # now we want to generate rates from raw numbers
        for drug in drugs:
            # for accuracies we divide raw counts by total predictions
            total = results_df['Total Predictions'][drug]
            results_df.at[drug,'Accuracy (1D)'] = results_df['Accuracy (1D)'][drug] / total
            results_df.at[drug,'Accuracy (Direct)'] = results_df['Accuracy (Direct)'][drug] / total

            # for errors we divide raw counts by total predictions
            num_errors = 0
            for error_type in ['Non-Major Error Rate','Major Error Rate','Very Major Error Rate']:
                num_errors+= results_df[error_type][drug]
            results_df.at[drug,'Non-Major Error Rate'] = results_df['Non-Major Error Rate'][drug] / total
            results_df.at[drug,'Major Error Rate'] = results_df['Major Error Rate'][drug] / total
            results_df.at[drug,'Very Major Error Rate'] = results_df['Very Major Error Rate'][drug] / total

        results_df.to_csv("results.csv")
