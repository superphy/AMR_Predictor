"""
Finds all samples matching a specific organism,
downloads all antimicrobial metadata
"""
#esearch -db biosample -query SAMN03988375 | efetch -mode xml

import pandas as pd
import numpy as np
import os, sys
from Bio import Entrez
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

path = "data/Entrez/"
pathogen = 'Salmonella'
query = "antibiogram[filter] AND {}[organism]".format(pathogen)
Entrez.email = 'rylanj.steinkey@gmail.com'

"""
if we have a dataframe with biosamples and SRR run ids, we can add a columns of runs
I have mine at https://github.com/superphy/AMR_Predictor/blob/master/data/no_ecoli_GenotypicAMR_Master.xlsx
but just point the line below at yours and make sure it has columns 'run' and 'biosample'
Otherwise this document will only return antibiogram data with biosample ID's
"""

run_excel_path = "data/no_ecoli_GenotypicAMR_Master.xlsx"


antimicrobials = ['amoxicillin-clavulanic acid', 'ampicillin', 'azithromycin',
'cefoxitin', 'ceftiofur', 'ceftriaxone', 'chloramphenicol', 'ciprofloxacin',
'gentamicin', 'nalidixic acid', 'streptomycin', 'sulfisoxazole', 'tetracycline',
'trimethoprim-sulfamethoxazole','kanamycin']

mics = ['AMC','AMP','AZM','FOX','TIO','CRO','CHL','CIP','GEN','NAL','STR','FIS',
'TET','SXT','KAN']

other_available = []

"""
Generate list of samples for wildcards prior to snakemake start
"""
def query_to_ids(query):
    """
    input: NCBI query string
    output: List of ids from the biosample database
    """
    handle = Entrez.esearch(db='biosample', retmax = 10000, term = query)
    records = Entrez.read(handle)
    handle.close()
    return records['IdList']

def id_to_mic(sample):
    """
    input: id to a BioSample
    output: dictionary of metadata
    """

    info = {}
    headers = []
    mic_rows = []
    for i, name in enumerate(sample):

        # BioSample Acc id
        if i == 0:
            info['BioSample'] = name[0].text

        # antibiogram table
        if i == 1:
            # Description
            try:
                for header in name[2][0][1]:
                    headers.append(header.text)

                for row in name[2][0][2]:
                    mic_rows.append([i.text for i in row])
            except:
                return 'skip'

        # supplementary information such as serovar
        if i == 5:
            # Attributes
            for attribute in name:
                info[attribute.attrib['attribute_name']]=attribute.text

    for drug in mic_rows:
        if drug[0] not in antimicrobials:
            # instead of returning the invalid drug name, it could be appended
            # and the full name used as the 3 letter code
            continue
        mic_3l = mics[antimicrobials.index(drug[0])]
        info["SIR_{}".format(mic_3l)] = drug[1]
        info["MIC_{}".format(mic_3l)] = drug[2]+' '+drug[3].split('/')[0]

        if "Typing Method" in info:
            # check for consistency in typing method for a sample
            if info["Typing Method"] != drug[8]:
                # method inconsistent, set to blank
                info["Typing Method"] = ' '
        if "Testing Standard" in info:
            # check for consistency in testing standard for a sample
            if info["Testing Standard"] != drug[9]:
                # standard inconsistent, set to blank
                info["Testing Standard"] = ' '

        info["Typing Method"] = drug[8]
        info["Testing Standard"] = drug[9]

    return info

def amr_row(info_dict, df_columns):
    """
    If value isnt found, set it to empty
    """
    vals = []
    for i in df_columns:
        try:
            vals.append(info_dict[i])
        except:
            vals.append(' ')
    return vals

try:
    ids = list(np.load(path+'ids.npy'))
except:
    ids = list(query_to_ids(query))
    if not os.path.isdir(path):
        os.makedirs(path)
    np.save(path+'ids.npy', ids)

handle = Entrez.efetch(db='biosample', id = ids)
data = handle.read()
handle.close()
root = ET.fromstring(data)

df_columns = ['BioSample']+['MIC_'+i for i in mics]+['SIR_'+i for i in mics]+[
'isolation_source', 'serovar', 'collection_date', 'collected_by',
'geo_loc_name', 'strain', 'sub_species', 'Typing Method', 'Testing Standard']

mic_data = []

with ProcessPoolExecutor(max_workers = cpu_count()-1) as ppe:
    for info_dict in ppe.map(id_to_mic, root):
        if isinstance(info_dict, str):
            if info_dict != 'skip':
                other_available.append(info_dict)
        else:
            row = amr_row(info_dict, df_columns)
            mic_data.append(amr_row(info_dict, df_columns))

if len(other_available) > 0:
    print("The following antimicrobials had data but their 3 letter codes were not declared:")
    print(set(other_available))
    print("To use these, append their names and 3 letter codes to the lists on lines 30 and 35.")

amr_df = pd.DataFrame(data = mic_data, columns = df_columns)

try:
    run_df = pd.read_excel(run_excel_path)

except:
    print('To add a column of runs alongside biosample, change run_excel_path')

else:
    old_runs = list(run_df['run'])
    old_biosamples = list(run_df['biosample'])
    runs = []
    for i in amr_df['BioSample']:
        try:
            runs.append(old_runs[old_biosamples.index(i)])
        except:
            runs.append(' ')
    amr_df.insert(loc = 0, column = "run", value = runs)

print(amr_df)
amr_df.to_csv("data/dec_2019_antibiogram.csv")
