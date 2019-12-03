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

antimicrobials = ['amoxicillin-clavulanic acid', 'ampicillin', 'azithromycin',
'cefoxitin', 'ceftiofur', 'ceftriaxone', 'chloramphenicol', 'ciprofloxacin',
'gentamicin', 'nalidixic acid', 'streptomycin', 'sulfisoxazole', 'tetracycline',
'trimethoprim-sulfamethoxazole']

mics = ['AMC','AMP','AZM','FOX','TIO','CRO','CHL','CIP','GEN','NAL','STR','FIS',
'TET','SXT']


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
            for header in name[2][0][1]:
                headers.append(header.text)

            for row in name[2][0][2]:
                mic_rows.append([i.text for i in row])

        # supplementary information such as serovar
        if i == 5:
            # Attributes
            for attribute in name:
                info[attribute.attrib['attribute_name']]=attribute.text

    for drug in mic_rows:
        mic_3l = mics[antimicrobials.index(drug[0])]
        info["{}_SIR".format(mic_3l)] = drug[1]
        info["{}_MIC".format(mic_3l)] = drug[2]+' '+drug[3].split('/')[0]
        info["Typing Method"] = drug[8]
        info["Testing Standard"] = drug[9]


    return info

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

df_columns = ['BioSample']+[i+'_MIC' for i in mics]+[i+'_SIR' for i in mics]+[
'isolation_source', 'serovar', 'collection_date', 'collected_by',
'geo_loc_name', 'strain', 'sub_species', 'Typing Method', 'Testing Standard']

mic_data = []

with ProcessPoolExecutor(max_workers = cpu_count()-1) as ppe:
    for info_dict in ppe.map(id_to_mic, root):
        mic_data.append([info_dict[i] for i in df_columns])

amr_df = pd.DataFrame(data = mic_data, columns = df_columns)

amr_df.to_csv("data/dec_2019_antibiogram.csv")
