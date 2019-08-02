"""
This takes in a CARD database tsv with protein accession numbers
and adds in the gene codes for comparasion with prokka output
"""
from bioservices import UniProt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os, sys

def find_gene(prot_id):
    u = UniProt(verbose=False)
    res = u.mapping("EMBL", "ACC", query=prot_id)
    for key, values in res.items():
        for value in values:
            res = u.search(value, frmt="tab", limit=3, columns="genes", database='uniparc')

            genes = set(res[11:].split(';'))
            genes = [i for i in genes if (0<len(i) and i !='\n')]

            if len(genes)<1:
                genes = 'none'

            return key, genes
    return prot_id, 'none'

if __name__ == "__main__":
    aro_data = pd.read_csv("data/aro_categories_index.tsv", sep = '\t')
    prot_ids = [ i for i in aro_data['Protein Accession']]

    gene_ids = {}
    print('0',len(prot_ids)) #2594

    with ProcessPoolExecutor(max_workers=cpu_count()-1) as ppe:
        for key, genes in ppe.map(find_gene, prot_ids):
            print("finished {}".format(key))
            gene_ids[key] = genes

    print('1',len(gene_ids)) #2588

    ordered_ids = []

    for id in prot_ids:
        try:
            ordered_ids.append(gene_ids[id])
        except:
            ordered_ids.append(["NotFound"])

    print(aro_data.shape)
    print(len(ordered_ids))


    #gene_ids = [gene_ids[i] for i in prot_ids]

    aro_data['gene_codes'] = pd.Series(ordered_ids)

    #aro_data = pd.Dataframe(data = (aro_data.values))

    print(aro_data)

    aro_data.to_pickle("data/gene_labels.csv")
