"""
Takes as input a hit_df and groups it for readability in a csv
"""
def merge_df(df_path, drug, dataset):
    """
    Takes path to pandas df with the columns:
    [kmer, gene_up, dist_up, gene_down, dist_down, start, stop, genome_id, contig_name]
    and returns a df with the columns:
    [drug, dataset, kmer, gene_up, gene_up_count, avg_dist_up, gene_down, gene_down_count, avg_dist_down]
    """
    df = pd.read_pickle(df_path)
    hit_summary = []

    for kmer in set(df['kmer']):
        # filter for only a single kmer
        kmer_df = df[df['kmer']==kmer]

        for gene_direc, gene_dist in [['gene_up','dist_up'],['gene_down','dist_down']]:
            for gene in set(kmer_df[gene_direc]):
                if(len(gene)==0):
                    continue
                # filter for only a single gene
                gene_df = kmer_df[kmer_df[gene_direc]==gene]

                total_dist = 0

                for dist in gene_df[gene_dist]:
                        total_dist += abs(float(dist))

                count = gene_df.shape[0]
                average_dist = total_dist/count

                hit_summary.append([dataset, drug, kmer, gene, count, average_dist])

    return hit_summary



if __name__ == "__main__":
    import os, sys
    import pandas as pd
    import numpy as np

    drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]


    dataset = sys.argv[1]
    output = sys.argv[2]

    all_hits = []

    for drug in drugs:
        if(dataset == 'grdi' and drug in ['FIS','AZM']):
            continue
        df_path = "annotation/search/11mer_data/{}_hits_for_{}.pkl".format(dataset,drug)
        drug_list = merge_df(df_path, drug, dataset)
        for hit in drug_list:
            all_hits.append(hit)

    all_df = pd.DataFrame(data=np.asarray(all_hits),columns=['dataset','drug','kmer','gene','count','average_dist'])

    all_df.to_csv(output)
