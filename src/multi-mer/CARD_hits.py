"""
Compares kmers against card database through api
instead of comparing proteins across databases
"""
import numpy as np
import pandas as pd
import os,sys
from Bio import SeqIO

drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
datasets=["public","grdi"]

def rewrite_fastas(input, output):
    '''
    puts species name into contig header
    '''
    for record in SeqIO.parse(input,"fasta"):
        print(record.id)
        contig, species = record.id.split('[')
        record.id =(contig[:-1]+'|'+species[:-1]).replace(' ','_')
        SeqIO.write(record, output, "fasta")

def strip_amg_hits(df):
    '''
    takes best_hits.csv and returns only top hits overall, not just AMG's
    '''
    df = df.tail(24)
    df = df.reset_index(drop=True)
    df = df.drop(columns = ['Unnamed: 0', 'AMG'])
    return df

def make_blast_query(df, save_loc):
    '''
    Builds blast query using dataset and drug to build contig headers
    and kmer sequences to search off of
    '''
    with open(save_loc,'a') as fh:
        for i in df.index.values:
            fh.write(">{}_{}\n".format(df['dataset'][i],df['drug'][i]))
            fh.write(df['kmer'][i]+'\n')

def find_top_gene(df):
    '''
    Takes pandas DataFrame and returns the top gene for each dataset/drug combo
    '''
    card_hits = []
    for dataset in datasets:
        for drug in drugs:
            if dataset == 'grdi' and drug in ['AZM','FIS']:
                continue
            set_df = df[df['qseqid']=="{}_{}".format(dataset,drug)]
            if len(set_df['stitle'])>0:
                card_hits.append(set_df[['stitle','pident','length','bitscore','evalue']].iloc[0])
                #print('appending')
                continue
            #print("No Salmonella gene for {} {}".format(dataset,drug))
            card_hits.append('0')
    return card_hits

def card_column(header, card_series):
    '''
    Builds a column for the specific header for entry into a pandas df
    '''
    hits = []
    for hit in card_series:
        if isinstance(hit, str):
            hits.append('')
        else:
            hits.append(hit[header])
    if header == 'stitle':
        hits = [i.split('|')[-1] for i in hits]
    return hits

def public_to_ncbi(dataset):
    if dataset == 'public':
        return 'NCBI'
    else:
        return 'GRDI'

if __name__ == "__main__":
    kmer_length = sys.argv[1]
    all_hits_df = pd.read_csv("results/multi-mer/{}mer/best_hits.csv".format(kmer_length))

    # removed AMG hits
    general_hits_df = strip_amg_hits(all_hits_df)

    # build blast query
    if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/CARD/top_{}mer_hits_search_query.fasta".format(kmer_length)):
        make_blast_query(general_hits_df, "data/CARD/top_{}mer_hits_search_query.fasta".format(kmer_length))

    os.system("blastx -db data/CARD/CARDdb -query data/CARD/top_"+kmer_length+"mer_hits_search_query.fasta -outfmt '6 qseqid stitle pident length mismatch gapopen qstart qend sstart send evalue bitscore' -out data/CARD/all_CARD_hits.tsv")

    cols = ['qseqid' ,'stitle' ,'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend' ,'sstart', 'send', 'evalue', 'bitscore']
    card_hits_df = pd.read_csv("data/CARD/all_CARD_hits.tsv",delimiter = '\t', names = cols)

    #print(CARD_hits_df)

    card_series = find_top_gene(card_hits_df)

    for header in ['stitle','pident','length','bitscore','evalue']:
        general_hits_df[header] = card_column(header, card_series)

    general_hits_df['dataset'] = pd.Series([public_to_ncbi(i) for i in general_hits_df['dataset']])

    general_hits_df.to_csv("results/multi-mer/{}mer/top_hit_with_card.csv".format(kmer_length))
