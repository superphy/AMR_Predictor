"""
Given a directory of jellyfish outputs, find the union of kmers
"""

import os, sys
from Bio import Seq, SeqIO
import numpy as np
import time
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def parse_fasta(args):
    genome, kmer_length = args
    current_multi_mers = []
    for record in SeqIO.parse("data/genomes/jellyfish_results{}/{}".format(kmer_length, genome), "fasta"):
        kmer_seq = record.seq
        kmer_seq = kmer_seq._get_seq_str_and_check_alphabet(kmer_seq)
        if(len(kmer_seq) == int(kmer_length)):
            current_multi_mers.append(kmer_seq)
    return current_multi_mers

if __name__ == "__main__":
    kmer_length = sys.argv[1]
    out = sys.argv[2]

    # set is designed to only run certain genomes at a time, to prevent maxing RAM
    # set = 0 to run on all sets
    set_num = int(sys.argv[3])

    genomes = ([files for r,d,files in os.walk("data/genomes/jellyfish_results{}/".format(kmer_length))][0])

    if (set_num == 1):
        genomes = np.array(genomes)[np.array([i[:4]=='SRR1' for i in genomes])]
    elif (set_num == 2):
        genomes = np.array(genomes)[np.array([i[:4]=='SRR2' for i in genomes])]
    elif (set_num == 3):
        genomes = np.array(genomes)[np.array([i[:5]=='SRR31' or i[:5]=='SRR32' for i in genomes])]
    elif (set_num == 4):
        genomes = np.array(genomes)[np.array([i[:5]=='SRR36' or i[:5]=='SRR39' for i in genomes])]
    elif (set_num == 5):
        genomes = np.array(genomes)[np.array([i[:4] in ['SRR4', 'SRR5'] or i[:5] in ['SRR30','SRR33','SRR34','SRR35','SRR37'] for i in genomes])]

    assert(len(genomes)>1000)

    print("There are {} genomes in set {}".format(len(genomes), set_num))

    multi_mers = []

    print("Starting parse")
    start = time.time()
    genome_counter = 0
    times_printed = 0
    with ProcessPoolExecutor(max_workers = min(16, cpu_count())) as ppe:
        for current_multi_mer in ppe.map(parse_fasta, zip(genomes,[kmer_length for i in range(len(genomes))])):
            genome_counter+=1
            multi_mers.append(current_multi_mer)
            if(genome_counter>100):
                times_printed += 1
                print("done:", times_printed*100)
                genome_counter -= 100


    print("Parse took {}s, starting unions".format(time.time()-start))
    #start = time.time()
    #master_mers = list(set().union(*multi_mers))
    #print("Union Time:",time.time()-start)
    start = time.time()
    master_mers = np.array(list(set(chain(*multi_mers))))
    bool_mask = [len(master_mers[i]) for i in range(len(master_mers))]
    bool_mask = np.array([i == int(kmer_length) for i in bool_mask])
    master_mers = master_mers[bool_mask]
    print("Chain Time:",time.time()-start)

    print("Found {} unique multi-mers".format(len(master_mers)))

    np.save(out.split('.')[0]+str(set_num)+'.npy', master_mers)
