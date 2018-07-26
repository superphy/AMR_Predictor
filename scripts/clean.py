#!/usr/bin/env python

import sys
import os
import re
from Bio import SeqIO

def get_files_to_analyze(file_or_directory):
    """
    :param file_or_directory: name of either file or directory of files
    :return: a list of all files (even if there is only one)
    """

    # This will be the list we send back
    files_list = []

    if os.path.isdir(file_or_directory):
        # Using a loop, we will go through every file in every directory
        # within the specified directory and create a list of the file names
        for root, dirs, files in os.walk(file_or_directory):
            for filename in files:
                files_list.append(os.path.join(root, filename))
    else:
        # We will just use the file given by the user
        files_list.append(os.path.abspath(file_or_directory))

    # We will sort the list when we send it back, so it always returns the
    # same order
    return sorted(files_list)

def find_recurring_char(record, start, end):
    length = len(record)
    window = end - start
    recur_char = 'X'
    #this is the density threshold for finding a garbage sequence
    #0.8 means that 80% of nucleotides must be the same to trigger a cut
    perc_cut = 0.75
    A_count = (record).count("A",start,end)
    T_count = (record).count("T",start,end)
    G_count = (record).count("G",start,end)
    C_count = (record).count("C",start,end)
    N_count = (record).count("N",start,end)
    if(((A_count)/window)>=perc_cut):
        recur_char = 'A'
    elif(((T_count)/window)>=perc_cut):
        recur_char = 'T'
    elif(((G_count)/window)>=perc_cut):
        recur_char = 'G'
    elif(((C_count)/window)>=perc_cut):
        recur_char = 'C'
    elif(((N_count)/window)>=perc_cut):
        recur_char = 'N'
    return recur_char

def format_files(files_list, output_dir):
    """
    Print to a new directory the re-formatted fasta files.
    We will remove anything smaller than 500bp, under 5x coverage,
    and any sequences of repeating or near repeating bases at the
    beginning or end of a contig.
    :param files_list: list of fasta files to format headers
    :return: success
    """
    max_score=25
    for f in files_list:
        file_name = f
        with open(os.path.join(output_dir, file_name.split('/')[-1]), "w") as oh:
            contig_number = 1
            with open(f, "r") as fh:
                for record in SeqIO.parse(fh, "fasta"):
                    if len(record.seq) < 500:
                        #print("Skipping {}, less than 500bp".format(record.id), file=sys.stderr)
                        continue

                    m = re.search(r"_cov_([\d\.]+)", record.id)
                    if m:
                        if float(m.group(1)) < 5:
                            #print("Skipping {}, low coverage {}".format(record.id, m.group(1)), file=sys.stderr)
                            continue
                    #else:
                        #print("Could not find coverage for {}".format(record.id), file=sys.stderr)
                    length = len(record.seq)
                    str = record.seq

                    #searching end of file for garbage
                    recur_char ="X"
                    window_size =0

                    #search blocks at the end of the contig for a sequence with a high density of a
                    #specific nucleotide, if there are no garbage sequences, recur_char will be 'X'
                    for i in range (30,410,20):
                        recur_char = find_recurring_char(record.seq,length-i, length)
                        if(recur_char != 'X'):
                            window_size = i
                            break
                    if(recur_char !='X'):
                        index = length-window_size+1
                        score = max_score
                        #until the score hits zero, traverse the string char by char and then change
                        #the score
                        while(score != 0):
                            index -=1
                            curr_char = record.seq[index]
                            if(curr_char==recur_char and score != max_score):
                                #if the next char matches the one we saw with increase density, add 1
                                score+=1
                            elif(curr_char!=recur_char):
                                #if the next char doesnt match the one with high density, minus 1
                                score-=1
                            if(score == max_score):
                                #every time we see a max score we mark everything past that point for deletion
                                window_size = length - index
                        str = record.seq[0:(length-window_size)]
                        print("Trimming {}, {} mostly {} bases removed".format(record.id, (length-len(str)),recur_char),"from end")
                    #searching front of file for garbage
                    length = len(str)
                    recur_char ="X"
                    window_size =0
                    for i in range (30,410,20):
                        recur_char = find_recurring_char(record.seq,0,i)
                        if(recur_char != 'X'):
                            window_size = i
                            break
                    if(recur_char !='X'):
                        index = window_size-1
                        score = max_score
                        while(score != 0):
                            index +=1
                            curr_char = record.seq[index]
                            if(curr_char==recur_char and score != max_score):
                                score+=1
                            elif(curr_char!=recur_char):
                                score-=1
                            if(score == max_score):
                                window_size = index
                        str = record.seq[window_size+1:length-1]
                        print("Trimming {}, {} mostly {} bases removed".format(record.id, (length-len(str)),recur_char),"from start")

                    record.seq=str
                    SeqIO.write(record, oh, "fasta")


if __name__ == "__main__":
    all_files = get_files_to_analyze(sys.argv[1])
    format_files(all_files, sys.argv[2])