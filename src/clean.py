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

    files_list = []
    if os.path.isdir(file_or_directory):
        # Build list of files from provided directory
        for root, dirs, files in os.walk(file_or_directory):
            for filename in files:
                files_list.append(os.path.join(root, filename))
    else:
        # We will just use the file given by the user
        files_list.append(os.path.abspath(file_or_directory))

    return sorted(files_list)


def find_recurring_char(record, start, end):
    """
    :param record: sequence to check for reccuring character
    :param start: start of window to look in
    :param end: end of window to look in
    :return: character occuring with > X% density, else returns 'X'
    """
    window = end - start

    # nucleotide simularity threshhold
    nuc_sim = 0.75

    for nucleotide in ['A','T','G','C','N']:
        if record.count(nucleotide,start,end) / window > nuc_sim:
            return nucleotide
    return 'X'

def format_files(files_list, output_dir):
    """
    Print to a new directory the re-formatted fasta files.
    We will remove anything smaller than 500bp, under 5x coverage,
    Any sequences of repeating or near repeating bases at the
    beginning or end of a contig are trimmed.
    :param files_list: list of fasta files to format headers
    :param output_dir: user supplied output directory
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
                    contig = record.seq
                    entirely_trash = False

                    #searching end of file for garbage
                    recur_char ="X"
                    window_size =0


                    # search blocks at the end of the contig for a sequence with a high density of a
                    # specific nucleotide, if there are no garbage sequences, recur_char will be 'X'
                    for i in range (30,410,20):
                        recur_char = find_recurring_char(record.seq,length-i, length)
                        if(recur_char != 'X'):
                            window_size = i
                            break
                    if(recur_char !='X'):
                        index = length-window_size+1
                        score = max_score
                        # until the score hits zero, traverse the string char by char and then change
                        # the score
                        while(score != 0):
                            index -=1

                            if(abs(index)>length):
                                # this occurs when the entire sequence is spanned (going out of index from the front)
                                entirely_trash = True
                                print("Skipping {} as it appears to be entirely garbage".format(record.id))
                                break

                            curr_char = record.seq[index]

                            if(curr_char==recur_char and score != max_score):
                                # if the next char matches the one we saw with increase density, add 1
                                score+=1
                            elif(curr_char!=recur_char):
                                # if the next char doesnt match the one with high density, minus 1
                                score-=1
                            if(score == max_score):
                                # every time we see a max score we mark everything past that point for deletion
                                window_size = length - index
                        if(entirely_trash):
                            continue
                        contig = record.seq[0:(length-window_size)]
                        print("Trimming {}, {} mostly {} bases removed".format(record.id, (length-len(contig)),recur_char),"from end")
                    # searching front of file for garbage
                    length = len(contig)
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

                            if(index>=length):
                                entirely_trash = True
                                print("Skipping {} as it appears to be entirely garbage".format(record.id))
                                break

                            curr_char = record.seq[index]

                            if(curr_char==recur_char and score != max_score):
                                score+=1
                            elif(curr_char!=recur_char):
                                score-=1
                            if(score == max_score):
                                window_size = index
                        if(entirely_trash):
                            continue
                        contig = record.seq[window_size+1:length-1]
                        print("Trimming {}, {} mostly {} bases removed".format(record.id, (length-len(contig)),recur_char),"from start")

                    if len(contig) < 500:
                        # post trimmed sequence is now below 500 nt threshhold
                        continue
                    record.seq=contig
                    SeqIO.write(record, oh, "fasta")


if __name__ == "__main__":
    all_files = get_files_to_analyze(sys.argv[1])
    format_files(all_files, sys.argv[2])
