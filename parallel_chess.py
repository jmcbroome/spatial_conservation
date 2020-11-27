#!/usr/bin/env python3
#this script is intended as a wrapper for CHESS-HIC, which is seriously bad at reading in large contact matrices
#it takes in the same set of positional arguments as CHESS does plus a max_threads argument
#it assumes the data is already O/E calculated, so make sure hicTransform has been ran on it
#then it splits the matrices into single chromosomes among all chromosomes mentioned in the bedpe file
#then it starts a chess run for each unique chromosome pair with an appropriate number of threads for the total runs that need to be completed and the max_threads option

import argparse
from subprocess import Popen
import subprocess
import sys
import time
import math
import traceback
import glob
import cooler
import numpy as np
from multiprocessing import Pool
import pandas as pd

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action = 'store_true', help = "Print status updates.")
    parser.add_argument('-t', '--threads', type = int, help = 'Maximum number of threads to use across all CHESS processes. Default 24', default = 24)
    parser.add_argument('-l', '--log', help = 'Name of a log file to track all chromosome splittings and chess runs.', default = 'parallel_chess_log.txt')
    parser.add_argument('-oe', '--observed_expected', action = 'store_true', help = 'Use if the input is already O/E transformed to avoid the CHESS implementation of O/E correction.')
    parser.add_argument('reference_hic', help = 'Path to position 1 (reference) O/E hic matrix in cooler format')
    parser.add_argument('query_hic', help = 'Path to position 2 (query) O/E hic matrix in cooler format')
    parser.add_argument('bedpe', help = 'Path to a bedpe file representing contact pairs of interest')
    args = parser.parse_args()
    return args

def parse_bedpe(path):
    '''
    Bedpe pairs format input must be:
    ref_chro ref_start ref_stop query_chro query_start query_stop pairID (null column) reference_strand query_strand
    where pairID is a unique identifier for each syntenic region pair and the null column is ignored
    
    This function reads in this text file format and returns a dictionary keyed on pairs of syntenic chromosomes with values of all regions assigned to that unique pair.
    '''
    pairs = {}
    with open(path) as inf:
        for entry in inf:
            try:
                rchro, rstart, rstop, qchro, qstart, qstop, grbid, _, rstrand, qstrand = entry.strip().split()
            except:
                print('ERROR: malformed bedpe input- check columns', out = sys.stderr)
                raise ValueError
            key = (rchro, qchro)
            if key not in pairs:
                pairs[key] = []
            pairs[key].append(entry.strip())
    return pairs

def chunks(lst, n):
    #borrowed from stackoverflow
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]    

def run_commands(commands, log):
    #print("QC: First ten commands in chess set ", len(commands[:10]), commands[:10], file = sys.stderr)
    procs = [Popen(' '.join(com), shell = True, stdout = subprocess.PIPE) for com in commands]
    for p in procs:
        pstdout, pstderr = p.communicate()
        logv = p.args
        if pstdout:
            logv.append(pstdout)
        if pstderr:
            logv.append(pstderr)
        log.append('\t'.join([l if type(l) == str else l.decode('utf-8') for l in logv]))
    return log

def cooler_split(args):
    tchro, cobj, cool_name, oe_input = args #needs to be an iterable because of Pool.
    #extract the bins and pixels which match this from the selector
    bins = cobj.bins().fetch(tchro) #more memory efficient reading
    #need to reset the bins indeces on the pixels so they range 0-X
    if not oe_input:
        #the values are already corrected when the O/E was calculated (I think) so they should be fine as is if its O/E
        #otherwise apply the correction factors for the two bins involved first
        matrix = cobj.matrix(balance = True, as_pixels = True).fetch(tchro)
        #create a pixels frame setting the count column equal to the matrix fetch balance column
        pixels = pd.DataFrame({'bin1_id':matrix['bin1_id'], 'bin2_id':matrix['bin2_id'], 'count':matrix['balanced']})
    else:
        #there is no weights, just get the raw pixel values as presented (should be o/e transformed)
        pixels = cobj.pixels().fetch(tchro)
    pixels['bin1_id'] = pixels['bin1_id'] - bins.index[0]
    pixels['bin2_id'] = pixels['bin2_id'] - bins.index[0]
    cooler.create_cooler(tchro + "_" + cool_name, bins, pixels, ordered = True, dtypes = {'count':np.float64})
    return tchro + "_" + cool_name

def split_matrices(pairs, rmatrix, qmatrix, log, oe_input, threads = 24):
    '''
    Use the cooler API to create new subcoolers representing various individual chromosomes in the set for downstream parallelization.
    Doesn't suffer from chromosome name inconsistency issues in the same way the hicexplorer implementation does
    '''
    rchros = list(set([k[0] for k in pairs.keys()]))
    qchros = list(set([k[1] for k in pairs.keys()]))
    log.append("Splitting coolers; {} chromosome targets".format(len(rchros) + len(qchros)))
    outd = {}
    rcool = cooler.Cooler(rmatrix)
    qcool = cooler.Cooler(qmatrix)
    #pool this
    #define an iterable of arguments
    commands = [(rc, rcool, rmatrix, oe_input) for rc in rchros] + [(qc, qcool, qmatrix, oe_input) for qc in qchros]
    #before proceeding, check that these files do not already exist
    #the filenames are keyed as rc + "_" + rmatrix for example
    fcommands = []
    for tc, tcool, tmat, oei in commands:
        if tc + "_" + tmat in glob.glob("*cool"):
            outd[tc] = tc + "_" + tmat
        else:
            fcommands.append((tc, tcool, tmat, oei))
    log.append("{} coolers already exist; creating {}".format(len(commands)-len(fcommands), len(fcommands)))    
    #break the command set into chunks of 24
    #may add a divider here on the number of threads to limit memory problems.
    if len(fcommands) > 0:
        for group in chunks(fcommands, threads):
            with Pool(processes=threads) as pool:
                for nn in pool.imap_unordered(cooler_split, group):
                    outd[nn.split('_')[0]] = nn
    log.append("Coolers processed.") #I really do need to learn the actual python Logging module at some point so I don't look silly. 
#    for rc in rchros:
#        nn = cooler_split(rc, rcool, rmatrix)
#        outd[rc] = nn
#    log.append("Reference cooler processed")
#    for qc in qchros:
#        nn = cooler_split(qc, qcool, qmatrix)
#        outd[qc] = nn
#    log.append("Query cooler processed")
    return outd, log

def split_matrices_old(pairs, rmatrix, qmatrix, log, threads = 24):
    '''
    Use HiCExplorer to extract unique chromosomes mentioned in the pairs file from the reference and query matrices. Parallelized step.
    '''
    #this is the first parallelized step
    #get the set of chromosomes for each matrix from pairs
    #then start a set of parallel commands dividing each of those matrices into those chromosomes
    #then return another dictionary keyed on chromosome name (and whether its query/reference) with a value of the string path of the split chromosome for downstream accession
    rchros = list(set([k[0] for k in pairs.keys()]))
    qchros = list(set([k[1] for k in pairs.keys()]))
    #get the full set of arguments I'll want to run in a list
    #hicConvertFormat can extract the chromosome from a cooler for me, by setting both input and output formats to cooler and limiting it to the chromosome in question
    split_commands = [['hicAdjustMatrix', '-c', rchro, '--matrix', rmatrix, '--outFileName', rchro + "_" + rmatrix] for rchro in rchros]
    split_commands.extend([['hicAdjustMatrix', '-c', qchro, '--matrix', qmatrix, '--outFileName', qchro + "_" + qmatrix] for qchro in qchros])
    #additionally, check to see if any of these target files already exist in the current directory
    #if they do, drop those commands from the set
    #this saves runtime when repeatedly iterating on these files
    split_commands = [com for com in split_commands if com[-1] not in glob.glob('./*cool')]

    #split these into groups based on the maximum threads parameter
    group_commands = chunks(split_commands, threads)
    #now start each group
    outd = {}
    for g in group_commands:
        #for each command in the group g, start a Popen constructor process for it, and wait until the whole set completes before starting the next set
        #most of the time there will only be one group, but for genomes with large numbers of chromosomes this is better handling.
        #(NOTE TO SELF: this may need reworking if it turns out the matrix doesn't like multiple simultaneous accessions)
        log = run_commands(g, log)
        #once these are done, update the out dictionary with chromosomes in this group
        for c in g:
            outd[c[2]] = c[-1]
    #once everything is all the way done, return the outd
    return outd, log

def start_chesses(pairs, cpd, log, threads = 24, oe_input = False):
    '''
    For each chromosome pair and regions in pairs, write the regions to a text file and start a chess command using that text file and the paths to the chromosome files created in the previous step. Parallelized.
    '''
    bed_d = {}
    for ck, regions in pairs.items():
        fn = '_'.join(list(ck) + ['regions.bedpe'])
        with open(fn, 'w+') as outf:
            for r in regions:
                print(r, file = outf) #should already be stripped of newlines, which this print will add back in
                bed_d[ck] = fn
    #now that those are created, time to start more popens
    #calculate the number of threads I can use per command
    #one command per entry in pairs
    maxt = max(1,math.floor(threads / min(len(pairs),24))) #if I have 12 pairs, that's 2 threads per command, etc. Always gives back at least 1 thread.
    #WARNING: THIS IS A HORRIBLE HACK REQUIRED BECAUSE THE CHROMOSOME NAMES FOR THE DOG TEST SET HAVE UNDERSCORES
    #NEED SOMETHING SMARTER IN THE FUTURE. IT WILL BREAK IMMEDIATELY ON THE NEXT THING.
    if oe_input:
        chess_commands = [['chess', 'sim', '--oe-input', '--background-query', '--limit-background', '-p', str(maxt), cpd[k[0]], cpd[k[1]], bed_d[k], '_'.join([cpd[k[0]].split("_")[0], '-'.join(cpd[k[1]].split("_")[:3]), 'chess.txt'])] for k in pairs.keys()]
    else:
        #same command without the O-E parameter, basically.
        chess_commands = [['chess', 'sim', '--background-query', '--limit-background', '-p', str(maxt), cpd[k[0]], cpd[k[1]], bed_d[k], '_'.join([cpd[k[0]].split("_")[0], '-'.join(cpd[k[1]].split("_")[:3]), 'chess.txt'])] for k in pairs.keys()]
    if oe_input:
        log.append("Running with precalculated O/E")
    else:
        log.append("Calculating O/E within CHESS")
    group_commands = chunks(chess_commands, threads) #groups of up to 24. 
    log.append("Starting CHESS runs- {} total runs, using {} threads each".format(len(chess_commands), maxt))
    start_time = time.time()

    for g in group_commands:
        log = run_commands(g, log)

    log.append("Chess runs complete. Total time {}".format((time.time() - start_time)/60/24)) #in hours
    return log

def main():
    args = argparser()
    logstrings = ['Initiating...']
    try:
        #if an error happens at any point, print whatever is in the log file before quitting
        pairs = parse_bedpe(args.bedpe)
        chro_paths, logstrings = split_matrices(pairs, args.reference_hic, args.query_hic, logstrings, args.observed_expected, args.threads)
        if args.verbose:
            print("QC: chro paths identified", chro_paths.keys(), file = sys.stdout)
        logstrings = start_chesses(pairs, chro_paths, logstrings, args.threads, args.observed_expected)
    except Exception as e:
        #print(e)
        logstrings.append("Failed with error: " + str(e))
        traceback.print_exc()
        pass
    with open(args.log, 'w+') as logout:
        for l in logstrings:
            print(l, file = logout)

if __name__ == "__main__":
    main()