#!/usr/bin/env python3

#this is a simple script I use to retrive a sparse matrix file representing all of Rao's data from their file structure at some resolution for all chromosomes
import argparse
import glob
import sys
import numpy as np

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resolution', help = 'Resolution value to extract from the rao dataset. A string that matches the Rao filenames (e.g. 25kb)', default = '25kb')
    parser.add_argument('-p', '--path', help = 'Path to the outermost Rao data directory. Assumed to be in the current wd if not given', default = '.')
    parser.add_argument('-c', '--chromsizes', help = 'Path to a chromsizes file to check that bins are labeled appropriately.', default = 'hg38_chromsizes.txt')
    parser.add_argument('-t', '--transform', action = 'store_true', help = 'Instead of collecting raw data, normalize and KR O/E transform the data before saving.')
    args = parser.parse_args()
    return args

def parse_csizes(path):
    with open(path) as inf:
        return {e[0]:int(e[1]) for e in [v.split() for v in inf]}

def get_paths(p, res):
    pathd = {}
    basep = p + '/GM12878_combined/' + res + '_resolution_intrachromosomal'
    for dirn in glob.glob(basep + '/*'):
        #relative path to each chr directory.
        chro = dirn.split('/')[-1]
        target = dirn + '/MAPQGE30/' + chro + "_" + res + '.RAWobserved'
        pathd[chro] = target
    return pathd

def read_factor(path):
    with open(path) as inf:
        return {i:float(v) for i,v in enumerate([e.strip() for e in inf])}

def read_transform(textpath):
    #change the ending of textpath to get the paths to the other two requisite files
    #KRnorm and KRexpected
    #print('QC: textpath is ' + textpath, file = sys.stderr)
    base = textpath.split("RAW")[0]
    knp = base + 'KRnorm'
    kep = base + 'KRexpected'
    #the output will be a dictionary which takes in a start/stop value and returns a factor to divide the raw value by before printing
    #for the norm step, you divide the raw number by the factors for the bins multiplied together
    #expected is based on bin difference
    knpd = read_factor(knp)
    kepd = read_factor(kep)
    return knpd, kepd

def simple_to_bg2(textpath, chro, csd, resolution, transform = False):

    resv = int(resolution[:-2])*1000

    if transform:
        #need to read in transformation data to apply to each entry.
        knpd, kepd = read_transform(textpath)
        def calc_trans(b1, b2):
            norm = knpd[int(b1)/resv+1] * knpd[int(b2)/resv+1]
            exp = kepd[abs(int(b1)-int(b2))/resv+1]
            #if norm == np.nan:
            #    print("QCT, norm is nan", file = sys.stderr)
            #if exp == np.nan:
            #    print("QCT, exp is nan", file = sys.stderr)
            return norm * exp

    with open(textpath) as inf:
        for entry in inf:
            b1, b2, v = entry.strip().split()
            #check that the entry falls within the chromosome size.
            if int(b2) + int(resolution[:-2])*1000 < csd[chro]: #otherwise cooler will lose its shit.
                if not transform:
                    tv = v
                else:
                    tv = str(float(v) / calc_trans(b1, b2))
                    #if tv == 'nan':
                        #print('QC', v, calc_trans(b1, b2), file = sys.stderr)
                nent = '\t'.join([chro, b1, str(int(b1) + resv), chro, b2, str(int(b2) + resv), tv])
                print(nent)

def main():
    args = argparser()
    csd = parse_csizes(args.chromsizes)
    pathd = get_paths(args.path, args.resolution)
    for chro, d in pathd.items():
        simple_to_bg2(d, chro, csd, args.resolution, args.transform)
    
if __name__ == "__main__":
    main()