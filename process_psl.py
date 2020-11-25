import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action = 'store_true', help = "Print status updates.")
    parser.add_argument('-p', '--psl', help = 'Path to the BLAT (psl) alignment output.')
    parser.add_argument('-t', '--order_threshold', type = float, help = 'Set to a maximum cosine similarity of block order value to say that overall synteny has been conserved.', default = 0.05)
    parser.add_argument('-l', '--min_length', type = int, help = 'Minimum total length of aligned bases to be included in the bedgraph output.', default = 150)
    parser.add_argument('-b', '--block_size', type = int, help = 'Minimum length of BLAT block alignment for inclusion in downstream analysis. Default 50 bases', default = 50)
    parser.add_argument('-e', '--bed_file', help = 'Set to a string to save the intermediary BLAT block table as a bedgraph with that name. Default does not save', default = None)
    parser.add_argument('-o', '--output', help = 'Name of the final bedgraph output containing full quality information for syntenic GRBs in the target species. One GRB per line.')
    args = parser.parse_args()
    return args

def psl_to_bdf(path, block_thresh = 50, bed_file = None):
    #read in and add columns
    blatdf = pd.read_csv(path, skiprows = [0,1,2,3,4], sep = '\t', names = ['Match','Mismatch','RepMatch','Ns','Qgapcount','Qgapbases','Tgapcount','Tgapbases','strand','Qname','Qsize','Qstart','Qend','Tname','Tsize','Tstart','Tend','BlockCount','BlockSizes','qStarts','tStarts'])
    blatdf = blatdf.dropna()
    blatdf['Species'] = [path.split('_')[0] for i in range(blatdf.shape[0])] #species tag is the file name with the _CNEblat.psl stripped off.
    blatdf['SeqDiv'] = blatdf.Match / (blatdf.Match + blatdf.Mismatch)
    blatdf['AlnCov'] = blatdf.Match / (blatdf.Tend - blatdf.Tstart)
    blatdf['GRBID'] = blatdf.Qname.apply(lambda x:x.split(';GRBID=')[1])
    #filter the set down to legit mappings based on quality filter parameters
    #requiring at least 50 bases mapped for each chunk and that they mapped at all ofc.
    #and load these into a new frame with one row per block. 
    if bed_file != None:
        outf = open(bed_file, 'w+')
    else:
        outf = None
    blatdf = blatdf.set_index('Qname') #I think this is correct
    bdf = {k:[] for k in ['TChro','TStart','TStop','TLen','Qname','Species']}
    for i,d in blatdf.iterrows():
        if d.tStarts == np.nan:
            continue #skip these ones that don't actually map, ofc.
        chro = d.Tname
        blocks = [int(l) for l in d.tStarts.split(",") if l != '']
        sizes = [int(s) for s in d.BlockSizes.split(',') if s != '']
        spec = d.Species
        for start, bl in zip(blocks, sizes):
            if bl > 50: #less than this are junk, skip them
                bdf['TChro'].append(chro)
                bdf['TStart'].append(start)
                bdf['TStop'].append(start + bl)
                bdf['TLen'].append(bl)
                bdf['Qname'].append(i)
                bdf['Species'].append(spec)
                if outf != None:
                    print('\t'.join([chro, str(start), str(start + bl), str(bl), str(i), spec]), file = outf)    
    bdf = pd.DataFrame(bdf)
    if outf != None:
        outf.close()
    return bdf

def process_bdf(bdf):
    bdf['GRBID'] = bdf.Qname.apply(lambda x:int(x.split("=")[-1]))
    bdf['CChro'] = bdf.Qname.apply(lambda x:x.split(':')[0])
    bdf['CStart'] = bdf.Qname.apply(lambda x:int(x.split(':')[1].split(';')[0].split('-')[0]))
    bdf['CStop'] = bdf.Qname.apply(lambda x:int(x.split(':')[1].split(';')[0].split('-')[1]))
    bdf['CLen'] = bdf.CStop - bdf.CStart
    return bdf

def chro_check(cvec, thresh = .75):
    vc = cvec.value_counts()
    return vc[0] > thresh * sum(vc), vc.index[0]

def write_bedgraph(bdf, out, max_cos, min_len):
    with open(out, "w+") as outf:
        for i, gdf in bdf.groupby(by='GRBID'):
            cont, chro = chro_check(gdf.TChro) #at least 75% of the blat blocks should go to the same chromosome before I even try to calculate cosine vectors
            if not cont:
                continue 
            gdf = gdf[gdf.TChro == chro].reset_index() #remove the up to 25% that won't go in right
            hdf = gdf.sort_values("CStart")
            human_order = hdf.index.tolist()
            human_length = abs(hdf.iloc[-1].CStop - hdf.iloc[0].CStart)
            tdf = gdf.sort_values("TStart")
            target_order = tdf.index.tolist() 
            target_length = abs(tdf.iloc[-1].TStop - hdf.iloc[0].TStart)
            forward_cos = spatial.distance.cosine(human_order, target_order)
            human_order.reverse()
            backward_cos = spatial.distance.cosine(human_order, target_order)
            if min([forward_cos,backward_cos]) < max_cos and sum(gdf.TLen) > min_len:
                #one of our handfuls of real Grbs?
                compact = target_length / human_length
                if gdf.iloc[-1].TStop < gdf.iloc[0].TStart:
                    tloc = [chro, gdf.iloc[-1].TStop, gdf.iloc[0].TStart] #to make it a proper bed type file, has to be less>more with explicit strand
                    strand = '-'
                else:
                    tloc = [chro, gdf.iloc[0].TStart, gdf.iloc[-1].TStop]
                    strand = '+'
                #okay, filter setups aside, time to print out the GRB orthology information
                #format- target chrom, target start, target end, strand, GRBID, human chro, human start, human end, block count, total bases in target, total, cosine similarity of order, compaction metric
                outv = [str(v) for v in tloc + [strand, i, gdf.iloc[0].CChro, gdf.iloc[0].CStart, gdf.iloc[-1].CStop, gdf.shape[0], sum(gdf.TLen), min([forward_cos, backward_cos]), compact]]
                print('\t'.join(outv), file = outf)


def main():
    args = argparser()
    bdf = psl_to_bdf(args.psl, args.block_size, args.bed_file)
    bdf = process_bdf(bdf)
    write_bedgraph(bdf, args.output, args.order_threshold, args.min_length)

if __name__ == "__main__":
    main()