import awkward as ak
import uproot
import math
from os import listdir
from os.path import isfile, join, getsize, abspath
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.set_loglevel("warning")
import numpy as np
import guppy
from Utils import heap_size,OUTPUT_PATH,PHPINDEX_FILE
from PlottingFunctions import ArrayOfRecords_HistogramBins,unpackVFATStatus_toBin,OHStatus_toBin
from config_parser import config
import argparse
from shutil import copy
from pathlib import Path
from myLogger import logger as default_logger

logger = default_logger.getLogger(__name__)
logger.setLevel(default_logger.INFO)

parser = argparse.ArgumentParser(description='Analyzer parser')
parser.add_argument('config', help='Analysis description file')
parser.add_argument('--folder_name', type=str , help="Output folder name",required=False,default="test")
args = parser.parse_args()

## CREATE FOLDERS and COPY FILES
output_folder_path = Path(OUTPUT_PATH,args.folder_name)
output_folder_path.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,output_folder_path)
output_folder_path = Path(output_folder_path,"DAQStatus")
output_folder_path.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,output_folder_path)
## 

the_heap = guppy.hpy()
the_heap.setref()  

configuration = config(abspath(args.config))
input_par = configuration.parameters
output_name = configuration.analysis_label+time.strftime("_%-y%m%d%H%M")
max_evts = configuration.data_input["max_evts"]
runRange = sorted(configuration.data_input["runRange"])
avg_batch_size = 500 # MB

perLS_Quantities = ["HasStatus","Warnings","Errors"]
perVFAT_Quantites = ["VFATMasked","VFATMissing","VFATZSd"]
figures = { k:plt.subplots(2,2,figsize=(39,20),layout="constrained") for k in perLS_Quantities+perVFAT_Quantites}
book_HistPerLS = ArrayOfRecords_HistogramBins (0,10**4,  ## x limits
                                               0.5,36.5, ## y limits
                                               perLS_Quantities )
book_HistPerVFAT = ArrayOfRecords_HistogramBins(0.5,36.5,
                                                -0.5,23.5,
                                                perVFAT_Quantites)

plt.rc('grid', linestyle='-.', color='lightgrey')
def main():
    heap_size(the_heap,"starting")
    files = [join(folder, file) for folder in configuration.data_input["folders"] for file in listdir(folder) if isfile(join(folder, file)) and "root" in file]
    AVG_FileSize =  sum([getsize(f) for f in files]) / len(files)
    gemOHStatus_collector = None

    files_per_batch = math.ceil( (avg_batch_size*2**(20)) / AVG_FileSize )
    tree_names = [ f+":muNtupleProducer/MuDPGTree;1" for f in files ]
    batches = [files[x:x+files_per_batch] for x in range(0, len(files), files_per_batch)]
    logger.info(f"Processing the root files in \n"+"\n".join(configuration.data_input['folders']))
    logger.info(f"{len(batches)} batches containing {ak.num(ak.Array(batches))} files (aiming for {avg_batch_size} MB per batch)")

    total_events = 0
    station = 1 ## GE11 Only
    max_lumiblock = 0
    min_lumiblock = float('inf')

    for batch_index,b in enumerate(batches):
        if max_evts!=-1 and total_events>max_evts: 
            logging.warning(f"Processed at least {max_evts} events, more than max_evt option. Exiting loop")
            break
        logger.info(f"Processing file batch {batch_index+1}/{len(batches)}")
        event =  uproot.concatenate(b,filter_name="*event*") 
        heap_size(the_heap,"after loading the event branch") 
        runNumber_mask = (event.event_runNumber >= runRange[0]) & (event.event_runNumber <= runRange[1])
        if ak.sum(runNumber_mask) == 0: 
            logger.warning(f"Skipping current batch due to run number not in range")
            del event
            continue ## no good runs in this file batch
        gemOHStatus = uproot.concatenate(b,filter_name="*gemOHStatus*")
        heap_size(the_heap,"after loading the gemohstatus branch")
        logger.info(f"{len(event)} evts")
        total_events += len(event)

        gemOHStatus = gemOHStatus[runNumber_mask]
        event = event[runNumber_mask]

        gemOHStatus = gemOHStatus[gemOHStatus.gemOHStatus_station == station]
        heap_size(the_heap,"after filtering on GE11 Station")

        gemOHStatus["gemOHStatus_lumiblock"] = ak.broadcast_arrays(event.event_lumiBlock,gemOHStatus["gemOHStatus_station"])[0]       
        max_lumiblock = max(max_lumiblock, ak.max(gemOHStatus.gemOHStatus_lumiblock))+1
        min_lumiblock = min(min_lumiblock, ak.min(gemOHStatus.gemOHStatus_lumiblock))

        logger.info(f"Aggregating data")
        heap_size(the_heap,"before aggregating data")
        Counts_OHHasStatus, Counts_OHErrors, Counts_OHWarnings = OHStatus_toBin(gemOHStatus,book_HistPerLS._base_array.copy())
        Counts_VFATMasked, Counts_VFATMissing, Counts_VFATZSd = unpackVFATStatus_toBin(gemOHStatus,book_HistPerVFAT._base_array.copy())

        book_HistPerLS.AddEntriesFromBinCounts("HasStatus",Counts_OHHasStatus)
        book_HistPerLS.AddEntriesFromBinCounts("Errors",Counts_OHErrors)
        book_HistPerLS.AddEntriesFromBinCounts("Warnings",Counts_OHWarnings)
        book_HistPerVFAT.AddEntriesFromBinCounts("VFATMasked",Counts_VFATMasked)
        book_HistPerVFAT.AddEntriesFromBinCounts("VFATMissing",Counts_VFATMissing)
        book_HistPerVFAT.AddEntriesFromBinCounts("VFATZSd",Counts_VFATZSd)
        heap_size(the_heap,"after aggregating data")
        
        del gemOHStatus
        del Counts_OHHasStatus, Counts_OHErrors, Counts_OHWarnings
        del Counts_VFATMasked, Counts_VFATMissing, Counts_VFATZSd
        heap_size(the_heap,"loop end")
        print()  
    
    logger.info(f"Slicing DAQ array to min Lumiblock: [{min_lumiblock}:{max_lumiblock}]")
    book_HistPerLS.x_low_lim = min_lumiblock
    book_HistPerLS.x_high_lim = max_lumiblock
    book_HistPerLS.ArrayOfRecords = book_HistPerLS.ArrayOfRecords[:,:,:,min_lumiblock:max_lumiblock,:]
    
    cmaps = [plt.cm.Blues,plt.cm.Reds,plt.cm.RdPu]
    for index, monitorable in enumerate(book_HistPerLS.fields):
        book_HistPerLS.plot(monitorable,                                                    # field to be plotted
                            np.arange(min_lumiblock,max_lumiblock, 50, dtype=float),        # x_ticks
                            np.arange(1,37, 1, dtype=float),                                # y_ticks
                            "Lumiblock",                                                    # xaxis_label
                            "Chamber",                                                      # yaxis_label
                            cmaps[index],                                                   # colormap
                            figures[monitorable][1].flatten())                              # ax
    cmaps = [plt.cm.Reds,plt.cm.Reds,plt.cm.Greys]
    for index, monitorable in enumerate(book_HistPerVFAT.fields):
        im = book_HistPerVFAT.plot(monitorable,                                           # field to be plotted
                              np.arange(1,37, 1, dtype=float),                            # x_ticks
                              np.arange(0,24, 1, dtype=float),                            # y_ticks
                              "Chamber",                                                  # xaxis_label
                              "VFAT",                                                     # yaxis_label
                              cmaps[index],                                               # colormap
                              figures[monitorable][1].flatten(),                          # ax
                              100/total_events)                                           # normalization_factor
        im.set_clim(0,100)        
        color_bar = figures[monitorable][0].colorbar(im, ax=figures[monitorable][1], pad=0.01)
        color_bar.ax.text(2.7,0.4,"% event ",rotation=90,fontsize=20)
        color_bar.ax.tick_params(labelsize=20)

    for k,item in figures.items():
        fig, _ = item
        fig.savefig(Path(output_folder_path,f"{output_name}_{k}.png"),dpi=120)
        fig.savefig(Path(output_folder_path,f"{output_name}_{k}.pdf"))
    
        
if __name__ == '__main__':
    main()
