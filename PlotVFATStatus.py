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
from Utils import heap_size,EOS_OUTPUT_PATH,EOS_INDEX_FILE
from PlottingFunctions import plotArray_2D
from config_parser import config
import argparse
from shutil import copy
from pathlib import Path
from myLogger import logger as default_logger

logger = default_logger.getLogger(__name__)
logger.setLevel(default_logger.DEBUG)

parser = argparse.ArgumentParser(description='Analyzer parser')
parser.add_argument('config', help='Analysis description file')
parser.add_argument('--folder_name', type=str , help="Output folder name",required=False,default="test")
args = parser.parse_args()

## CREATE FOLDERS and COPY FILES
output_folder_path = Path(EOS_OUTPUT_PATH,args.folder_name)
output_folder_path.mkdir(parents=True, exist_ok=True)
copy(EOS_INDEX_FILE,output_folder_path)
output_folder_path = Path(output_folder_path,"DAQStatus")
output_folder_path.mkdir(parents=True, exist_ok=True)
copy(EOS_INDEX_FILE,output_folder_path)
## 

the_heap = guppy.hpy()
the_heap.setref()  

configuration = config(abspath(args.config))
input_par = configuration.parameters
output_name = configuration.analysis_label+time.strftime("_%-y%m%d%H%M")
max_evts = configuration.data_input["max_evts"]
runRange = sorted(configuration.data_input["runRange"])
avg_batch_size = 100000 # MB

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
    if len(batches) > 1:
        logger.warning("Currently this script only processes the first batch")

    heap_size(the_heap,"before_loop")
    total_events = 0
    
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

        logger.debug(f" Selecting on run number")
        gemOHStatus = gemOHStatus[runNumber_mask]
        event = event[runNumber_mask]
        heap_size(the_heap,"after filtering on run number")
        print(gemOHStatus[gemOHStatus["gemOHStatus_VFATMissing"]!=0]["gemOHStatus_VFATMissing"])

        gemOHStatus["gemOHStatus_lumiblock"] = ak.broadcast_arrays(event.event_lumiBlock,gemOHStatus["gemOHStatus_station"])[0]       

        mask = (gemOHStatus.gemOHStatus_errors != 0) 
        logger.info(f"unpacking VFAT status")
        for VFAT_number in range(24):
            gemOHStatus[f"vfat{VFAT_number}_masked"] = np.logical_not((gemOHStatus.gemOHStatus_VFATMasked>>VFAT_number) & 0b1)
            gemOHStatus[f"vfat{VFAT_number}_missing"] = (gemOHStatus.gemOHStatus_VFATMissing>>VFAT_number) & 0b1
            # gemOHStatus[f"vfat{VFAT_number}_ZS"] = (gemOHStatus.gemOHStatus_VFATZS>>VFAT_number) & 0b1
            mask = mask | (gemOHStatus[f"vfat{VFAT_number}_masked"] == True)
        heap_size(the_heap,"after unpacking status")

        logger.info(f"Filling histos")
        start = time.time()
        ## Plotting
        fig_hasstatus, axs_hasstatus = plt.subplots(2,2,figsize=(39,20),layout="constrained")
        axs_hasstatus = axs_hasstatus.flatten()
        fig_warning, axs_warning = plt.subplots(2,2,figsize=(39,20),layout="constrained")
        axs_warning = axs_warning.flatten()
        fig_error, axs_error = plt.subplots(2,2,figsize=(39,20),layout="constrained")
        axs_error = axs_error.flatten()
        fig_vfatMasked, axs_masked = plt.subplots(2,2,figsize=(39,20),layout="constrained")
        axs_masked = axs_masked.flatten()
        img_masked = [None,None,None,None]
        fig_vfatMissing, axs_missing = plt.subplots(2,2,figsize=(39,20),layout="constrained")
        axs_missing = axs_missing.flatten()
        img_missing = [None,None,None,None]
       
        station = 1
        max_lumiblock = ak.max(gemOHStatus.gemOHStatus_lumiblock)+1
        min_lumiblock = ak.min(gemOHStatus.gemOHStatus_lumiblock)
        
        heap_size(the_heap,"before plotting")
        ## Plot Has Status
        for idx, (region,layer) in enumerate([(-1,1),(1,1),(-1,2),(1,2)]):    
            endcapMask = (gemOHStatus.gemOHStatus_station == station)  & (gemOHStatus.gemOHStatus_region == region) & (gemOHStatus.gemOHStatus_layer == layer)
            plotArray_2D(gemOHStatus, endcapMask, "gemOHStatus_lumiblock", "gemOHStatus_chamber", (min_lumiblock,max_lumiblock), (0.5,36.5), np.arange(min_lumiblock,max_lumiblock, 50, dtype=float), np.arange(1,37, 1, dtype=float), f"GE{'+' if region>0 else '-'}{station}1 Ly{layer} HasStatus", "Lumiblock", "Chamber", plt.cm.Blues, axs_hasstatus[idx])
            
            
            #### ORIGINAL
            # axs_hasstatus[idx].hist(ak.flatten(gemOHStatus[endcapMask]["gemOHStatus_chamber"]), bins=range(1, 37 + 1, 1),edgecolor='black',range=np.array([(0.5, 37.5)]),alpha = 0.5,color='g',weights=(100/total_events)*np.ones_like(ak.flatten(gemOHStatus[endcapMask]["gemOHStatus_chamber"])))
            
            # axs_hasstatus[idx].set_title(f"GE{'+' if region>0 else '-'}{station}1 Ly{layer} hasStatus",fontweight="bold", size=24)
            # axs_hasstatus[idx].set_xlabel("Chamber", loc='right',size=20)
            # axs_hasstatus[idx].set_ylabel("% event OHHasStatus", loc='center',size=20)

            # # Major ticks
            # axs_hasstatus[idx].set_xticks(np.arange(1, 37, 1))
            # # axs_hasstatus[idx].set_yticks(np.arange(0, 1.2, 0.2))
            # # Labels for major ticks
            # axs_hasstatus[idx].set_xticklabels(np.arange(1, 37, 1),size=15)
            # # axs_hasstatus[idx].set_yticklabels(np.arange(0, 1.2, 0.2),size=15)
            # axs_hasstatus[idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # # Minor ticks
            # axs_hasstatus[idx].set_xticks(np.arange(.5,37.5, 1), minor=True)
            # # Gridlines based on minor ticks
            # axs_hasstatus[idx].grid(which='major')
        
        logger.info("discarding events with no errors/warnings/mask")
        gemOHStatus = gemOHStatus[mask]
        heap_size(the_heap,"after discarding meaningless events")

        # Plot Errors / Warnings / VFAT Masked
        for idx, (region,layer) in enumerate([(-1,1),(1,1),(-1,2),(1,2)]):    
            endcapMask = (gemOHStatus.gemOHStatus_station == station)  & (gemOHStatus.gemOHStatus_region == region) & (gemOHStatus.gemOHStatus_layer == layer)
            # Errors / Warnings
            plotArray_2D(gemOHStatus, endcapMask & (gemOHStatus.gemOHStatus_warnings >0), "gemOHStatus_lumiblock", "gemOHStatus_chamber", (min_lumiblock,max_lumiblock), (0.5,36.5), np.arange(min_lumiblock,max_lumiblock, 50, dtype=float), np.arange(1,37, 1, dtype=float), f"GE{'+' if region>0 else '-'}{station}1 Ly{layer} Warnings", "Lumiblock", "Chamber", plt.cm.Blues, axs_warning[idx])
            plotArray_2D(gemOHStatus, endcapMask & (gemOHStatus.gemOHStatus_errors >0), "gemOHStatus_lumiblock", "gemOHStatus_chamber", (min_lumiblock,max_lumiblock), (0.5,36.5), np.arange(min_lumiblock,max_lumiblock, 50, dtype=float), np.arange(1,37, 1, dtype=float), f"GE{'+' if region>0 else '-'}{station}1 Ly{layer} Errors", "Lumiblock", "Chamber", plt.cm.Reds, axs_error[idx])
            
            # VFAT Masked / Missing
            plotting_data_masked = {"Chamber":np.empty(0,dtype=int),"VFAT":np.empty(0,dtype=int)}
            plotting_data_missing = {"Chamber":np.empty(0,dtype=int),"VFAT":np.empty(0,dtype=int)}
            for vfat in range(24):
                mask = (gemOHStatus[f"vfat{vfat}_masked"] == True) & endcapMask
                ## Getting vfat masked data
                temp_sel = gemOHStatus[mask]
                temp_sel["VFAT"] = ak.broadcast_arrays(vfat,temp_sel.gemOHStatus_chamber)[0]
                plotting_data_masked["Chamber"] = np.append(plotting_data_masked["Chamber"],ak.flatten(temp_sel.gemOHStatus_chamber))
                plotting_data_masked["VFAT"] = np.append(plotting_data_masked["VFAT"],ak.flatten(ak.broadcast_arrays(vfat,temp_sel.gemOHStatus_chamber)[0]))
                
                ## Getting vfat missing data
                mask = (gemOHStatus[f"vfat{vfat}_missing"] == True) & endcapMask
                temp_sel = gemOHStatus[mask]
                temp_sel["VFAT"] = ak.broadcast_arrays(vfat,temp_sel.gemOHStatus_chamber)[0]
                plotting_data_missing["Chamber"] = np.append(plotting_data_missing["Chamber"],ak.flatten(temp_sel.gemOHStatus_chamber))
                plotting_data_missing["VFAT"] = np.append(plotting_data_missing["VFAT"],ak.flatten(ak.broadcast_arrays(vfat,temp_sel.gemOHStatus_chamber)[0]))
            
            img_masked[idx] = plotArray_2D(plotting_data_masked, None, "Chamber", "VFAT", (0.5,36.5), (-0.5,23.5), np.arange(1,37, 1, dtype=float), np.arange(0,24, 1, dtype=float), f"GE{'+' if region>0 else '-'}{station}1 Ly{layer} VFAT Masked", "Chamber", "VFAT", plt.cm.Reds, axs_masked[idx],normalization_factor=total_events/100)
            color_bar = fig_vfatMasked.colorbar(img_masked[idx], ax=axs_masked[idx], pad=0.01)
            color_bar.ax.text(2.7,0.4,"% event masked",rotation=90,fontsize=20)
            
            img_missing[idx] = plotArray_2D(plotting_data_missing, None, "Chamber", "VFAT", (0.5,36.5), (-0.5,23.5), np.arange(1,37, 1, dtype=float), np.arange(0,24, 1, dtype=float), f"GE{'+' if region>0 else '-'}{station}1 Ly{layer} VFAT Missing", "Chamber", "VFAT", plt.cm.Greys, axs_missing[idx],normalization_factor=total_events/100)   
            color_bar = fig_vfatMissing.colorbar(img_missing[idx], ax=axs_missing[idx], pad=0.01)
            color_bar.ax.text(2.7,0.4,"% event vfat missing",rotation=90,fontsize=20)
            
        heap_size(the_heap,"after plotting")            
        logger.debug(f" Plotting took {time.time()-start:.2f} s")
        logger.debug(f" Now saving plots...")
        
        fig_hasstatus.savefig(Path(output_folder_path,f"{output_name}_HasStatus.png"),dpi=200)
        fig_warning.savefig(Path(output_folder_path,f"{output_name}_Warnings.png"),dpi=200)
        fig_error.savefig(Path(output_folder_path,f"{output_name}_Errors.png"),dpi=200)
        fig_vfatMasked.savefig(Path(output_folder_path,f"{output_name}_VFATMasked.png"),dpi=200)
        fig_vfatMissing.savefig(Path(output_folder_path,f"{output_name}_VFATMissing.png"),dpi=200)
        
        fig_hasstatus.savefig(Path(output_folder_path,f"{output_name}_HasStatus.pdf"))
        fig_warning.savefig(Path(output_folder_path,f"{output_name}_Warnings.pdf"))
        fig_error.savefig(Path(output_folder_path,f"{output_name}_Errors.pdf"))
        fig_vfatMasked.savefig(Path(output_folder_path,f"{output_name}_VFATMasked.pdf"))
        fig_vfatMissing.savefig(Path(output_folder_path,f"{output_name}_VFATMissing.pdf"))
        
        break
        

    ### Plotting occupancy per VFAT
    # print(ak.max(ak.ravel(gemRecHit["gemRecHit_VFAT"])))
    # print(ak.min(ak.ravel(gemRecHit["gemRecHit_VFAT"])))

    # h = ROOT.TH2F("occupancy","occupancy",400,-60,60,400,-94,94)
    # hx = ROOT.TH2F("occupancy","occupancy",600,-340,340,600,-340,340)
    # h_test = ROOT.TH2F("ss","ss",600,-340,340,600,-340,340)
    # s = gemPropHit[(gemPropHit.mu_propagated_station==2) & (gemPropHit.propagated_VFAT == 11) ]
    # s = s[ak.count(s.mu_propagated_chamber,axis=-1) != 0]
    
    # xxx = ak.ravel(s.mu_propagatedGlb_x)
    # yyy = ak.ravel(s.mu_propagatedGlb_y)
    

    # g_xxx = ak.ravel(gemRecHit[(gemRecHit.gemRecHit_station == 2) & (gemRecHit.gemRecHit_VFAT == 11)].gemRecHit_g_x)
    # g_yyy = ak.ravel(gemRecHit[(gemRecHit.gemRecHit_station == 2) & (gemRecHit.gemRecHit_VFAT == 11)].gemRecHit_g_y)

    # for idx in range(len(xxx)):
    #     h_test.Fill(xxx[idx],yyy[idx])
    # for idx in range(len(g_xxx)):
    #     hx.Fill(g_xxx[idx],g_yyy[idx])
        
    # polyplotGE21.SetBinContent(map_GE21['GE21-P-16L2-M1'][1],100)
    # polyplotGE21.SetBinContent(map_GE21['GE21-P-16L2-M2'][1],100)
    # polyplotGE21.SetBinContent(map_GE21['GE21-P-16L2-M3'][1],100)
    # polyplotGE21.SetBinContent(map_GE21['GE21-P-16L2-M4'][1],100)
    ### Plotting occupancy per VFAT

    # c1 = ROOT.TCanvas("c1","c1",1600,1600)
    #h_test.Draw("COLZ")
    # c1.Modified()
    # c1.Update()
    # c1.SaveAs("prop.pdf")
    # c1.SaveAs("prop.png")
    
    #hx.Draw("COLZ")
    # c1.Modified()
    # c1.Update()
    # c1.SaveAs("rec.pdf")
    # c1.SaveAs("rec.png")

    #polyplotGE21.Draw("COLZ")
    # c1.Modified()
    # c1.Update()
    # c1.SaveAs("vfat.pdf")
    # c1.SaveAs("vfat.png")



if __name__ == '__main__':
    main()
