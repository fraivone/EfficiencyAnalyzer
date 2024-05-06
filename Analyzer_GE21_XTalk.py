import argparse
import math
from os import listdir
from os.path import isfile, join, getsize, abspath
from pathlib import Path
import time
from shutil import copy
import awkward as ak
import uproot
import numpy as np
import matplotlib.pyplot as plt

import guppy

from Utils import (
    heap_size,
    iEtaiPhi_2_VFAT,
    recHit2VFAT,
    EfficiencySummary,
    OUTPUT_PATH,
    PHPINDEX_FILE,
    BASE_DIR,
    ExtendEfficiencyCSV,
)
from MaskFunctionsGE21 import (
    calcMuonHit_masks,
    countNumberOfPropHits,
    calcDAQMaskedVFAT_mask,
    calcDAQMissingVFAT_mask,
    calcDAQError_mask,
    calcDAQenabledOH_mask,
    calcHV_mask,
)
from AkArray_Operations import find_boundaries, boundaries_translation, best_match
from PlottingFunctions import Fill_Histo_Residuals, Plot_Binned_Residuals,ArrayOfRecords_HistogramBins,unpackVFATStatus_toBin,OHStatus_toBin
from config_parser import config
from myLogger import logger as logger_default

logger = logger_default.getLogger(__name__)
logger.setLevel(logger_default.MEMORY)
the_heap = guppy.hpy()
the_heap.setref()

## PARSER
parser = argparse.ArgumentParser(description="Analyzer parser")
parser.add_argument("config", help="Analysis description file")
parser.add_argument("--folder_name", type=str, help="Output folder name", required=False, default="test")
parser.add_argument("--residuals", help="Enable plotting residuals", required=False, action="store_true", default=False)
parser.add_argument("--PlotDAQ", help="Enable plotting residuals", required=False, action="store_true", default=False)
args = parser.parse_args()

configuration = config(abspath(args.config))
input_par = configuration.parameters
matching_cuts = configuration.matching_window
match_by = "residual_rdphi"
output_folder_path = Path(OUTPUT_PATH, args.folder_name)
output_name = configuration.analysis_label + time.strftime("_%-y%m%d%H%M")
max_evts = configuration.data_input["max_evts"]
runRange = sorted(configuration.data_input["runRange"])
## PARSER
## CREATE FOLDERS and COPY FILES
output_folder_path.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, output_folder_path)
residual_output_folder_path = Path(output_folder_path, "Residuals")
residual_output_folder_path.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, residual_output_folder_path)
DAQStatus_output_folder_path = Path(output_folder_path,"DAQStatus")
DAQStatus_output_folder_path.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,DAQStatus_output_folder_path)
##

avg_batch_size = 1200  # MB
heap_dump_size = 500  # MB above which efficiency summary gets calculated from propagated_collector and matched_collector and dumped to file

## instances for DAQ status plotting
perLS_Quantities = ["HasStatus","Warnings","Errors"]
perVFAT_Quantites = ["VFATMasked","VFATMissing","VFATZSd"]
figures = { k:plt.subplots(2,2,figsize=(39,20),layout="constrained") for k in perLS_Quantities+perVFAT_Quantites}

book_HistPerLS = ArrayOfRecords_HistogramBins (0,10**4,  ## x limits
                                               0.5,36.5, ## y limits
                                               perLS_Quantities )
book_HistPerVFAT = ArrayOfRecords_HistogramBins(0.5,36.5,         ## x limits
                                                -0.5,23.5,        ## y limits
                                                perVFAT_Quantites)
max_lumiblock = 0
min_lumiblock = float('inf')
total_events = 0


def main():
    heap_size(the_heap, "starting")

    HVmask_path = configuration.data_input["HVMask_path"]
    files = [
        join(folder, file)
        for folder in configuration.data_input["folders"]
        for file in listdir(folder)
        if isfile(join(folder, file)) and "root" in file
    ]
    ### reRECO only
    # with open(Path(BASE_DIR / "data/maps/RunNumber_ReRECO_map.json"), "r") as f:
    #     RunNumber_Map = json.loads(f.read())
    # files = [f for f in files if runRange[0] in RunNumber_Map[f]]
    ### reRECO only
    AVG_FileSize = sum([getsize(f) for f in files]) / len(files)
    matched_collector = None
    propagated_collector = None
    compatible_collector = None
    cutSummary_collector = {}

    files_per_batch = math.ceil((avg_batch_size * 2 ** (20)) / AVG_FileSize)
    batches = [files[x : x + files_per_batch] for x in range(0, len(files), files_per_batch)]
    logger.info(f"Processing the root files in \n" + "\n".join(configuration.data_input["folders"]))
    logger.info(f"Matching based on: {match_by} <= {matching_cuts[match_by]}")
    logger.info(
        f"{len(batches)} batches containing {ak.num(ak.Array(batches))} files (aiming for {avg_batch_size} MB per batch)"
    )

    branches_rechit = [
        "gemRecHit_region",
        "gemRecHit_chamber",
        "gemRecHit_layer",
        "gemRecHit_etaPartition",
        "gemRecHit_g_r",
        "gemRecHit_loc_x",
        "gemRecHit_loc_y",
        "gemRecHit_g_x",
        "gemRecHit_g_y",
        "gemRecHit_g_z",
        "gemRecHit_g_phi",
        "gemRecHit_firstClusterStrip",
        "gemRecHit_cluster_size",
        "gemRecHit_station",
    ]
    branches_prophit = [
        "mu_propagated_region",
        "mu_propagated_chamber",
        "mu_propagated_layer",
        "mu_propagated_etaP",
        "mu_propagated_Outermost_z",
        "mu_propagated_isME11",
        "mu_propagated_isME21",
        "mu_propagatedGlb_r",
        "mu_propagatedLoc_x",
        "mu_propagatedLoc_y",
        "mu_propagatedLoc_phi",
        "mu_propagatedGlb_x",
        "mu_propagatedGlb_y",
        "mu_propagatedGlb_z",
        "mu_propagatedGlb_phi",
        "mu_propagatedGlb_errR",
        "mu_propagatedGlb_errPhi",
        "mu_propagatedLoc_dirX",
        "mu_propagatedLoc_dirY",
        "mu_propagated_pt",
        "mu_propagated_isGEM",
        "mu_propagated_TrackNormChi2",
        "mu_propagated_nME1hits",
        "mu_propagated_nME2hits",
        "mu_propagated_nME3hits",
        "mu_propagated_nME4hits",
        "mu_propagated_station",
        "mu_propagated_isME21",
    ]

    GE21_phi_range = 0.3631972
    ## boundaries values for iPhi in terms of cos(loc_x / glb_r)
    GE11_cos_phi_boundaries = (
        -0.029824804,
        0.029362374,
    )
    global total_events
    station = 2 ## GE21 Only
    global max_lumiblock
    global min_lumiblock

    for batch_index, b in enumerate(batches):
        if max_evts != -1 and total_events > max_evts:
            logger.warning(f"Processed at least {max_evts} events, more than max_evt option. Exiting loop")
            break

        logger.info(f"Processing file batch {batch_index+1}/{len(batches)}")
        logger.debug(f" Loading branches into awkward arrays")
        event = uproot.concatenate(b, filter_name="*event*")
        heap_size(the_heap, "after loading the event branch")
        runNumber_mask = (event.event_runNumber >= runRange[0]) & (event.event_runNumber <= runRange[1])
        if ak.sum(runNumber_mask) == 0:
            logger.warning(f"Skipping current batch due to run number not in range")
            del event
            print()
            continue  ## Run numbers out of range  in this file batch
        gemRecHit = uproot.concatenate(b, filter_name=branches_rechit)
        heap_size(the_heap, "after loading the rechit branch")
        gemPropHit = uproot.concatenate(b, filter_name=branches_prophit)
        heap_size(the_heap, "after loading the prophit branch")
        gemOHStatus = uproot.concatenate(b, filter_name="*gemOHStatus*")
        heap_size(the_heap, "after loading the gemohstatus branch")
        logger.info(f"\033[4m\033[1m{len(event)} evts\033[0m")

        total_events += len(event)

        logger.debug(f" Selecting on run number")
        gemPropHit = gemPropHit[runNumber_mask]
        gemRecHit = gemRecHit[runNumber_mask]
        gemOHStatus = gemOHStatus[runNumber_mask]
        event = event[runNumber_mask]
        heap_size(the_heap, "after filtering on run number")

        logger.debug(f" Add event info in the gemPropHit,gemRecHit,gemOHStatus arrays")
        gemPropHit["prop_eventNumber"] = ak.broadcast_arrays(event.event_eventNumber, gemPropHit["mu_propagated_isGEM"])[0]
        gemPropHit["mu_propagated_lumiblock"] = ak.broadcast_arrays(event.event_lumiBlock, gemPropHit["mu_propagated_isGEM"])[0]
        gemRecHit["rec_eventNumber"] = ak.broadcast_arrays(event.event_eventNumber, gemRecHit["gemRecHit_chamber"])[0]
        gemRecHit["gemRecHit_lumiblock"] = ak.broadcast_arrays(event.event_lumiBlock, gemRecHit["gemRecHit_chamber"])[0]
        gemOHStatus["gemOHStatus_lumiblock"] = ak.broadcast_arrays(event.event_lumiBlock, gemOHStatus["gemOHStatus_station"])[0]
        gemOHStatus["gemOHStatus_eventNumber"] = ak.broadcast_arrays(event.event_eventNumber, gemOHStatus["gemOHStatus_station"])[0]
        
        max_lumiblock = max(max_lumiblock, ak.max(gemOHStatus.gemOHStatus_lumiblock))+1
        min_lumiblock = min(min_lumiblock, ak.min(gemOHStatus.gemOHStatus_lumiblock))


        if args.PlotDAQ:
            ## DAQ Status data
            logger.info(f"Aggregating DAQStatus data")
            heap_size(the_heap,"before aggregating DAQStatus data")
            Counts_OHHasStatus, Counts_OHErrors, Counts_OHWarnings = OHStatus_toBin(gemOHStatus,book_HistPerLS._base_array.copy())
            Counts_VFATMasked, Counts_VFATMissing, Counts_VFATZSd = unpackVFATStatus_toBin(gemOHStatus,book_HistPerVFAT._base_array.copy())

            book_HistPerLS.AddEntriesFromBinCounts("HasStatus",Counts_OHHasStatus)
            book_HistPerLS.AddEntriesFromBinCounts("Errors",Counts_OHErrors)
            book_HistPerLS.AddEntriesFromBinCounts("Warnings",Counts_OHWarnings)
            book_HistPerVFAT.AddEntriesFromBinCounts("VFATMasked",Counts_VFATMasked)
            book_HistPerVFAT.AddEntriesFromBinCounts("VFATMissing",Counts_VFATMissing)
            book_HistPerVFAT.AddEntriesFromBinCounts("VFATZSd",Counts_VFATZSd)
            heap_size(the_heap,"after aggregating DAQStatus data")
        
        logger.debug(f" Adding eta partition ID")
        PropEtaID = (
            (2**17 - 1) & gemPropHit[:].mu_propagated_chamber << 11
            | (gemPropHit[:].mu_propagated_etaP << 6)
            | (gemPropHit[:].mu_propagated_station << 4)
            | (gemPropHit[:].mu_propagated_layer << 1)
            | (abs(gemPropHit[:].mu_propagated_region - 1) // 2)
        )
        RecEtaID = (
            (2**17 - 1) & gemRecHit[:].gemRecHit_chamber << 11
            | (gemRecHit[:].gemRecHit_etaPartition << 6)
            | (gemRecHit[:].gemRecHit_station << 4)
            | (gemRecHit[:].gemRecHit_layer << 1)
            | (abs(gemRecHit[:].gemRecHit_region - 1) // 2)
        )
        gemPropHit["prop_etaID"] = PropEtaID
        gemRecHit["rec_etaID"] = RecEtaID
        heap_size(the_heap, "after adding etaID")

        ## Selecting events having at least 1 prop hit on GE21
        atLeast1GE21Rec = ak.any(gemRecHit.gemRecHit_station == station, axis=-1)
        atLeast1GE21Prop = ak.any(gemPropHit.mu_propagated_station == station, axis=-1)
        GE21_RecHit_withProphits = atLeast1GE21Rec & atLeast1GE21Prop
        gemPropHit = gemPropHit[GE21_RecHit_withProphits]
        gemRecHit = gemRecHit[GE21_RecHit_withProphits]
        gemOHStatus = gemOHStatus[GE21_RecHit_withProphits]
        ## Excluding propagated hits from GE11
        print(ak.max( gemPropHit.mu_propagated_station))
        gemPropHit = gemPropHit[gemPropHit.mu_propagated_station == station]
        # Removing all GE11 prophits (no additional events removed)
        heap_size(the_heap, "after filtering out GE21")

        logger.debug(f" Extracting eta partition boundaries")
        etaID_boundaries_akarray_pre = ak.Array(map(find_boundaries, gemPropHit.prop_etaID))
        heap_size(the_heap, "arraying etaP boundaries")


        logger.debug(f" Calculating masks")
        logger.debug2(f" Adjusting for angle periodicity")
        boundary_translation_mask = etaID_boundaries_akarray_pre[..., 1] > etaID_boundaries_akarray_pre[..., 0]
        translation_array = ak.Array(map(boundaries_translation, boundary_translation_mask))
        etaID_boundaries_akarray = etaID_boundaries_akarray_pre + translation_array
        """
        Since some boundaries phiMax have been translated by 0,2pi , to consistently apply the mask 
        the same has to be done on mu_propagatedGlb_phi. The operation will then be inverted after
        applying the mask
        """
        propGlbPhi_translation_mask = np.logical_and(boundary_translation_mask, gemPropHit.mu_propagatedGlb_phi < 0)
        gemPropHit["mu_propagatedGlb_phi"] = gemPropHit.mu_propagatedGlb_phi + propGlbPhi_translation_mask * 2 * np.pi
        heap_size(the_heap, "after translating the boundaries")

        logger.debug(f" Calculating selection mask on muonTrack & propagation")
        input_par["gemprophit_array"] = gemPropHit
        input_par["etaID_boundaries_array"] = etaID_boundaries_akarray
        muonTrack_mask = calcMuonHit_masks(**input_par)

        logger.debug2(f" Restoring angle periodicity")
        gemPropHit["mu_propagatedGlb_phi"] = gemPropHit.mu_propagatedGlb_phi - propGlbPhi_translation_mask * 2 * np.pi

        logger.debug(f" Calculating selection mask on DAQ error")
        DAQError_mask = calcDAQError_mask(gemPropHit, gemOHStatus)
        muonTrack_mask["DAQError"] = DAQError_mask
        muonTrack_mask["overallGood_Mask"] = muonTrack_mask["overallGood_Mask"] & DAQError_mask
        heap_size(the_heap, "after calculating DAQ error mask")

        logger.debug(f" Calculating selection mask on DAQ enabled OH")
        DAQenabledOH_mask = calcDAQenabledOH_mask(gemPropHit, gemOHStatus)
        muonTrack_mask["DAQenabledOH"] = DAQenabledOH_mask
        muonTrack_mask["overallGood_Mask"] = muonTrack_mask["overallGood_Mask"] & DAQenabledOH_mask
        heap_size(the_heap, "after calculating DAQ enabled OH")

        logger.debug(f" Calculating HV selection mask")
        if HVmask_path is not None:
            HV_mask = calcHV_mask(gemPropHit, HVmask_path)
            muonTrack_mask["HVMask"] = HV_mask
            muonTrack_mask["overallGood_Mask"] = muonTrack_mask["overallGood_Mask"] & HV_mask
            heap_size(the_heap, "after calculating HV selection mask")

        ## PropHit selection summary
        survived_hits = countNumberOfPropHits(muonTrack_mask)
        ## Sorting
        temp_dict = {k: v for k, v in sorted(survived_hits.items(), key=lambda item: item[1], reverse=True)}
        logger.debug(f" Breakdown table with cuts")
        logger.debug(f" {'Label':<20}\t{'Survived Hits':>15}\t{'% Survived':>20}")
        for k in temp_dict:
            logger.debug(f" {k:<20}\t{temp_dict[k]:>15}\t{round(temp_dict[k]*100/temp_dict['no_Mask'],2):>19}%")
        ## bookeeping all cuts
        cutSummary_collector = {k: cutSummary_collector.get(k, 0) + temp_dict.get(k, 0) for k in set(temp_dict)}

        logger.debug(f" Applying cuts")
        ## PROPHIT SELECTION: keep ALL the events. Throw away the invalid prophits
        selectedRecHit = gemRecHit[ak.any(muonTrack_mask["overallGood_Mask"], axis=-1)]
        selectedPropHit = gemPropHit[muonTrack_mask["overallGood_Mask"]]
        event_hasProphits = ak.count(selectedPropHit.mu_propagated_chamber, axis=-1) != 0
        selectedPropHit = selectedPropHit[event_hasProphits]

        print(len(muonTrack_mask["overallGood_Mask"]))
        print(len(gemPropHit))
        print(ak.sum(ak.any(muonTrack_mask["overallGood_Mask"], axis=-1)))
        print(len(selectedPropHit))
        print(len(selectedRecHit))



        del gemPropHit
        del gemRecHit
        del gemOHStatus
        heap_size(the_heap, "after applying cuts")

        logger.debug(f" Pairing etaID")
        ## how many times the prophit's eventnumber matches the rechit's eventnumber
        syncd_events = ak.sum(ak.any(selectedPropHit.prop_eventNumber[..., 0] == selectedRecHit.rec_eventNumber,axis=-1))
        total_prophit = ak.sum(ak.num(selectedPropHit.prop_etaID), axis=-1)
        if syncd_events != len(selectedPropHit):
            logger.error(
                f"PropHit and RecHit have mismatches on eventNumber.\nTotal prophit = {total_prophit}\t Rechit with same evt number = {syncd_events}"
            )
        
        break
    return selectedPropHit,selectedRecHit


if __name__ == "__main__":
    p, r = main()
    fig_p, ax_p = plt.subplots(1,figsize=(10,15),layout="constrained")
    fig_r, ax_r = plt.subplots(1,figsize=(10,15),layout="constrained")
    fig_m, ax_m = plt.subplots(1,figsize=(10,10),layout="constrained")
    ax_p.hist2d(
        ak.flatten(p.mu_propagatedLoc_x).to_numpy(),
        ak.flatten(p.mu_propagatedLoc_y).to_numpy(),
        bins=(100, 100),
        range=np.array([(-60, 60), (-90, 90)]),
        cmap = "Blues"
    )
    ax_p.set_title("2D propagated on GE21", fontweight="bold", size=24)
    ax_p.set_xlabel("PropHit Loc x (cm)", loc="right", size=20)
    ax_p.set_ylabel("PropHit Loc y (cm)", loc="center", size=20)


    ax_r.hist2d(
        ak.flatten(r.gemRecHit_loc_x).to_numpy(),
        ak.flatten(r.gemRecHit_loc_y).to_numpy(),
        bins=(100, 100),
        range=np.array([(-40, 40), (-75, 75)]),
        cmap="Blues"
    )
    ax_r.set_title("2D recHit for evts with GE21 propHits ", fontweight="bold", size=24)
    ax_r.set_xlabel("RecHit Loc x (cm)", loc="right", size=20)
    ax_r.set_ylabel("RecHit Loc y (cm)", loc="center", size=20)

    ax_m.hist(ak.num(r.gemRecHit_chamber, axis = -1).to_numpy(), bins=100)
    ax_m.set_title("RecHits multiplicity per event", fontweight="bold", size=24)
    ax_m.set_xlabel("Hit multiplicity", loc="right", size=20)


    fig_r.savefig(Path(output_folder_path,f"recHits_GE21_XTalk.png"),dpi=120)                   
    fig_p.savefig(Path(output_folder_path,f"propHits_GE21_XTalk.png"),dpi=120)
    fig_m.savefig(Path(output_folder_path,f"recHits_GE21_MUltiplicityXTalk.png"),dpi=120)

    fig_r.savefig(Path(output_folder_path,f"recHits_GE21_XTalk.pdf"))
    fig_p.savefig(Path(output_folder_path,f"propHits_GE21_XTalk.pdf"))
    fig_m.savefig(Path(output_folder_path,f"recHits_GE21_MUltiplicityXTalk.pdf"))
