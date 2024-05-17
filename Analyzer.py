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
import hist
import pandas as pd
import guppy

from Utils import (
    heap_size,
    iEtaStrip_2_VFAT,
    iEta_2_chamberType,
    recHit2VFAT,
    EfficiencySummary,
    OUTPUT_PATH,
    PHPINDEX_FILE,
    ExtendEfficiencyCSV,
    ExtendROOTFile,
)
from MaskFunctions import (
    calcMuonHit_masks,
    countNumberOfPropHits,
    calcDAQMaskedVFAT_mask,
    calcDAQMissingVFAT_mask,
    calcDAQError_mask,
    calcDAQenabledOH_mask,
    calcHV_mask,
)

from AkArray_Operations import find_boundaries, boundaries_translation, best_match
from PlottingFunctions import Fill_Histo_Residuals, Store_Binned_Residuals
from config_parser import config
from myLogger import logger as logger_default

logger = logger_default.getLogger(__name__)
logger.setLevel(logger_default.INFO)
the_heap = guppy.hpy()
the_heap.setref()

## PARSER
parser = argparse.ArgumentParser(description="Analyzer parser")
parser.add_argument("config", help="Analysis description file")
parser.add_argument("--folder_name", type=str, help="Output folder name", required=False, default="")
parser.add_argument("--residuals", help="Enable plotting residuals", required=False, action="store_true", default=False)
parser.add_argument("--timestamp", type=str, help="label for unique analysis results", required=False, default=time.strftime("_%-y%m%d%H%M"))
parser.add_argument("--storeROOT", help="Enable the storing of best matches as a ROOT file", required=False, action="store_true", default=False)
args = parser.parse_args()


configuration = config(abspath(args.config))
input_par = configuration.parameters
matching_cuts = configuration.matching_window
match_by = "residual_rdphi"
output_folder_path = Path(OUTPUT_PATH, args.folder_name)
analysis_timestamp = args.timestamp
output_name = configuration.analysis_label + analysis_timestamp
max_evts = configuration.data_input["max_evts"]
runRange = sorted(configuration.data_input["runRange"])
## PARSER

## CREATE FOLDERS and COPY FILES
output_folder_path.mkdir(parents=True, exist_ok=True)
#file_index= str(PHPINDEX_FILE).split("/")[-1]
#if not Path(join(output_folder_path,file_index)).is_file():
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, output_folder_path)

if args.residuals:
    residual_output_folder_path = Path(output_folder_path, f"Residuals{analysis_timestamp}")
    residual_output_folder_path.mkdir(parents=True, exist_ok=True)
    if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, residual_output_folder_path)

    TrackQuality_output_folder_path = Path(output_folder_path,f"TrackQuality{analysis_timestamp}")
    TrackQuality_output_folder_path.mkdir(parents=True, exist_ok=True)
    if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,TrackQuality_output_folder_path)
    
    TrackQualityLayer_output_folder_path = Path(TrackQuality_output_folder_path,"Layer")
    TrackQualityLayer_output_folder_path.mkdir(parents=True, exist_ok=True)
    if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,TrackQualityLayer_output_folder_path)

    TrackQualityResidual_output_folder_path = Path(TrackQuality_output_folder_path,"Residuals")
    TrackQualityResidual_output_folder_path.mkdir(parents=True, exist_ok=True)
    if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,TrackQualityResidual_output_folder_path)

    TrackQualityCharge_output_folder_path = Path(TrackQuality_output_folder_path,"Charge")
    TrackQualityCharge_output_folder_path.mkdir(parents=True, exist_ok=True)
    if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,TrackQualityCharge_output_folder_path)

    TrackQualitySize_output_folder_path = Path(TrackQuality_output_folder_path,"ChamberSize")
    TrackQualitySize_output_folder_path.mkdir(parents=True, exist_ok=True)
    if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,TrackQualitySize_output_folder_path)
##

avg_batch_size = 600 # MB

total_events = 0

def main():
    heap_size(the_heap, "starting")

    ROOTFile = None
    if args.storeROOT:
        ROOTFile =  uproot.recreate(OUTPUT_PATH / f"{output_name}.root")
        
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
        "gemRecHit_g_phi",
        "gemRecHit_firstClusterStrip",
        "gemRecHit_station",
        "gemRecHit_cluster_size",
    ]
    branches_prophit = [
        "mu_propagated_charge",
        "mu_propagated_station",
        "mu_propagated_region",
        "mu_propagated_chamber",
        "mu_propagated_layer",
        "mu_propagated_etaP",
        "mu_propagated_strip",
        "mu_propagated_isME11",
        "mu_propagatedGlb_r",
        "mu_propagatedGlb_phi",
        "mu_propagatedGlb_errR",
        "mu_propagatedGlb_errPhi",
        "mu_propagatedLoc_dirX",
        "mu_propagatedLoc_dirY",
        "mu_propagated_pt",
        "mu_propagated_TrackNormChi2",
        "mu_propagated_nSTAHits",
        "mu_propagated_nME1hits",
        "mu_propagated_nME2hits",
        "mu_propagated_nME3hits",
        "mu_propagated_nME4hits",
        "mu_propagated_isME21",
    ]

    GE21_phi_range = 0.3631972
    ## boundaries values for iPhi in terms of cos(loc_x / glb_r)
    GE11_cos_phi_boundaries = (
        -0.029824804,
        0.029362374,
    )
    global total_events


    for batch_index, b in enumerate(batches):
        if max_evts != -1 and total_events >= max_evts:
            logger.warning(f"Processed at least {max_evts} events, hitting the max_evt cap. Exiting loop")
            break

        logger.info(f"Processing file batch {batch_index+1}/{len(batches)}")
        logger.debug(f" Loading branches into awkward arrays")
        event = uproot.concatenate(b, filter_name="*event*")
        heap_size(the_heap, "after loading the event branch")

        gemRecHit = uproot.concatenate(b, filter_name=branches_rechit)
        heap_size(the_heap, "after loading the rechit branch")
        gemPropHit = uproot.concatenate(b, filter_name=branches_prophit)
        heap_size(the_heap, "after loading the prophit branch")
        gemOHStatus = uproot.concatenate(b, filter_name="*gemOHStatus*")
        heap_size(the_heap, "after loading the gemohstatus branch")

        runNumber_mask = (event.event_runNumber >= runRange[0]) & (event.event_runNumber <= runRange[1])
        if ak.sum(runNumber_mask) == 0:
            logger.warning(f"Skipping current batch due to run number not in range")
            del event
            print()
            continue  ## Run numbers out of range  in this file batch
        logger.debug(f" Selecting on run number")
        gemPropHit = gemPropHit[runNumber_mask]
        gemRecHit = gemRecHit[runNumber_mask]
        gemOHStatus = gemOHStatus[runNumber_mask]
        event = event[runNumber_mask]
        heap_size(the_heap, "after filtering on run number")

        #gemPropHit["ImpactAngle"] = np.arccos(np.sqrt(gemPropHit["mu_propagatedLoc_dirX"]**2 + gemPropHit["mu_propagatedLoc_dirY"]**2))*180/np.pi

        
        ## skimming on nevents
        n_events = len(event)
        if total_events + n_events > max_evts and max_evts != -1:
            select = max_evts - total_events

            event = event[0:select]
            gemRecHit = gemRecHit[0:select]
            gemPropHit = gemPropHit[0:select]
            gemOHStatus = gemOHStatus[0:select]
            total_events += select
        else:
            total_events += n_events
        logger.info(f"\033[4m\033[1m{len(event)} evts\033[0m")
        

        logger.debug(f" Add event info in the gemPropHit,gemRecHit,gemOHStatus arrays")
        gemPropHit["prop_eventNumber"] = ak.broadcast_arrays(event.event_eventNumber, gemPropHit["mu_propagated_pt"])[0]
        gemPropHit["prop_RunNumber"] = ak.broadcast_arrays(event.event_runNumber, gemPropHit["mu_propagated_pt"])[0]
        gemPropHit["mu_propagated_lumiblock"] = ak.broadcast_arrays(event.event_lumiBlock, gemPropHit["mu_propagated_pt"])[0]
        gemRecHit["rec_eventNumber"] = ak.broadcast_arrays(event.event_eventNumber, gemRecHit["gemRecHit_chamber"])[0]
        gemRecHit["gemRecHit_lumiblock"] = ak.broadcast_arrays(event.event_lumiBlock, gemRecHit["gemRecHit_chamber"])[0]
        gemOHStatus["gemOHStatus_lumiblock"] = ak.broadcast_arrays(event.event_lumiBlock, gemOHStatus["gemOHStatus_station"])[0]
        gemOHStatus["gemOHStatus_eventNumber"] = ak.broadcast_arrays(event.event_eventNumber, gemOHStatus["gemOHStatus_station"])[0]
            
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

        ## Selecting events having at least 1 prop hit on GE11
        #atLeast1GE11Prop = ak.any(gemPropHit.mu_propagated_station == station, axis=-1)
        #gemPropHit = gemPropHit[atLeast1GE11Prop]
        #gemRecHit = gemRecHit[atLeast1GE11Prop]
        #gemOHStatus = gemOHStatus[atLeast1GE11Prop]
        ## Excluding propagated hits from GE21
        #gemPropHit = gemPropHit[gemPropHit.mu_propagated_station == station ]
        #gemPropHit = gemPropHit[gemPropHit.mu_propagated_region == 1 ]
        # Removing all GE21 prophits (no additional events removed)
        #heap_size(the_heap, "after filtering out GE21")
        
        logger.debug(f" Extracting eta partition boundaries")
        etaID_boundaries_akarray_pre = ak.Array(map(find_boundaries, gemPropHit.prop_etaID))
        heap_size(the_heap, "arraying etaP boundaries")

        logger.debug(f" Adding propHit VFAT")
        # GE11 approach to VFAT extrapolation
        # gemPropHit["prophit_cosine"] = gemPropHit.mu_propagatedLoc_x / gemPropHit.mu_propagatedGlb_r
        # gemPropHit["mu_propagated_phiP"] = (
        #     (gemPropHit["prophit_cosine"] <= GE11_cos_phi_boundaries[0]) * 0 +
        #     ((gemPropHit["prophit_cosine"] <= GE11_cos_phi_boundaries[1]) & (gemPropHit["prophit_cosine"] > GE11_cos_phi_boundaries[0]))* 1 +
        #     (gemPropHit["prophit_cosine"] > GE11_cos_phi_boundaries[1]) * 2
        # )


        # if I know the strip number i should get the "phiP" easily as the integer devision by 128 for GE11 - how is GE21 mapped?
 
        #gemPropHit["mu_propagated_phiP"] = ak.values_astype(gemPropHit["mu_propagated_phiP"], np.short)
        #gemPropHit["mu_propagated_VFAT"] = iEtaiPhi_2_VFAT(gemPropHit.mu_propagated_etaP, gemPropHit.mu_propagated_phiP)
        
        gemPropHit["mu_propagated_VFAT"] = iEtaStrip_2_VFAT(gemPropHit.mu_propagated_etaP, gemPropHit.mu_propagated_strip, gemPropHit.mu_propagated_station)
        gemPropHit["mu_propagated_chamberType"]= iEta_2_chamberType(gemPropHit.mu_propagated_etaP,gemPropHit.mu_propagated_layer, gemPropHit.mu_propagated_station)
        gemRecHit["gemRecHit_VFAT"] = recHit2VFAT(
            gemRecHit.gemRecHit_etaPartition,
            gemRecHit.gemRecHit_firstClusterStrip,
            gemRecHit.gemRecHit_cluster_size,
            gemRecHit.gemRecHit_station
        )
        gemRecHit["gemRecHit_chamberType"] = iEta_2_chamberType(gemRecHit.gemRecHit_etaPartition, gemRecHit.gemRecHit_layer, gemRecHit.gemRecHit_station)
        # GE21 approach to VFAT extrapolation TO BE TESTED
        # gemPropHit["propagated_offset_phi"] =  np.arcsin(  gemPropHit.mu_propagatedLoc_x/gemPropHit.mu_propagatedGlb_r )
        # gemPropHit["propagated_VFAT"] =  (11 - (gemPropHit.propagated_offset_phi - ak.min(gemPropHit.propagated_offset_phi)) // (GE21_phi_range/6)) - ((gemPropHit.mu_propagated_etaP-1)%4)//2
        # gemRecHit["gemRecHit_VFAT"] = (11 - ((gemRecHit.gemRecHit_firstClusterStrip+gemRecHit.gemRecHit_cluster_size/2)//64)) - ((gemRecHit.gemRecHit_etaPartition - 1)%4)//2
        heap_size(the_heap, "after extrapolating on VFATs")

        
        #why do i need to the following? don't I know already the VFAT?
        """
        in CMSSW angles are defined in the [-pi,pi] range.
        PhiMin > PhiMax happens for chambers 19 where phiMin = 174 degrees phiMax = -174.
        here the fix.
        find which boundaries have phiMin > phiMAx (index 1 corresponds to phiMin, 0 to phiMax)
        """
        logger.debug(f" Calculating masks")
        logger.debug2(f" Adjusting for angle periodicity")
        boundary_translation_mask = etaID_boundaries_akarray_pre[..., 1] > etaID_boundaries_akarray_pre[..., 0]
        ## Generate translation array. For every entry in mask, generate a boundary translation array
        ## mask is True --> translate by [2 pi , 0 , 0, 0]
        ## mask is False --> translate by [0 , 0 , 0, 0]
        ## Then add the aforementioned array of translation arrays to etaID_boundaries_akarray
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
        
        #print("etaID_boundaries_akarray ", etaID_boundaries_akarray)
        #print("etaID_boundaries_akarray[...,1] ", etaID_boundaries_akarray[...,1])
        muonTrack_mask = calcMuonHit_masks(**input_par)
        
        logger.debug2(f" Restoring angle periodicity")
        gemPropHit["mu_propagatedGlb_phi"] = gemPropHit.mu_propagatedGlb_phi - propGlbPhi_translation_mask * 2 * np.pi

        logger.debug(f" Calculating selection mask on VFAT DAQ mask")
        DAQMaskedVFAT_mask = calcDAQMaskedVFAT_mask(gemPropHit, gemOHStatus)
        muonTrack_mask["DAQMaskedVFAT"] = DAQMaskedVFAT_mask
        #print ("DAQMaskedVFAT_mask",  DAQMaskedVFAT_mask)
        muonTrack_mask["overallGood_Mask"] = ak.where(input_par["gemprophit_array"].mu_propagated_station<2, muonTrack_mask["overallGood_Mask"] & DAQMaskedVFAT_mask, muonTrack_mask["overallGood_Mask"])
        heap_size(the_heap, "after calculating VFAT DAQ mask")

        logger.debug(f" Calculating selection mask on VFAT DAQ missing")
        DAQMissingVFAT_mask = calcDAQMissingVFAT_mask(gemPropHit, gemOHStatus)
        muonTrack_mask["DAQMissingVFAT"] = DAQMissingVFAT_mask
        muonTrack_mask["overallGood_Mask"] = ak.where(input_par["gemprophit_array"].mu_propagated_station<2, muonTrack_mask["overallGood_Mask"] & DAQMissingVFAT_mask, muonTrack_mask["overallGood_Mask"])
        heap_size(the_heap, "after calculating VFAT DAQ missing")

        logger.debug(f" Calculating selection mask on DAQ error")
        DAQError_mask = calcDAQError_mask(gemPropHit, gemOHStatus)
        muonTrack_mask["DAQError"] = DAQError_mask
        muonTrack_mask["overallGood_Mask"] = ak.where(input_par["gemprophit_array"].mu_propagated_station<2, muonTrack_mask["overallGood_Mask"] & DAQError_mask, muonTrack_mask["overallGood_Mask"])
        heap_size(the_heap, "after calculating DAQ error mask")

        logger.debug(f" Calculating selection mask on DAQ enabled OH")
        DAQenabledOH_mask = calcDAQenabledOH_mask(gemPropHit, gemOHStatus)
        muonTrack_mask["DAQenabledOH"] = DAQenabledOH_mask
        muonTrack_mask["overallGood_Mask"] = ak.where(input_par["gemprophit_array"].mu_propagated_station<2, muonTrack_mask["overallGood_Mask"] & DAQenabledOH_mask, muonTrack_mask["overallGood_Mask"])
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
        selectedPropHit = gemPropHit[muonTrack_mask["overallGood_Mask"]]
        ## RECHIT SELECTION: keep ONLY events with at least 1 valid prophit
        selectedRecHit = gemRecHit[ak.any(muonTrack_mask["overallGood_Mask"], axis=-1)]
        ## Removing empty events, corresponding to discarded prophits
        event_hasProphits = ak.count(selectedPropHit.mu_propagated_chamber, axis=-1) != 0
        selectedPropHit = selectedPropHit[event_hasProphits]

        del gemPropHit
        del gemRecHit
        del gemOHStatus
        heap_size(the_heap, "after applying cuts")

        # Matching propHits and recHits belonging to the SAME evt and in the same etapartitionID
        logger.debug(f" Pairing etaID")
        ## how many times the prophit's eventnumber matches the rechit's eventnumber
        evtHasRechits = ak.num(selectedRecHit.rec_eventNumber)>0
        syncd_events = ak.sum(ak.any(selectedPropHit[evtHasRechits].prop_eventNumber[..., 0] == selectedRecHit[evtHasRechits].rec_eventNumber,axis=-1))
        n_valid_events = ak.sum(evtHasRechits)
        if syncd_events != n_valid_events:
            logger.error(
                f"PropHit and RecHit have mismatches on eventNumber.\nN of valid events = {n_valid_events}\t Rechit with same evt number = {syncd_events}"
            )

        ## all possible pairs of (propHit,recHit) for each event
        product = ak.cartesian({"prop": selectedPropHit.prop_etaID, "rec": selectedRecHit.rec_etaID})
        cartesian_indeces = ak.argcartesian({"prop": selectedPropHit.prop_etaID, "rec": selectedRecHit.rec_etaID})

        ## For dumping events only
        #my_gemPropHit = selectedPropHit[ (selectedPropHit.mu_propagated_station == 1) & (selectedPropHit.mu_propagated_region == -1) &(selectedPropHit.mu_propagated_chamber == 10) & (selectedPropHit.mu_propagated_layer == 2) ]
        #useful_events = ak.num(my_gemPropHit.mu_propagated_chamber)!=0
        #my_gemPropHit = my_gemPropHit[useful_events]
        #my_gemRecHits = selectedRecHit[useful_events]
        #side_product = ak.cartesian({"prop": my_gemPropHit.prop_etaID, "rec": my_gemRecHits.rec_etaID})
        #events_unmatched = ak.all(side_product["prop"] != side_product["rec"],axis=-1)
        #number_of_unmatchable_prophits = ak.sum(events_unmatched)
        #logger.error(f"Found {number_of_unmatchable_prophits} unmatchable selected_prophits for GE11-M-10-L2\n")
        #print("Evt Number\tVFAT\LumiBlock")
        #for evt in range(len(my_gemPropHit)):
            #print(f"{my_gemPropHit[evt].prop_eventNumber}\t{my_gemPropHit[evt].mu_propagated_VFAT}\t{my_gemPropHit[evt].mu_propagated_lumiblock}")
        
        ## Compatible Combinations: propHit,recHit in a pair have the same etaID
        matchable_event_mask = product["prop"] == product["rec"]
        matchable_prop_idx, matchable_rec_idx = ak.unzip(cartesian_indeces[matchable_event_mask])
        ## Crating an ak.array that contains all compatible hits
        ## Due to the cartesian product each prophit gets duplicated as many times as there are compatible rechits
        ## Example: evt 10 has prop_etaID = [1,2,60], rec_etaID = [1,1,1,1,1,60,60]
        ## the matchable prop will be [1,1,1,1,1,60]
        ## the matchable rec will be [1,1,1,1,1,60]
        Compatible_Dict = {
            **{key: selectedRecHit[key][matchable_rec_idx] for key in selectedRecHit.fields},
            **{key: selectedPropHit[key][matchable_prop_idx] for key in selectedPropHit.fields},
        }
        compatibleHitsArray = ak.Array(Compatible_Dict)
        heap_size(the_heap, "pairing hits")

        logger.debug(f" Calculating residuals")
        ## CMSSW has chosen to have angles in [-pi,pi]. In some events (<0.01%) residuals phi> pi
        ## fixing it
        # angle_translation_rec = (((compatibleHitsArray.mu_propagatedGlb_phi - compatibleHitsArray.gemRecHit_g_phi) > 5) * 2 * np.pi)
        # angle_translation_prop = (((compatibleHitsArray.mu_propagatedGlb_phi - compatibleHitsArray.gemRecHit_g_phi) < -5) * 2 * np.pi)
        #compatibleHitsArray["gemRecHit_g_phi"] = compatibleHitsArray["gemRecHit_g_phi"] + angle_translation_rec
        #compatibleHitsArray["mu_propagatedGlb_phi"] = (compatibleHitsArray["mu_propagatedGlb_phi"] + angle_translation_prop)

        ## Residual Calc
        compatibleHitsArray["residual_phi"] = (compatibleHitsArray.mu_propagatedGlb_phi - compatibleHitsArray.gemRecHit_g_phi)
        compatibleHitsArray["residual_rdphi"] = (compatibleHitsArray.mu_propagatedGlb_phi - compatibleHitsArray.gemRecHit_g_phi) * compatibleHitsArray.mu_propagatedGlb_r
        heap_size(the_heap, "calculating residuals")

        logger.debug(f" Cut on residuals for efficiency")
        best_matches = best_match(compatibleHitsArray, match_by)
        accepted_hits = best_matches[abs(best_matches[match_by]) < matching_cuts[match_by]]
        accepted_hits = accepted_hits[ak.num(accepted_hits.prop_etaID, axis=-1) > 0]
        heap_size(the_heap, " selection based on residuals")

        logger.debug2(f"Number of good prophits: {len(selectedPropHit)}")
        logger.debug2(f"Number of cartesian prophits (contains duplicates): {ak.sum(ak.num(compatibleHitsArray.mu_propagatedGlb_phi,axis=-1))}")
        logger.debug2(f"Number of matched prophits: {ak.sum(ak.num(accepted_hits.prop_etaID,axis=-1))}")


        heap_size(the_heap, "before cleaning")
        ## Dump array summary after processing each batch
        ExtendEfficiencyCSV(EfficiencySummary(accepted_hits,selectedPropHit),OUTPUT_PATH / f"{output_name}.csv")
        logger.info("Efficiency csv file updated")

        if args.storeROOT:
            """
            Store best matche hits that passed the cut window in a 
            ROOT file for future usage. 
            """
            df_prop = ak.to_dataframe(selectedPropHit)
            df_accepted = ak.to_dataframe(accepted_hits)
            df_output = pd.merge(df_prop, df_accepted,  how='left', left_on=selectedPropHit.fields, right_on = selectedPropHit.fields)
            ROOTFile = ExtendROOTFile(ROOTFile, df_output)


        for station in [0,1,2]:
            temp_m = accepted_hits[accepted_hits['gemRecHit_station']==station]
            temp_p = selectedPropHit[selectedPropHit['mu_propagated_station']==station]
            temp_den = ak.sum(ak.num(temp_p.prop_etaID,axis=-1))
            if temp_den!= 0:
                logger.info(f"AVG Efficiency station {station}: {ak.sum(ak.num(temp_m.prop_etaID,axis=-1))}/{temp_den} = {ak.sum(ak.count(temp_m.prop_etaID,axis=-1))/temp_den}")
            else:
                logger.info(f"AVG Efficiency station {station}: NO HITS")


        cutSummary_collector = {k: v for k, v in sorted(cutSummary_collector.items(), key=lambda item: item[1], reverse=True)}
        logger.info(f"")
        logger.info(f"Breakdown table with cuts")
        logger.info(f"{'Label':<20}\t{'Survived Hits':>15}\t{'% Survived':>20}")
        for k in cutSummary_collector:
            logger.info(f"{k:<20}\t{cutSummary_collector[k]:>15}\t{round(cutSummary_collector[k]*100/cutSummary_collector['no_Mask'],2):>19}%")
        print()

        
        del selectedPropHit
        del selectedRecHit
        del accepted_hits
        del event
        heap_size(the_heap, "at the loop end")
        

if __name__ == "__main__":

    main()
    ## SAVE CSV WITH RESIDUALS
    # mydf = ak.to_dataframe(compatible_collector)
    # mydf.to_csv(f"{residual_output_folder_path}/test.csv")

    """ ## PLOT DISTRIBUTION MARCELLO
    core_residual = 0.5
    fig_res, axs_res = plt.subplots(1, 2, figsize=(10, 10))
    fig_ly, axs_ly = plt.subplots(1, 2, figsize=(10, 10))
    fig_ratio = plt.figure(figsize=(10, 8))
    compatible_collector_core  =  compatible_collector[(compatible_collector[match_by] < core_residual) & (compatible_collector[match_by] > -core_residual)]
    compatible_collector_tails = compatible_collector[(compatible_collector[match_by] < -core_residual) | (compatible_collector[match_by] > core_residual)]
    compatible_collector_L1  =  compatible_collector[compatible_collector["mu_propagated_layer"] == 1]
    compatible_collector_L2 = compatible_collector[compatible_collector["mu_propagated_layer"] == 2]
    compatible_collector_positive  =  compatible_collector[compatible_collector["mu_propagated_charge"] == 1]
    compatible_collector_negative = compatible_collector[compatible_collector["mu_propagated_charge"] == -1]
    compatible_collector_short  =  compatible_collector[compatible_collector["mu_propagated_chamber"] %2 == 1]
    compatible_collector_long = compatible_collector[compatible_collector["mu_propagated_chamber"] %2 == 0]
    for k in [match_by, "mu_propagated_nSTAHits",  "mu_propagated_nME1hits", "mu_propagated_nME2hits", "mu_propagated_nME3hits", "mu_propagated_nME4hits", "mu_propagated_isRPC", "mu_propagated_TrackNormChi2", "mu_propagated_pt", "mu_propagatedGlb_errR", "mu_propagatedGlb_errPhi", "mu_propagated_layer", "mu_propagated_etaP", "mu_propagated_region", "gemRecHit_cluster_size","mu_propagated_charge","ImpactAngle","mu_propagatedLoc_dirY","mu_propagatedLoc_dirX"]:
        
        min_x_tails = ak.min(compatible_collector_tails[k])
        min_x_core = ak.min(compatible_collector_core[k])
        max_x_tails = min(ak.max(compatible_collector_tails[k]), 500)
        max_x_core = min(ak.max(compatible_collector_core[k]), 500)
        
        min_x_L1 = ak.min(compatible_collector_L1[k])
        min_x_L2 = ak.min(compatible_collector_L2[k])
        max_x_L1 = min(ak.max(compatible_collector_L1[k]), 500)
        max_x_L2 = min(ak.max(compatible_collector_L2[k]), 500)
        
        min_x_pos = ak.min(compatible_collector_positive[k])
        min_x_neg = ak.min(compatible_collector_negative[k])
        max_x_pos = min(ak.max(compatible_collector_positive[k]), 500)
        max_x_neg = min(ak.max(compatible_collector_negative[k]), 500)
        
        min_x_long = ak.min(compatible_collector_long[k])
        min_x_short = ak.min(compatible_collector_short[k])
        max_x_long = min(ak.max(compatible_collector_long[k]), 500)
        max_x_short = min(ak.max(compatible_collector_short[k]), 500)
        
        if int(min_x_tails) - min_x_tails == 0 and int(min_x_core) - min_x_core == 0 and int(max_x_tails) - max_x_tails == 0  and int(max_x_core) - max_x_core == 0:
            discrete_distr = True
        else: discrete_distr = False
        
        
        min_x_res = min(min_x_tails,min_x_core) if k != match_by else -5 ## common min
        max_x_res = min(max_x_tails,max_x_core) if k != match_by else 5  ## common max
        min_x_ly = min(min_x_L1,min_x_L2) if k != match_by else -5 ## common min
        max_x_ly = min(max_x_L1,max_x_L2) if k != match_by else 5  ## common max
        min_x_chrg = min(min_x_neg,min_x_pos) if k != match_by else -5 ## common min
        max_x_chrg = min(max_x_neg,max_x_pos) if k != match_by else 5  ## common max
        min_x_size = min(min_x_short,min_x_long) if k != match_by else -5 ## common min
        max_x_size = min(max_x_long,max_x_short) if k != match_by else 5  ## common max

        if discrete_distr:
            min_x_res = min_x_res - 1 ## reducing by 1.5
            min_x_ly = min_x_ly - 1 ## reducing by 1.5
            min_x_chrg = min_x_chrg - 1 ## reducing by 1.5
            min_x_size = min_x_size - 1 ## reducing by 1.5
            max_x_res = max_x_res + 1 ## increasing by 1.5
            max_x_ly = max_x_ly + 1 ## increasing by 1.5
            max_x_chrg = max_x_chrg + 1 ## increasing by 1.5
            max_x_size = max_x_size + 1 ## increasing by 1.5
            
            nbins_res = int(max_x_res - min_x_res)
            nbins_ly = int(max_x_ly - min_x_ly)
            nbins_chrg = int(max_x_chrg - min_x_chrg)
            nbins_size = int(max_x_size - min_x_size)
        else:
            min_x_res = min_x_res - min_x_res*0.1 ## reducing by 10%
            min_x_chrg = min_x_chrg - min_x_chrg*0.1 ## reducing by 10%
            min_x_ly = min_x_ly - min_x_ly*0.1 ## reducing by 10%
            min_x_size = min_x_size - min_x_size*0.1 ## reducing by 10%
            max_x_res = max_x_res + max_x_res*0.1 ## increasing by 10%
            max_x_ly = max_x_ly + max_x_ly*0.1 ## increasing by 10%
            max_x_chrg = max_x_chrg + max_x_chrg*0.1 ## increasing by 10%
            max_x_size = max_x_size + max_x_size*0.1 ## increasing by 10%

            nbins_res = 100
            nbins_ly = 100
            nbins_chrg = 100
            nbins_size = 100
        logger.debug(f"Plotting histogram for {k}")    
        
        
        h_core = hist.Hist(hist.axis.Regular(nbins_res, min_x_res, max_x_res, label=f"Core {k}")).fill( ak.flatten(compatible_collector_core[k]))
        h_tails = hist.Hist(hist.axis.Regular(nbins_res, min_x_res, max_x_res, label=f"Tails {k}")).fill( ak.flatten(compatible_collector_tails[k]))
        h_L1 = hist.Hist(hist.axis.Regular(nbins_ly, min_x_ly, max_x_ly, label=f"Ly1 {k}")).fill( ak.flatten(compatible_collector_L1[k]))
        h_L2 = hist.Hist(hist.axis.Regular(nbins_ly, min_x_ly, max_x_ly, label=f"Ly2 {k}")).fill( ak.flatten(compatible_collector_L2[k]))
        h_pos = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_positive[k]))
        h_neg = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_negative[k]))
        h_long = hist.Hist(hist.axis.Regular(nbins_size, min_x_size, max_x_size, label=f"{k}")).fill( ak.flatten(compatible_collector_long[k]))
        h_short = hist.Hist(hist.axis.Regular(nbins_size, min_x_size, max_x_size, label=f"{k}")).fill( ak.flatten(compatible_collector_short[k]))

        if k != match_by:
            
            main_ax_artists, sublot_ax_arists = h_core.plot_ratio(
                h_tails,
                rp_ylabel=r"Ratio Core / Tails",
                rp_num_label="Core",
                rp_denom_label="Tails",
                rp_uncert_draw_type="bar",  # line or bar
            )
            fig_ratio.savefig(f"{TrackQualityResidual_output_folder_path}/TailCore_Ratio_{k}.png")
            fig_ratio.savefig(f"{TrackQualityResidual_output_folder_path}/TailCore_Ratio_{k}.pdf")
            fig_ratio.clear()

        
        if k != "mu_propagated_layer":
            main_ax_artists, sublot_ax_arists = h_L1.plot_ratio(
                h_L2,
                rp_ylabel=r"Ratio Ly 1 / Ly 2",
                rp_num_label="Layer1",
                rp_denom_label="Layer2",
                rp_uncert_draw_type="bar",  # line or bar
            )
            fig_ratio.savefig(f"{TrackQualityLayer_output_folder_path}/Layer_Ratio_{k}.png")
            fig_ratio.savefig(f"{TrackQualityLayer_output_folder_path}/Layer_Ratio_{k}.pdf")
            fig_ratio.clear()
        if k != "mu_propagated_charge":
            main_ax_artists, sublot_ax_arists = h_neg.plot_ratio(
                h_pos,
                rp_ylabel=r"Ratio NegaMuon / PosiMuon",
                rp_num_label="NegaMuon",
                rp_denom_label="PosiMuon",
                rp_uncert_draw_type="bar",  # line or bar
            )
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/Charge_Ratio_{k}.png")
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/Charge_Ratio_{k}.pdf")
            fig_ratio.clear()
        
        if k == match_by:
            compatible_collector_positive_far = compatible_collector_positive[compatible_collector_positive["mu_propagatedGlb_r"] > 170]
            compatible_collector_positive_near = compatible_collector_positive[compatible_collector_positive["mu_propagatedGlb_r"] <= 170]
            compatible_collector_negative_far = compatible_collector_negative[compatible_collector_negative["mu_propagatedGlb_r"] > 170]
            compatible_collector_negative_near = compatible_collector_negative[compatible_collector_negative["mu_propagatedGlb_r"] <= 170]
            
            compatible_collector_positive_perp = compatible_collector_positive[compatible_collector_positive["ImpactAngle"] > 70]
            compatible_collector_positive_obli = compatible_collector_positive[compatible_collector_positive["ImpactAngle"] <= 70]
            compatible_collector_negative_perp = compatible_collector_negative[compatible_collector_negative["ImpactAngle"] > 70]
            compatible_collector_negative_obli = compatible_collector_negative[compatible_collector_negative["ImpactAngle"] <= 70]


            h_pos_far = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_positive_far[k]))
            h_pos_near = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_positive_near[k]))
            h_neg_far = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_negative_far[k]))
            h_neg_near = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_negative_near[k]))
            
            h_pos_perp = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_positive_perp[k]))
            h_pos_obli = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_positive_obli[k]))
            h_neg_perp = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_negative_perp[k]))
            h_neg_obli = hist.Hist(hist.axis.Regular(nbins_chrg, min_x_chrg, max_x_chrg, label=f"{k}")).fill( ak.flatten(compatible_collector_negative_obli[k]))
            main_ax_artists, sublot_ax_arists = h_pos_far.plot_ratio(
                h_neg_far,
                rp_ylabel=r"Ratio Positve GlbR>170 / Negative GlbR>170",
                rp_num_label="Positve GlbR>170",
                rp_denom_label="Negative GlbR>170",
                rp_uncert_draw_type="bar",  # line or bar
            )
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/Far_R_{k}.png")
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/Far_R_{k}.pdf")
            fig_ratio.clear()
            
            main_ax_artists, sublot_ax_arists = h_pos_near.plot_ratio(
                h_neg_near,
                rp_ylabel=r"Ratio Positve GlbR<=170 / Negative GlbR<=170",
                rp_num_label="Positive GlbR<=170",
                rp_denom_label="Negative GlbR<=170",
                rp_uncert_draw_type="bar",  # line or bar
            )
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/Near_R_{k}.png")
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/Near_R_{k}.pdf")
            fig_ratio.clear()
            main_ax_artists, sublot_ax_arists = h_pos_perp.plot_ratio(
                h_neg_perp,
                rp_ylabel=r"Ratio Positive Perp / Negative Perp",
                rp_num_label="Positve angle>70°",
                rp_denom_label="Negative angle>70°",
                rp_uncert_draw_type="bar",  # line or bar
            )
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/PerpTracks_{k}.png")
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/PerpTracks_{k}.pdf")
            fig_ratio.clear()
            
            main_ax_artists, sublot_ax_arists = h_pos_obli.plot_ratio(
                h_neg_obli,
                rp_ylabel=r"Ratio Positive Obli / Negative Obli",
                rp_num_label="Positve angle<70°",
                rp_denom_label="Negative angle<70°",
                rp_uncert_draw_type="bar",  # line or bar
            )
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/ObliTracks_{k}.png")
            fig_ratio.savefig(f"{TrackQualityCharge_output_folder_path}/ObliTracks_{k}.pdf")
            fig_ratio.clear()


        main_ax_artists, sublot_ax_arists = h_short.plot_ratio(
            h_long,
            rp_ylabel=r"Ratio short / long",
            rp_num_label="short",
            rp_denom_label="long",
            rp_uncert_draw_type="bar",  # line or bar
        )
        fig_ratio.savefig(f"{TrackQualitySize_output_folder_path}/ChamberSize_Ratio_{k}.png")
        fig_ratio.savefig(f"{TrackQualitySize_output_folder_path}/ChamberSize_Ratio_{k}.pdf")
        fig_ratio.clear()


        h_core.plot(ax=axs_res[0],color="orange")
        h_tails.plot(ax=axs_res[1],color="darkviolet")    
        fig_res.savefig(f"{TrackQualityResidual_output_folder_path}/TailCore_Hist_{k}.png")
        fig_res.savefig(f"{TrackQualityResidual_output_folder_path}/TailCore_Hist_{k}.pdf")
        axs_res[0].clear()
        axs_res[1].clear()
        
        h_L1.plot(ax=axs_ly[0],color="orange")
        h_L2.plot(ax=axs_ly[1],color="darkviolet")    
        fig_ly.savefig(f"{TrackQualityLayer_output_folder_path}/Layer_Hist_{k}.png")
        fig_ly.savefig(f"{TrackQualityLayer_output_folder_path}/Layer_Hist_{k}.pdf")
        axs_ly[0].clear()
        axs_ly[1].clear()
    ## END PLOT DISTRIBUTION MARCELLO """

    # start = time.time()
    # df = EfficiencySummary(matched, prop)
    # logger.info(f"Summary generated in {time.time()-start:.3f} s")
    # ExtendEfficiencyCSV(df, OUTPUT_PATH / f"{output_name}.csv")
    # configuration.dump_config(OUTPUT_PATH / f"{output_name}.yml")
