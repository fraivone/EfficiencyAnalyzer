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
parser.add_argument("--folder_name", type=str, help="Output folder name", required=False, default="test")
parser.add_argument("--residuals", help="Enable plotting residuals", required=False, action="store_true", default=False)
parser.add_argument("--test", help="Test run", required=False, action="store_true", default=False)
parser.add_argument("--timestamp", type=str, help="label for unique analysis results", required=False, default=time.strftime("_%-y%m%d%H%M"))
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
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, output_folder_path)
if args.residuals:
    residual_output_folder_path = Path(output_folder_path, f"Residuals{analysis_timestamp}")
    residual_output_folder_path.mkdir(parents=True, exist_ok=True)
    if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, residual_output_folder_path)

DAQStatus_output_folder_path = Path(output_folder_path,"DAQStatus")
DAQStatus_output_folder_path.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,DAQStatus_output_folder_path)

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

avg_batch_size = 600 if args.test == False else 600 # MB
heap_dump_size = 500  # MB above which efficiency summary gets calculated from propagated_collector and matched_collector and dumped to file
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
    selected_prop_perEvent = []
    selected_rech_perEvent = []

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
        "mu_propagated_charge",
        "mu_propagated_station",
        "mu_propagated_region",
        "mu_propagated_chamber",
        "mu_propagated_layer",
        "mu_propagated_etaP",
        "mu_propagated_Outermost_z",
        "mu_propagated_isME11",
        "mu_propagated_isRPC",
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
        "mu_propagatedLoc_dirZ",
        "mu_propagated_pt",
        "mu_propagated_isGEM",
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
    station = 1 ## GE11 Only

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
        atLeast1GE11Prop = ak.any(gemPropHit.mu_propagated_station == station, axis=-1)
        gemPropHit = gemPropHit[atLeast1GE11Prop]
        gemRecHit = gemRecHit[atLeast1GE11Prop]
        gemOHStatus = gemOHStatus[atLeast1GE11Prop]
        ## Excluding propagated hits from GE21
        gemPropHit = gemPropHit[gemPropHit.mu_propagated_station == station]
        # Removing all GE21 prophits (no additional events removed)
        heap_size(the_heap, "after filtering out GE21")

        logger.debug(f" Extracting eta partition boundaries")
        etaID_boundaries_akarray_pre = ak.Array(map(find_boundaries, gemPropHit.prop_etaID))
        heap_size(the_heap, "arraying etaP boundaries")

        logger.debug(f" Extrapolating onto VFATs")
        # GE11 approach to VFAT extrapolation
        gemPropHit["prophit_cosine"] = gemPropHit.mu_propagatedLoc_x / gemPropHit.mu_propagatedGlb_r
        gemPropHit["mu_propagated_phiP"] = (
            (gemPropHit["prophit_cosine"] <= GE11_cos_phi_boundaries[0]) * 0 +
            ((gemPropHit["prophit_cosine"] <= GE11_cos_phi_boundaries[1]) & (gemPropHit["prophit_cosine"] > GE11_cos_phi_boundaries[0]))* 1 +
            (gemPropHit["prophit_cosine"] > GE11_cos_phi_boundaries[1]) * 2
        )
        gemPropHit["mu_propagated_phiP"] = ak.values_astype(gemPropHit["mu_propagated_phiP"], np.short)
        gemPropHit["mu_propagated_VFAT"] = iEtaiPhi_2_VFAT(gemPropHit.mu_propagated_etaP, gemPropHit.mu_propagated_phiP)
        gemRecHit["gemRecHit_VFAT"] = recHit2VFAT(
            gemRecHit.gemRecHit_etaPartition,
            gemRecHit.gemRecHit_firstClusterStrip,
            gemRecHit.gemRecHit_cluster_size,
        )
        # GE21 approach to VFAT extrapolation TO BE TESTED
        # gemPropHit["propagated_offset_phi"] =  np.arcsin(  gemPropHit.mu_propagatedLoc_x/gemPropHit.mu_propagatedGlb_r )
        # gemPropHit["propagated_VFAT"] =  (11 - (gemPropHit.propagated_offset_phi - ak.min(gemPropHit.propagated_offset_phi)) // (GE21_phi_range/6)) - ((gemPropHit.mu_propagated_etaP-1)%4)//2
        # gemRecHit["gemRecHit_VFAT"] = (11 - ((gemRecHit.gemRecHit_firstClusterStrip+gemRecHit.gemRecHit_cluster_size/2)//64)) - ((gemRecHit.gemRecHit_etaPartition - 1)%4)//2
        heap_size(the_heap, "after extrapolating on VFATs")

        logger.debug(f" Calculating masks")
        logger.debug2(f" Adjusting for angle periodicity")
        # in CMSSW angles are defined in the [-pi,pi] range.
        # PhiMin > PhiMax happens for chambers 19 where phiMin = 174 degrees phiMax = -174.
        # here the fix.
        ## find which boundaries have phiMin > phiMAx (index 1 corresponds to phiMin, 0 to phiMax)
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
        muonTrack_mask = calcMuonHit_masks(**input_par)

        logger.debug2(f" Restoring angle periodicity")
        gemPropHit["mu_propagatedGlb_phi"] = gemPropHit.mu_propagatedGlb_phi - propGlbPhi_translation_mask * 2 * np.pi

        logger.debug(f" Calculating selection mask on VFAT DAQ mask")
        DAQMaskedVFAT_mask = calcDAQMaskedVFAT_mask(gemPropHit, gemOHStatus)
        muonTrack_mask["DAQMaskedVFAT"] = DAQMaskedVFAT_mask
        muonTrack_mask["overallGood_Mask"] = muonTrack_mask["overallGood_Mask"] & DAQMaskedVFAT_mask
        heap_size(the_heap, "after calculating VFAT DAQ mask")

        logger.debug(f" Calculating selection mask on VFAT DAQ missing")
        DAQMissingVFAT_mask = calcDAQMissingVFAT_mask(gemPropHit, gemOHStatus)
        muonTrack_mask["DAQMissingVFAT"] = DAQMissingVFAT_mask
        muonTrack_mask["overallGood_Mask"] = muonTrack_mask["overallGood_Mask"] & DAQMissingVFAT_mask
        heap_size(the_heap, "after calculating VFAT DAQ missing")

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
        syncd_events = ak.sum(ak.any(selectedPropHit.prop_eventNumber[..., 0] == selectedRecHit.rec_eventNumber,axis=-1))
        total_prophit = ak.sum(ak.num(selectedPropHit.prop_etaID), axis=-1)
        if syncd_events != len(selectedPropHit):
            logger.error(
                f"PropHit and RecHit have mismatches on eventNumber.\nTotal prophit = {total_prophit}\t Rechit with same evt number = {syncd_events}"
            )
        selected_prop_perEvent.append(ak.to_list(ak.num(selectedPropHit.prop_etaID)))
        selected_rech_perEvent.append(ak.to_list(ak.num(selectedRecHit.rec_etaID)))

        ## all possible pairs of (propHit,recHit) for each event
        product = ak.cartesian({"prop": selectedPropHit.prop_etaID, "rec": selectedRecHit.rec_etaID})
        cartesian_indeces = ak.argcartesian({"prop": selectedPropHit.prop_etaID, "rec": selectedRecHit.rec_etaID})
        
        ## Compatible Combinations: propHit,recHit in a pair have the same etaID
        # matchable_event_mask = product["prop"] == product["rec"]
        # matchable_prop_idx, matchable_rec_idx = ak.unzip(cartesian_indeces[matchable_event_mask])
        
        ## Compatible combinations: proprHit and rechit have the same layer, region, station 
        matchable_event_mask = ((product["prop"] & 2**6-1) == (product["rec"] & 2**6-1)) & ((product["prop"]>>11) != (product["rec"]>> 11)) 
        matchable_prop_idx, matchable_rec_idx = ak.unzip(cartesian_indeces[matchable_event_mask])
        
        
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
        # compatibleHitsArray["gemRecHit_g_phi"] = compatibleHitsArray["gemRecHit_g_phi"] + angle_translation_rec
        # compatibleHitsArray["mu_propagatedGlb_phi"] = (compatibleHitsArray["mu_propagatedGlb_phi"] + angle_translation_prop)

        ## Residual Calc
        offset_angle = (compatibleHitsArray.gemRecHit_chamber - compatibleHitsArray.mu_propagated_chamber)  * np.pi/18
        compatibleHitsArray["residual_phi"] = (compatibleHitsArray.mu_propagatedLoc_x - compatibleHitsArray.gemRecHit_loc_x)
        compatibleHitsArray["residual_rdphi"] = (compatibleHitsArray.mu_propagatedGlb_phi - compatibleHitsArray.gemRecHit_g_phi + offset_angle) * compatibleHitsArray.mu_propagatedGlb_r
        heap_size(the_heap, "calculating residuals")

        logger.debug(f" Cut on residuals for efficiency")
        best_matches = best_match(compatibleHitsArray, match_by)
        accepted_hits = best_matches[best_matches[match_by] < matching_cuts[match_by]]
        accepted_hits = accepted_hits[ak.num(accepted_hits.prop_etaID, axis=-1) > 0]
        heap_size(the_heap, "after selection based on residuals")

        matched_collector = (ak.concatenate([matched_collector, accepted_hits]) if matched_collector is not None else accepted_hits)
        propagated_collector = (ak.concatenate([propagated_collector, selectedPropHit]) if propagated_collector is not None else selectedPropHit)
        compatible_collector = (ak.concatenate([compatible_collector, best_matches]) if compatible_collector is not None else best_matches)

        logger.debug2(f"Number of good prophits: {total_prophit}")
        logger.debug2(f"Number of cartesian prophits (contains duplicates): {ak.sum(ak.num(compatibleHitsArray.mu_propagatedGlb_phi,axis=-1))}")
        logger.debug2(f"Number of matched prophits: {ak.sum(ak.num(accepted_hits.prop_etaID,axis=-1))}")
        logger.debug(f" matched collector {ak.sum(ak.num(matched_collector.prop_etaID,axis=-1))}")
        logger.debug(f" propagated_collector {ak.sum(ak.num(propagated_collector.prop_etaID,axis=-1))}")

        heap_size(the_heap, "before cleaning")
        del selectedPropHit
        del selectedRecHit
        del event
        heap_size(the_heap, "at the loop end")
        ## If the RAM goes full the process is terminated. Contain heap size by regurarly dumping the efficiency collectors
        if the_heap.heap().size / 2**20 > heap_dump_size:
            logger.warning(f"Heap size exceeds {heap_dump_size} MB, dumping collectors into temporary files")
            ExtendEfficiencyCSV(EfficiencySummary(matched_collector, propagated_collector),BASE_DIR / f"data/output/{output_name}.csv")
            matched_collector, propagated_collector = None, None
            heap_size(the_heap, "after dumping the efficiency collectors")
            if the_heap.heap().size / 2**20 > heap_dump_size:
                logger.error(f"After dumping the collectors heap size is still > {heap_dump_size}")

        if args.test == True and batch_index == 2: break

    if matched_collector is not None and propagated_collector is not None:
        logger.info(f"AVG Efficiency: {ak.sum(ak.num(matched_collector.prop_etaID,axis=-1))}/{ak.sum(ak.num(propagated_collector.prop_etaID,axis=-1))} = {ak.sum(ak.count(matched_collector.prop_etaID,axis=-1))/ak.sum(ak.num(propagated_collector.prop_etaID,axis=-1))}")
    cutSummary_collector = {k: v for k, v in sorted(cutSummary_collector.items(), key=lambda item: item[1], reverse=True)}
    logger.info(f"")
    logger.info(f"Breakdown table with cuts")
    logger.info(f"{'Label':<20}\t{'Survived Hits':>15}\t{'% Survived':>20}")
    for k in cutSummary_collector:
        logger.info(f"{k:<20}\t{cutSummary_collector[k]:>15}\t{round(cutSummary_collector[k]*100/cutSummary_collector['no_Mask'],2):>19}%")
    print()
    return matched_collector, propagated_collector, compatible_collector, selected_prop_perEvent, selected_rech_perEvent

if __name__ == "__main__":
    matched, prop, compatible_collector, selected_prop_perEvent, selected_rech_perEvent = main()



    start = time.time()
    df = EfficiencySummary(matched, prop)
    logger.info(f"Summary generated in {time.time()-start:.3f} s")
    ExtendEfficiencyCSV(df, BASE_DIR / f"data/output/{output_name}.csv")
    configuration.dump_config(BASE_DIR / f"data/output/{output_name}.yml")
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].hist(selected_prop_perEvent,10,range=(0,10),label="PropHit per evt")
    axs[1].hist(selected_rech_perEvent,100,range=(0,100),label="RecHit per evt")
    axs[0].legend()
    axs[1].legend()
    fig.savefig(f"{residual_output_folder_path}/HitsPerEvent.png")
    
    if args.residuals:
        #SAVE CSV WITH RESIDUALS
        mydf = ak.to_dataframe(compatible_collector)
        mydf.to_csv(f"./test.csv")
        tmp_compatible_collector = compatible_collector
        start = time.time()
        residual_hist, bin_edges = Fill_Histo_Residuals(tmp_compatible_collector, np.array([-0.18, 0.18]), 300)
        logger.info(f"Residuals binned in {time.time()-start:.3f} s")
        start = time.time()
        Store_Binned_Residuals(residual_hist, bin_edges, residual_output_folder_path,enable_plot=True)
        logger.info(f"Residuals PosiMuons plotted in {time.time()-start:.3f} s")
    logger.info(f"Timestamp {analysis_timestamp}")
    fig, axs = plt.subplots(3, 8, figsize=(60, 20))
    for eta in range(1,9):
        axs[0][eta-1].hist(ak.flatten(tmp_compatible_collector[tmp_compatible_collector["mu_propagated_etaP"]==eta].residual_phi),bins=200,range=(-40,40),label=f"Eta{eta} Residuals Phi")
        axs[1][eta-1].hist(ak.flatten(tmp_compatible_collector[tmp_compatible_collector["mu_propagated_etaP"]==eta].residual_rdphi),bins=200,range=(-40,40),label=f"Eta{eta} Residuals R /\Phi")
        axs[2][eta-1].hist(ak.flatten(tmp_compatible_collector[tmp_compatible_collector["mu_propagated_etaP"]==eta].mu_propagatedLoc_x),bins=200,range=(-20,20),label=f"Eta{eta} Propagated Loc x",alpha=0.5)
        axs[2][eta-1].hist(ak.flatten(tmp_compatible_collector[tmp_compatible_collector["mu_propagated_etaP"]==eta].gemRecHit_loc_x),bins=200,range=(-20,20),label=f"Eta{eta} rechit loc x",alpha=0.5)
        axs[0][eta-1].legend()
        axs[1][eta-1].legend()
        axs[2][eta-1].legend()
    fig.tight_layout()
    fig.savefig(f"{residual_output_folder_path}/AllResiduals.png")
    fig.savefig(f"{residual_output_folder_path}/AllResiduals.pdf")
    

