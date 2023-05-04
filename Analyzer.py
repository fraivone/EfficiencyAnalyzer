import awkward as ak
import uproot
import math
from os import listdir
from os.path import isfile, join, getsize, abspath
from pathlib import Path
import time
import numpy as np
import argparse
import guppy
from shutil import copy
import json
from Utils import (
    heap_size,
    iEtaiPhi_2_VFAT,
    recHit2VFAT,
    EfficiencySummary,
    EOS_OUTPUT_PATH,
    EOS_INDEX_FILE,
    BASE_DIR,
    ExtendEfficiencyCSV,
)
from MaskFunctions import (
    calcMuonHit_masks,
    countNumberOfPropHits,
    calcDAQMaskedVFAT_mask,
    calcDAQError_mask,
    calcDAQenabledOH_mask,
    calcHV_mask,
)
from AkArray_Operations import find_boundaries, boundaries_translation, best_match
from PlottingFunctions import Fill_Histo_Residuals, Plot_Binned_Residuals
from config_parser import config
from myLogger import logger as logger_default

logger = logger_default.getLogger(__name__)
logger.setLevel(logger_default.DEBUG)
the_heap = guppy.hpy()
the_heap.setref()

## PARSER
parser = argparse.ArgumentParser(description="Analyzer parser")
parser.add_argument("config", help="Analysis description file")
parser.add_argument("--folder_name", type=str, help="Output folder name", required=False, default="test")
parser.add_argument("--residuals", help="Enable plotting residuals", required=False, action="store_true", default=False)
args = parser.parse_args()

configuration = config(abspath(args.config))
input_par = configuration.parameters
matching_cuts = configuration.matching_window
match_by = "residual_rdphi"
output_folder_path = Path(EOS_OUTPUT_PATH, args.folder_name)
output_name = configuration.analysis_label + time.strftime("_%-y%m%d%H%M")
max_evts = configuration.data_input["max_evts"]
runRange = sorted(configuration.data_input["runRange"])
## PARSER
## CREATE FOLDERS and COPY FILES
output_folder_path.mkdir(parents=True, exist_ok=True)
copy(EOS_INDEX_FILE, output_folder_path)
output_folder_path = Path(output_folder_path, "Residuals")
output_folder_path.mkdir(parents=True, exist_ok=True)
copy(EOS_INDEX_FILE, output_folder_path)
##

avg_batch_size = 900  # MB
heap_dump_size = 1000  # MB above which efficiency summary gets calculated from propagated_collector and matched_collector and dumped to file


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
    total_events = 0

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
        atLeast1GE11Prop = ak.any(gemPropHit.mu_propagated_station == 1, axis=-1) & ak.any(gemPropHit.mu_propagated_station == 1, axis=-1)
        gemPropHit = gemPropHit[atLeast1GE11Prop]
        gemRecHit = gemRecHit[atLeast1GE11Prop]
        gemOHStatus = gemOHStatus[atLeast1GE11Prop]
        # Removing all GE21 prophits (no additional events removed)
        gemPropHit = gemPropHit[gemPropHit.mu_propagated_station == 1]
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

        ## all possible pairs of (propHit,recHit) for each event
        product = ak.cartesian({"prop": selectedPropHit.prop_etaID, "rec": selectedRecHit.rec_etaID})
        cartesian_indeces = ak.argcartesian({"prop": selectedPropHit.prop_etaID, "rec": selectedRecHit.rec_etaID})

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
        angle_translation_rec = (((compatibleHitsArray.mu_propagatedGlb_phi - compatibleHitsArray.gemRecHit_g_phi) > 5) * 2 * np.pi)
        angle_translation_prop = (((compatibleHitsArray.mu_propagatedGlb_phi - compatibleHitsArray.gemRecHit_g_phi) < -5) * 2 * np.pi)
        compatibleHitsArray["gemRecHit_g_phi"] = compatibleHitsArray["gemRecHit_g_phi"] + angle_translation_rec
        compatibleHitsArray["mu_propagatedGlb_phi"] = (compatibleHitsArray["mu_propagatedGlb_phi"] + angle_translation_prop)

        ## Residual Calc
        compatibleHitsArray["residual_phi"] = (compatibleHitsArray.mu_propagatedGlb_phi - compatibleHitsArray.gemRecHit_g_phi)
        compatibleHitsArray["residual_rdphi"] = (compatibleHitsArray.mu_propagatedGlb_phi - compatibleHitsArray.gemRecHit_g_phi) * compatibleHitsArray.mu_propagatedGlb_r
        heap_size(the_heap, "calculating residuals")

        logger.debug(f" Cut on residuals for efficiency")
        best_matches = best_match(compatibleHitsArray, match_by)
        accepted_hits = best_matches[best_matches[match_by] < matching_cuts[match_by]]
        accepted_hits = accepted_hits[ak.num(accepted_hits.prop_etaID, axis=-1) > 0]
        heap_size(the_heap, "after selection based on residuals")

        matched_collector = (ak.concatenate([matched_collector, accepted_hits]) if matched_collector is not None else accepted_hits)
        propagated_collector = (ak.concatenate([propagated_collector, selectedPropHit]) if propagated_collector is not None else selectedPropHit)
        if args.residuals:
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
            heap_size(the_heap, "Dumping the efficiency collectors")
            if the_heap.heap().size / 2**20 > heap_dump_size:
                logger.error(f"After dumping the collectors heap size is still > {heap_dump_size}")

    if matched_collector is not None and propagated_collector is not None:
        logger.info(f"AVG Efficiency: {ak.sum(ak.num(matched_collector.prop_etaID,axis=-1))}/{ak.sum(ak.num(propagated_collector.prop_etaID,axis=-1))} = {ak.sum(ak.count(matched_collector.prop_etaID,axis=-1))/ak.sum(ak.num(propagated_collector.prop_etaID,axis=-1))}")

    cutSummary_collector = {k: v for k, v in sorted(cutSummary_collector.items(), key=lambda item: item[1], reverse=True)}
    logger.info(f"")
    logger.info(f"Breakdown table with cuts")
    logger.info(f"{'Label':<20}\t{'Survived Hits':>15}\t{'% Survived':>20}")
    for k in cutSummary_collector:
        logger.info(f"{k:<20}\t{cutSummary_collector[k]:>15}\t{round(cutSummary_collector[k]*100/cutSummary_collector['no_Mask'],2):>19}%")
    print()
    return matched_collector, propagated_collector, compatible_collector

if __name__ == "__main__":
    matched, prop, compatible_collector = main()

    start = time.time()
    df = EfficiencySummary(matched, prop)
    logger.info(f"Summary generated in {time.time()-start:.3f} s")
    ExtendEfficiencyCSV(df, BASE_DIR / f"data/output/{output_name}.csv")
    configuration.dump_config(BASE_DIR / f"data/output/{output_name}.yml")

    if args.residuals:
        start = time.time()
        residual_hist, bin_edges = Fill_Histo_Residuals(compatible_collector, np.array([-2, 2]), 300)
        logger.info(f"Residuals binned in {time.time()-start:.3f} s")

        start = time.time()
        Plot_Binned_Residuals(residual_hist, bin_edges, output_folder_path)
        logger.info(f"Residuals plotted in {time.time()-start:.3f} s")

    # from matplotlib import pyplot as plt
    # fig,ax = plt.subplots(1,figsize=(20,20),layout="constrained")
    # max_lumiblock = ak.max(rechit_collector.gemRecHit_lumiblock)+1
    # min_lumiblock = ak.min(rechit_collector.gemRecHit_lumiblock)
    # endcapMask = (rechit_collector.gemRecHit_region == -1)  & (rechit_collector.gemRecHit_layer == 1) & (rechit_collector.gemRecHit_station == 1)
    # rechit_collector["hasHits"] = rechit_collector.gemRecHit_station != 0
    # plotArray_2D(rechit_collector, endcapMask, "gemRecHit_lumiblock", "hasHits", (min_lumiblock,max_lumiblock), (-0.5,1.5), np.arange(min_lumiblock,max_lumiblock, 50, dtype=float),  [0,1], f"GE11-M-15L1", "Lumiblock", "Chamber", plt.cm.Blues, ax)
    # fig.savefig("361512_GE11-M-15L1.png",dpi=200)
    # fig,ax = plt.subplots(1,figsize=(20,20),layout="constrained")
    # endcapMask = (rechit_collector.gemRecHit_region == 1)  & (rechit_collector.gemRecHit_layer == 2) & (rechit_collector.gemRecHit_station == 1)
    # rechit_collector["hasHits"] = rechit_collector.gemRecHit_station != 0
    # plotArray_2D(rechit_collector, endcapMask, "gemRecHit_lumiblock", "hasHits", (min_lumiblock,max_lumiblock), (-0.5,1.5), np.arange(min_lumiblock,max_lumiblock, 50, dtype=float),  [0,1], f"GE11-P-15L2", "Lumiblock", "Chamber", plt.cm.Blues, ax)
    # fig.savefig("361512_GE11-P-15L2.png",dpi=200)
