import awkward as ak
import numpy as np
import json

cut_labels = ["maxPropPhi_Err", "maxPropR_Err", "minpT", "fiducialR", "fiducialPhi", "RangeChi2", "minME1", "minME2", "minME3", "minME4"]
mask_labels = ["no_Mask","overallGood_Mask","isME11_Mask","isGE11_Mask"] + [ k+"_Mask" for k in cut_labels] + ["DAQMaskedVFAT","DAQError","DAQenabledOH","HVMask"]


def chamberName2_SRCL(chamberName):
    ## Accepts as input either 
        # GE11-M-03L1 or GE11-M-03L1-S
    if len(chamberName)==13:
        chamberName = chamberName[:11]
    st = 2 if "GE21" in chamberName else 1
    re = -1 if "M" in chamberName else 1
    ch = int( chamberName.split("-")[-1][:2] )
    la = int( chamberName.split("-")[-1][-1] )

    return (st,re,ch,la)

## Returns a dict containing the number of prophits that survived each cut and the parameters used 
def countNumberOfPropHits(dict_of_masks):
    output_dict = {}
    for k in mask_labels: output_dict[k] = ak.sum(dict_of_masks[k])
    return output_dict

## Retruns a dict containing the masks and cuts used
def calcMuonHit_masks(gemprophit_array, etaID_boundaries_array, maxPropPhi_Err=99999,maxPropR_Err=99999,minpT=0,fiducialR=0,fiducialPhi=0,RangeChi2=[0,99999999],minME1=0,minME2=0,minME3=0,minME4=0):
    
    dict_of_masks = {}
    dict_of_masks["maxPropPhi_Err"] = maxPropPhi_Err
    dict_of_masks["maxPropR_Err"] = maxPropR_Err
    dict_of_masks["minpT"] = minpT
    dict_of_masks["fiducialR"] = fiducialR
    dict_of_masks["fiducialPhi"] = fiducialPhi
    dict_of_masks["RangeChi2"] = RangeChi2
    dict_of_masks["minME1"] = minME1
    dict_of_masks["minME2"] = minME2
    dict_of_masks["minME3"] = minME3
    dict_of_masks["minME4"] = minME4
    
    dict_of_masks["no_Mask"] = ak.full_like(gemprophit_array.prop_etaID,True)
    dict_of_masks["isME11_Mask"] =  gemprophit_array.mu_propagated_isME11 == True
    dict_of_masks["isGE11_Mask"] =  gemprophit_array.mu_propagated_station == 1
    dict_of_masks["maxPropPhi_Err_Mask"] =  gemprophit_array.mu_propagatedGlb_errPhi < maxPropPhi_Err
    dict_of_masks["maxPropR_Err_Mask"] =  gemprophit_array.mu_propagatedGlb_errR < maxPropR_Err
    dict_of_masks["RangeChi2_Mask"] =  (gemprophit_array.mu_propagated_TrackNormChi2 <= RangeChi2[1]) & (gemprophit_array.mu_propagated_TrackNormChi2 >= RangeChi2[0])
    dict_of_masks["minpT_Mask"] =  gemprophit_array.mu_propagated_pt >= minpT
    dict_of_masks["minME1_Mask"] =  gemprophit_array.mu_propagated_nME1hits >= minME1
    dict_of_masks["minME2_Mask"] =  gemprophit_array.mu_propagated_nME2hits >= minME2
    dict_of_masks["minME3_Mask"] =  gemprophit_array.mu_propagated_nME3hits >= minME3
    dict_of_masks["minME4_Mask"] =  gemprophit_array.mu_propagated_nME4hits >= minME4
    dict_of_masks["fiducialPhi_Mask"] = np.logical_and(  gemprophit_array.mu_propagatedGlb_phi >= (etaID_boundaries_array[...,1] + fiducialPhi) ,  gemprophit_array.mu_propagatedGlb_phi <= (etaID_boundaries_array[...,0] - fiducialPhi))
    dict_of_masks["fiducialR_Mask"] = np.logical_and(  gemprophit_array.mu_propagatedGlb_r >= (etaID_boundaries_array[...,3] + fiducialR),  gemprophit_array.mu_propagatedGlb_r <= (etaID_boundaries_array[...,2] - fiducialR) )
    
    overallGood_Mask = dict_of_masks["isME11_Mask"] & dict_of_masks["isGE11_Mask"] & dict_of_masks["maxPropR_Err_Mask"] & dict_of_masks["maxPropPhi_Err_Mask"] & dict_of_masks["RangeChi2_Mask"] & dict_of_masks["minpT_Mask"] & dict_of_masks["minME1_Mask"] & dict_of_masks["minME2_Mask"] & dict_of_masks["minME3_Mask"] & dict_of_masks["minME4_Mask"] & dict_of_masks["fiducialPhi_Mask"] & dict_of_masks["fiducialR_Mask"]
    dict_of_masks["overallGood_Mask"] = overallGood_Mask

    return dict_of_masks

    ## returns a filtering mask that excludes the prophits for which the associated VFAT was masked by the DAQ during the evt
def calcDAQMaskedVFAT_mask(gemPropHit,gemOHStatus):
    """
    #Goal 
    Ignore the propagated hits associated with a VFAT that was masked by the DAQ

    The OHStatus array contains, for each evt, info for all chambers included in the run. Therefore it has a different shape than the PropHit  array. Additionally the VFAT mask is stored in a bitword.
    This is the fastest approach I could come up with

    STEP1: build arrays of cartesian product between gemPropHit and gemOHStatus to match the relevant quantities (station,region,chamber,layer). 
    Here it is important to use gemPropHit as first item of the product and use the option nested=True. 
    In such way the output will be something like

    array[event_index][<prop or oh label>][ array n_rows x m_column] 

    n_rows == number of propagated hits in the event
    m_column == number of gemOHStatus chambers

    STEP2: check whether the prophit belongs to a DAQ masked VFAT
    filter based on condition
    (
    OHstation == PropHitstation  & 
    conditionOHregion == PropHitregion & 
    conditionOHchamber == PropHitchamber & 
    conditionOHLayer == PropHitLayer & 
    PropHitVFAT was masked
    )
    VFAT is masked if the bitword VFAT Mask has the nth bit == 0, where n is the propHit VFAT number

    STEP3: evaluate a mask for the PropHit to be discarded
    For each PropHit in one event, there will be >100 combinations with GEMOHStatus in the cartesian product
    Only one combo might (or might not) be a match (i.e. prophit belongs to one chamber only)
    The ak.any checks whether at least 1 element of the array was true
    If there is a match the prophit has to be thrown away. 
    """
    ## STEP1
    x_station = ak.cartesian({"Prop":gemPropHit.mu_propagated_station,"OH":gemOHStatus.gemOHStatus_station},nested=True)
    x_region = ak.cartesian({"Prop":gemPropHit.mu_propagated_region,"OH":gemOHStatus.gemOHStatus_region},nested=True)
    x_chamber = ak.cartesian({"Prop":gemPropHit.mu_propagated_chamber,"OH":gemOHStatus.gemOHStatus_chamber},nested=True)
    x_layer = ak.cartesian({"Prop":gemPropHit.mu_propagated_layer,"OH":gemOHStatus.gemOHStatus_layer},nested=True)
    x_vfat = ak.values_astype(ak.cartesian({"Prop":gemPropHit.mu_propagated_VFAT,"OH":gemOHStatus.gemOHStatus_VFATMasked},nested=True),np.uint32) ## forcing uint32 to avoid issues with the following 2**x_vfat["prop"]
    ## STEP2 & STEP3
    PropHit_Ignored_mask = ak.any((x_station["OH"] == x_station["Prop"]) & (x_region["OH"] == x_region["Prop"]) & (x_chamber["OH"] == x_chamber["Prop"])  & (x_layer["OH"] == x_layer["Prop"])& (ak.values_astype((x_vfat["OH"] / 2**(x_vfat["Prop"])) %2,np.uint32) == 0),axis=-1) 

    return np.logical_not(PropHit_Ignored_mask)

## returns a filtering mask that excludes the prophits belonging to a OH in error in the evt
def calcDAQError_mask(gemPropHit,gemOHStatus):
    """
    same as in calcDAQMaskedVFAT_mask but now excluding prophits if associated to OH in error
    """
    gemOHStatus = gemOHStatus[gemOHStatus.gemOHStatus_errors >0]
    ## STEP1
    x_station = ak.cartesian({"Prop":gemPropHit.mu_propagated_station,"OH":gemOHStatus.gemOHStatus_station},nested=True)
    x_region = ak.cartesian({"Prop":gemPropHit.mu_propagated_region,"OH":gemOHStatus.gemOHStatus_region},nested=True)
    x_chamber = ak.cartesian({"Prop":gemPropHit.mu_propagated_chamber,"OH":gemOHStatus.gemOHStatus_chamber},nested=True)
    x_layer = ak.cartesian({"Prop":gemPropHit.mu_propagated_layer,"OH":gemOHStatus.gemOHStatus_layer},nested=True)
    ## STEP2 & STEP3
    PropHit_Ignored_mask = ak.any((x_station["OH"] == x_station["Prop"]) & (x_region["OH"] == x_region["Prop"]) & (x_chamber["OH"] == x_chamber["Prop"])  & (x_layer["OH"] == x_layer["Prop"]),axis=-1) 

    return np.logical_not(PropHit_Ignored_mask)    
    
def calcDAQenabledOH_mask(gemPropHit,gemOHStatus):
    """
    OH without GEMOHStatus are considered not enabled
    """
    x_station = ak.cartesian({"Prop":gemPropHit.mu_propagated_station,"OH":gemOHStatus.gemOHStatus_station},nested=True)
    x_region = ak.cartesian({"Prop":gemPropHit.mu_propagated_region,"OH":gemOHStatus.gemOHStatus_region},nested=True)
    x_chamber = ak.cartesian({"Prop":gemPropHit.mu_propagated_chamber,"OH":gemOHStatus.gemOHStatus_chamber},nested=True)
    x_layer = ak.cartesian({"Prop":gemPropHit.mu_propagated_layer,"OH":gemOHStatus.gemOHStatus_layer},nested=True)

    propHitOH_has_foundin_gemOHStatus = (x_station["OH"] == x_station["Prop"]) & (x_region["OH"] == x_region["Prop"]) & (x_chamber["OH"] == x_chamber["Prop"])  & (x_layer["OH"] == x_layer["Prop"])
    mask = ak.any(propHitOH_has_foundin_gemOHStatus,axis=-1)
    return mask


def import_HVmask(file_path):
    io = open(file_path,"r")
    dictionary = json.load(io)  
    for key,item in list(dictionary.items()):
        if item == []:
            dictionary.pop(key)
    return dictionary

## i.e. [1,2,3,4,5,6,9,10,11] --> [[1,6],[9,11]]
## to speed up the masking step
def group_consecutive_elements_in_ranges(array):
    # Sort the array in ascending order
    array = np.sort(array)
    # Initialize variables to keep track of current range
    current_range = np.empty(2,dtype=np.int16)
    grouped_elements = np.empty((0,2),dtype=np.int16)

    for idx,element in enumerate(array):

        if idx==0: 
            current_range[0] = element
            current_range[1] = element
        else:
            # Check if the current element is consecutive to the last element in current_range
            if element == current_range[1] + 1: current_range[1] = element
            # If not consecutive, add the current range to the grouped_elements list
            else:
                grouped_elements = np.append(grouped_elements,np.asarray([current_range],dtype=np.int16),axis=0)
                # Start a new range with the current element
                current_range[0] = element
                current_range[1] = element

    # Add the last range to the grouped_elements list
    if len(array) != 0:
        grouped_elements = np.append(grouped_elements,np.asarray([current_range],dtype=np.int16),axis=0)

    return grouped_elements

##TODO make it more efficient
## returns a filtering mask that excludes the prophits in a given LS if it was a LS with bad HV
def calcHV_mask(gemPropHit,HV_filepath):
    HVMask_byChamber = import_HVmask(HV_filepath)
    outputMask = ak.broadcast_arrays(False,gemPropHit.mu_propagated_isGEM)[0]

    for chamber in HVMask_byChamber:
        st,re,ch,la = chamberName2_SRCL(chamber)
        ## aggreagate lumisection in ranges
        BadLumis = group_consecutive_elements_in_ranges(np.asarray(HVMask_byChamber[chamber],dtype=np.int16))
        select_chamber = (gemPropHit.mu_propagated_station == st) & (gemPropHit.mu_propagated_region == re) & (gemPropHit.mu_propagated_chamber == ch) & (gemPropHit.mu_propagated_layer == la) & (gemPropHit.mu_propagated_layer == la)
        if -1 in np.unique(BadLumis.flatten()): outputMask = outputMask | select_chamber
        else:
            for _range in BadLumis:
                strt = _range[0]
                stp = _range[1]
                PropHit_Ignored_mask = select_chamber &  ((gemPropHit.mu_propagated_lumiblock >= strt) & (gemPropHit.mu_propagated_lumiblock <= stp))
                outputMask = outputMask | PropHit_Ignored_mask
    return np.logical_not(outputMask)