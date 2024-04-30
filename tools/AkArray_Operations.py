import awkward as ak
import numpy as np
from myLogger import logger as default_logger
from EtaPartitionBoundaries_2024 import *

logger = default_logger.getLogger(__name__)
logger.setLevel(default_logger.INFO)
etaP_boundaries = {**boundariesGE11, **boundariesGE21}

#### 
## Recursive function that takes the cartesian combination
## array of prophit and rechits with the same etaID 
## and returns a new array containing only the matches
## having the lowest "minimize_label", where minimized label
## must be a key in the ak.record passed as input. 
## The output array has one row per best match, 
## thus it does NOT maintain the shape of the original array 
## which can possibly have more than one match per event
def best_match(original,minimize_label="residual_phi",_output=None,_step=0):
    uff = original[ak.num(original.prop_etaID,axis=-1)>0]
    if _step == 0: logger.debug2(f"Selecting pairs based on the best {minimize_label}")
    
    same_etaID = uff.prop_etaID[...,0] == uff.prop_etaID[...,]
    same_pt = uff.mu_propagated_pt[...,0] == uff.mu_propagated_pt[...,]
    idx_to_keep = np.logical_and(same_etaID,same_pt)
    idx_to_reprocess = np.logical_not(idx_to_keep)
    firstEtaID = uff[idx_to_keep]
    remainingEtaIDs = uff[idx_to_reprocess]
    
    _output = uff[uff.mu_propagated_station < 0][0:1] if _output is None else _output
    
    n_processed_matches = ak.sum(ak.num(firstEtaID.prop_etaID,axis=-1))
    n_total_matches = ak.sum(ak.num(uff.prop_etaID,axis=-1))
    n_reamining_matches = ak.sum(ak.num(remainingEtaIDs.prop_etaID,axis=-1))
    
    best_Match = firstEtaID[ak.argsort(abs(firstEtaID[minimize_label]))][...,0:1]
    n_best_matches = ak.sum(ak.num(best_Match.prop_etaID,axis=-1))
    _output = ak.concatenate([_output,best_Match])
    
    logger.debug2(f"Step{_step}:")
    logger.debug2(f"\t{'Processed pairs':<44}{n_processed_matches:>10}/{n_total_matches}")
    logger.debug2(f"\t{'Best pairs':<44}{n_best_matches:>10}/{n_processed_matches}")
    logger.debug2(f"\t{'etaID multiplicity':<44}{round(n_processed_matches/n_best_matches,2):>10}")
    logger.debug2(f"\t{'Pairs passed to the next step':<44}{n_reamining_matches:>10}/{n_total_matches}")
    
    if n_reamining_matches == 0: 
        return _output
    
    else:
        _step += 1
        return best_match(remainingEtaIDs,minimize_label,_output,_step)

# if passed arg is int, assume it is etaPID and return boundaries
# if ak.array, repeat recursively 
# etaP_boundaries[etaID] = [phiMax, phiMin, rMax, rMin]
def find_boundaries(a,verbose=False):
    if verbose: logger.debug2(f"Passed var: {a}, type: {type(a)}")
    if isinstance(a,np.int32):
        if verbose: logger.debug2(f"Since {a} is a int, return the boundaries:\t{etaP_boundaries[a]}")
        return etaP_boundaries[a]
    elif isinstance(a,ak.Array):
        if verbose: logger.debug2(f"Since {a} is an awkward Array, map its elements to a etaP boundary")
        return map(find_boundaries,a)
    else:
        raise ValueError(f"Something went wrong with the selection type {type(a)}")

## function to ease boundary phi translation form [-pi,pi] to [0,2pi]
magic_array_True =  ak.Array([2*np.pi,0,0,0])
magic_array_False =  ak.Array([0,0,0,0])
def boundaries_translation(input_value,verbose = False):
    if verbose: logger.debug2(f"Passed var: {input_value}, type: {type(input_value)}")
    if isinstance(input_value,np.bool_):
        if verbose: logger.debug2(f"Since {input_value} is a bool, returning array")
        if input_value: return magic_array_True
        else: return magic_array_False
    
    elif isinstance(input_value,ak.highlevel.Array):
        if verbose: logger.debug2(f"Since {input_value} is an awkward Array, map its elements to array")
        return map(boundaries_translation,input_value)
    
    else:
        raise ValueError("Something went wrong with the selection type")
