import awkward as ak
from os import environ
#import math as math
import numpy as np
import numba
import pandas as pd
from myLogger import logger as default_logger
from pathlib import Path
from typing import List
from Statistics import generateClopperPearsonInterval




BASE_DIR = Path(__file__).parent.parent
OUTPUT_PATH = BASE_DIR/"data/output/" if environ.get("OUTPUT_PATH","") == "" else Path(environ.get("OUTPUT_PATH",""))
PHPINDEX_FILE = None if environ.get("INDEXPHP","") == "" else Path(environ.get("INDEXPHP",""))

logger = default_logger.getLogger(__name__)



def heap_size(the_hp, label: str) -> None:
    thp = the_hp.heap()
    logger.memory(f"Heap Size {label}: {thp.size/2**20:.3f} MB")  # type: ignore
    # print(thp.byvia[0:3])


## given iEta,iPhi having the same structure, returns VFAT number with consistent structure (i.e. int,int --> int  ; array,array --> array)
def iEtaiPhi_2_VFAT(iEta, iPhi):
    ##TODO define procedure in case inputs are ints (values_astype would raise an error)
    VFAT = iPhi * 8 + (8 - iEta)
    return ak.values_astype(VFAT, np.short)

def iEta_2_chamberType(iEta, layer, station):
    #Using layer per GE11, Module per GE21
#    print ("station ", station)

    chType_ge11 = ak.values_astype(layer, np.short)  
    chType_ge21 = ak.values_astype(np.ceil((17-iEta)/4), np.short)  

    return  ak.where(station<2, chType_ge11,  chType_ge21) 


def iEtaStrip_2_VFAT(iEta, strip, station):
    # implementing with DPG eta convention: not accurate for the DAQ convetion. 
#    if station==1:
#        vfat_ge11 = (strip // 128) * 8 + (8 - iEta)   
#        return ak.values_astype(VFAT, np.short)  
#    if station==2:
#        vfat_ge21 = (strip // 64) * 2 +  (4 * np.ceil(iEta / 4) - iEta) // 2  
# return ak.values_astype(VFAT, np.short)
    vfat_ge11 = ak.values_astype( (strip // 128) * 8 + (8 - iEta),  np.short)
    vfat_ge21 = ak.values_astype((strip // 64) * 2 +  (4 * np.ceil(iEta / 4) - iEta) // 2 ,  np.short)      
    return ak.where(station<2, vfat_ge11,  vfat_ge21) # considering just Ge11 and GE21 atm.     

## given iEta,firstStrip, CLS having the same structure, returns VFAT number with consistent structure (i.e. int,int,int --> int  ; array,array,array --> array)
def recHit2VFAT(etaP, firstStrip, CLS, station):
    ##TODO define procedure in case inputs are ints (values_astype would raise an error)
    centerStrip = ak.values_astype(firstStrip + CLS // 2 , np.short)
    return iEtaStrip_2_VFAT(etaP, centerStrip, station)


@numba.njit(cache=True)
def empty_collector() -> np.ndarray:
    return np.zeros((2, 2, 36, 2, 4, 24), dtype=np.uint16)  # st,re,ch,lay,type,vfat for GE11-GE21
    

## using option parallel had caused race condition. Dropped
@numba.njit(cache=True)
def aggregateHitsVFAT(array):
    collector = empty_collector()
    for evt_idx in range(len(array)):
        for hits_idx in range(len(array[evt_idx].mu_propagated_station)):
            st = array[evt_idx].mu_propagated_station[hits_idx]
            re = (array[evt_idx].mu_propagated_region[hits_idx] + 1) // 2  ## transforming -1 -> 0, 1 -> 1 to respect the indexing
            ch = array[evt_idx].mu_propagated_chamber[hits_idx]
            layer = array[evt_idx].mu_propagated_layer[hits_idx]
            chamberType = array[evt_idx].mu_propagated_chamberType[hits_idx]
            vfat = array[evt_idx].mu_propagated_VFAT[hits_idx]
            collector[(st - 1, re, ch - 1, layer - 1, chamberType-1, vfat)] += 1

    return collector


def npArray_2_dataframe(array, name, columns) -> pd.DataFrame:
    """
    Multidimensional numpy array to dataframe using its indexes as columns named after the input list 'columns'
    """
    index = pd.MultiIndex.from_product([range(s) for s in array.shape], names=columns)
    df = pd.DataFrame({name: array.flatten()}, index=index).reset_index()
    return df

def EfficiencySummary(matched, prop) -> pd.DataFrame:
    columns = ["Station","Region","Chamber","Layer", "ChamberType","VFAT"]  ## depends on function aggregateHitsVFAT
    
    if matched is not None and prop is not None:
        match_aggregate = aggregateHitsVFAT(matched)
        prop_aggregate = aggregateHitsVFAT(prop)
    else:
        match_aggregate, prop_aggregate = empty_collector(), empty_collector()
    match_df = npArray_2_dataframe(match_aggregate, "matchedRecHit", columns)
    prop_df = npArray_2_dataframe(prop_aggregate, "propHit", columns)

    df_merge = pd.merge(match_df, prop_df, how="outer", on=columns)
    #df_merge.to_csv("Debug.txt", sep=";", index=False)
    ## transforming back
    df_merge["Station"] += 1
    df_merge["Region"] = df_merge["Region"] * 2 - 1
    df_merge["Chamber"] += 1
    df_merge["ChamberType"] += 1
    df_merge["Layer"] += 1

    #Removing empty/zero duplicates for GE11 as chamberType is just a duplication of the layer and 
    #for GE21 entries with chambers that are different than 16 and 18 or vfat above number 12.   
    df_merge=df_merge.loc[((df_merge["Layer"]==df_merge["ChamberType"]) & (df_merge["Station"]==1) )
                          | ( (df_merge["Station"]==2) & (df_merge["VFAT"]<12) & ((df_merge["Chamber"]==16) | (df_merge["Chamber"]==18)))]
    
    return df_merge


def mergeEfficiencyCSV(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    s = pd.concat(df_list)
    s = s.groupby(["Station", "Region", "Chamber", "Layer", "ChamberType", "VFAT"], as_index=False).sum()
    return s


def ExtendEfficiencyCSV(new_df: pd.DataFrame, existing_csv_path: Path) -> None:
    """
    Extended the efficiency csv by merging the new_df
    if the csv doesn't exists, it creates one
    """
    if existing_csv_path.is_file():
        mergeEfficiencyCSV([new_df, pd.read_csv(existing_csv_path, sep=",")]).to_csv(existing_csv_path, index=False)
    else:
        new_df.to_csv(existing_csv_path, index=False)


if __name__ == "__main__":
    print(f"438V: 2196 discharges for 5 propagations\t{generateClopperPearsonInterval(5,2196)}")
    print(f"500V: 686 discharges for 0 propagations\t{generateClopperPearsonInterval(0,686)}")
    print(f"550V: 2132 discharges for 112 propagations\t{generateClopperPearsonInterval(112,2132)}")
    print(f"600V: 269 discharges for 6 propagations\t{generateClopperPearsonInterval(6,269)}")
    pass
