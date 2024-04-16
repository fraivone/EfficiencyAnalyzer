import awkward as ak
import numpy as np
import numba
import pandas as pd
from myLogger import logger as default_logger
from pathlib import Path
from typing import List
from Statistics import generateClopperPearsonInterval

EOS_OUTPUT_PATH = Path("/eos/user/c/cgalloni/www/P5_Operations_test/")
BASE_DIR = Path(__file__).parent.parent
EOS_INDEX_FILE = Path("/eos/user/c/cgalloni/www/Plots/index.php")


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



def iEtaStrip_2_VFAT(iEta, strip, station):
     # implementing with DPG convention: not accurate for the DAQ convetion. 
    if station==1:
         VFAT = (strip // 128) * 8 + (8 - iEta)   
         return ak.values_astype(VFAT, np.short)  
    if station==2:
    # return: puppa visto che non e' nel file:
          VFAT = (strip // 64) * 2 +  (4 * math.ceil(iEta / 4) - iEta) // 2  
          return ak.values_astype(VFAT, np.short)     

## given iEta,firstStrip, CLS having the same structure, returns VFAT number with consistent structure (i.e. int,int,int --> int  ; array,array,array --> array)
def recHit2VFAT(etaP, firstStrip, CLS, station):
    ##TODO define procedure in case inputs are ints (values_astype would raise an error)
    centerStrip = ak.values_astype(firstStrip + CLS // 2 , np.short)
    return iEtaStrip_2_VFAT(etaP, centerStrip, station)


@numba.njit(cache=True)
def empty_collector() -> np.ndarray:
    return np.zeros((1, 2, 36, 2, 24), dtype=np.uint16)  # st,re,ch,la,vfat


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
            vfat = array[evt_idx].mu_propagated_VFAT[hits_idx]

            collector[(st - 1, re, ch - 1, layer - 1, vfat)] += 1

    return collector


def npArray_2_dataframe(array, name, columns) -> pd.DataFrame:
    """
    Multidimensional numpy array to dataframe using its indexes as columns named after the input list 'columns'
    """
    index = pd.MultiIndex.from_product([range(s) for s in array.shape], names=columns)
    df = pd.DataFrame({name: array.flatten()}, index=index).reset_index()
    return df

def EfficiencySummary(matched, prop) -> pd.DataFrame:
    columns = ["Station","Region","Chamber","Layer","VFAT"]  ## depends on function aggregateHitsVFAT
    if matched is not None and prop is not None:
        match_aggregate = aggregateHitsVFAT(matched)
        prop_aggregate = aggregateHitsVFAT(prop)
    else:
        match_aggregate, prop_aggregate = empty_collector(), empty_collector()
    match_df = npArray_2_dataframe(match_aggregate, "matchedRecHit", columns)
    prop_df = npArray_2_dataframe(prop_aggregate, "propHit", columns)

    df_merge = pd.merge(match_df, prop_df, how="outer", on=columns)
    ## transforming back
    df_merge["Station"] += 1
    df_merge["Region"] = df_merge["Region"] * 2 - 1
    df_merge["Chamber"] += 1
    df_merge["Layer"] += 1

    return df_merge


def mergeEfficiencyCSV(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    s = pd.concat(df_list)
    s = s.groupby(["Station", "Region", "Chamber", "Layer", "VFAT"], as_index=False).sum()
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
