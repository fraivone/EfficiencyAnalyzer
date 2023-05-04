import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from Utils import BASE_DIR
from scipy.stats import norm
import math
import itertools
from deepdiff import DeepDiff
import time

## GE11 only
station = 1
##### Parser
parser = argparse.ArgumentParser(description="Short list chambers that have non consistent effficiency in the given set of runs")
parser.add_argument("inputs", help="Input csv(s)", nargs="*")
args = parser.parse_args()

if len(args.inputs) < 2:
    parser.error('At least two input files are required')


output_folder_path = Path(BASE_DIR, "data/discrepancies/")
output_folder_path.mkdir(parents=True, exist_ok=True)
output_name =  f'{time.strftime("_%-y%m%d%H%M")}'

def stdError(avg1,d1,avg2,d2):
    stdError = math.sqrt( avg1*(1-avg1)/d1 + avg2*(1-avg2)/d2 )
    return stdError

# returns None when both n1 and n2 are 0
def zStat(n1,d1,n2,d2):
    avg1 = float(n1) / d1
    avg2 = float(n2) / d2

    err = stdError(avg1,d1,avg2,d2)
    if err != 0:
        return ( avg1 - avg2 )  / err
    else:
        return None

## Returns None when z_stat is None
def pValue(n1,d1,n2,d2):
    z_stat = zStat(n1,d1,n2,d2)
    if z_stat is not None:
        z_stat = abs(z_stat)
        return 1 - (norm.cdf(z_stat) - norm.cdf(-z_stat))
    else:
        return None


class ItemEfficiency:
    def __init__(self,name):
        self.name = name
        self.format = np.dtype([('source', 'U100'), ('numerator', 'i4'), ('denominator', 'i4'),('efficiency', 'f4')])
        self.data = np.array([],dtype=self.format)
        self.verbose = False
        self.__consistent = True
        self.consistencyMap = {}
        self.sourceIndex = {}

    def AddData(self,source,Num,Den):
        if source in self.data['source']:
            print(f"Data is present for source {source} item name {self.name}.\nData points not added")
        elif Den == 0:
            print(f"Parsed denominator is 0.\nData points not added")
        elif Num > Den:
            print(f"Parsed numerator {Num} > denominator {Den}.\nData points not added")
        else:
            self.data = np.append(self.data,np.array([(source,Num,Den,float(Num)/Den)],dtype=self.format))
    def ClearData(self):
        self.data = np.array([],dtype=self.format)

    def __sortData(self):
        idx = np.argsort(self.data, order=['source'])
        sorted_data = self.data[idx]
        for k in range(len(idx)):
            self.sourceIndex[k] = sorted_data['source'][k]
        return sorted_data

    def isConsistent(self):
        return self.__consistent

    def getConsistencyMap(self):
        return self.consistencyMap

    def getSourceIndex(self):
        return self.sourceIndex

    def checkConsistency(self):
        sortedData = self.__sortData()
        data_size = len(sortedData)
        ## all possible pairs 
        for idx1,idx2 in itertools.combinations(list(range(0,data_size)),2):
            p = pValue(sortedData['numerator'][idx1],
                       sortedData['denominator'][idx1],
                       sortedData['numerator'][idx2],
                       sortedData['denominator'][idx2])
            if p is None:
                continue
            ## If the pvalue < 0.001
            if p < 0.001 : 
                consistent = False
                self.__consistent = False
            else: consistent = True

            self.consistencyMap[(idx1,idx2)] = consistent

            if self.verbose:
                print(f"Eff[{sortedData['source'][idx1]}]:{sortedData['efficiency'][idx1]}")
                print(f"Eff[{sortedData['source'][idx2]}]:{sortedData['efficiency'][idx2]}")
                print(f"pvalue: {p}")
                if consistent: print(f"\033[1;92mConsistent\033[1;0m")
                else: print(f"\033[1;31mInconsistent\033[1;0m")
                print()
        


if __name__ == '__main__':
    chambersCollector = {}
    aggregation_functions = {
                "Station": "first",
                "Region": "first",
                "Layer": "first",
                "Chamber": "first",
                "chamberName": "first",
                "matchedRecHit": "sum",
                "propHit": "sum",
            }

    for input_idx, df in enumerate([ pd.read_csv(file_path, sep=",") for file_path in args.inputs]):
        df = df[df["propHit"] != 0]
        df = df[(df.Station == station)]
        df["chamberName"] = df.apply(lambda x: f"GE{x['Station']}1-{'M'if x['Region']==-1 else 'P'}-{x['Chamber']:02d}L{x['Layer']}",axis=1)
        df = df.groupby(df["chamberName"]).aggregate(aggregation_functions)
        for index, row in df.iterrows():
            chID = row["chamberName"]
            if chID in chambersCollector: s = chambersCollector[chID]
            else: s = ItemEfficiency(chID)
            matchedRecHit = row["matchedRecHit"]
            propHit = row["propHit"]
            s.AddData(args.inputs[input_idx],matchedRecHit,propHit)
            chambersCollector[chID] = s

    data  = []
    filemap = None
    for k in chambersCollector:
        chambersCollector[k].checkConsistency()
        temp_dict = chambersCollector[k].getConsistencyMap()
        if chambersCollector[k].isConsistent() == False:
            data.append(temp_dict)
            temp_dict['ChamberName'] = k
        
        if filemap is not None:
            if len(DeepDiff(filemap,chambersCollector[k].getSourceIndex())) == 0: pass
            else:
                if len(filemap) <  len(chambersCollector[k].getSourceIndex()):
                    filemap = chambersCollector[k].getSourceIndex()        
        else:
            filemap = chambersCollector[k].getSourceIndex()
    
    
    df = pd.DataFrame.from_records(data)
    cols = list(df.columns.values)
    cols.remove("ChamberName")
    df =  df[["ChamberName"] + cols]
    df = df.sort_values('ChamberName')

    df.to_csv(output_folder_path / f"ConsistencyMap_{output_name}.csv", index=False)

    df = pd.DataFrame(filemap.items())
    df.to_csv(output_folder_path / f"IndexesMap_{output_name}.csv", index=False,header=False)

                
