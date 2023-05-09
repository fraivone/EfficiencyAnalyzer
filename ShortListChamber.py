import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from Utils import BASE_DIR
from Statistics import ItemEfficiency
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
outputname =  f'Discrepancy_{time.strftime("%-y%m%d%H%M")}.txt'
        


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


    ## loading data in a dict of objects
    for input_idx, df in enumerate([ pd.read_csv(file_path, sep=",") for file_path in args.inputs]):
        df = df[(df["propHit"] != 0)]
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
output_string = ""
for k in chambersCollector:
        eff_obj = chambersCollector[k]
        eff_obj.checkConsistency()
        output_string += eff_obj.GetInconsistencySummary()
with open(output_folder_path / outputname, 'w') as textfile:
    textfile.write(output_string)



    # data  = []
    # filemap = None
    # for k in chambersCollector:
    #     eff_obj = chambersCollector[k]
    #     eff_obj.checkConsistency()

    #     consistency_result = chambersCollector[k].getConsistencyResults()
    #     eff_is_consistent =  eff_obj.isConsistent()
    #     if eff_is_consistent == False:
    #         data.append(consistency_result)
    #         consistency_result['ChamberName'] = k
    #         if filemap is not None:
    #             if len(filemap) <  len(chambersCollector[k].getSourceIndex()):
    #                 filemap = chambersCollector[k].getSourceIndex()                            
    #         else:
    #             filemap = chambersCollector[k].getSourceIndex()
    # print(data)
    # df = pd.DataFrame.from_records(data)
    # cols = list(df.columns.values)
    # cols.remove("ChamberName")
    # df =  df[["ChamberName"] + cols]
    # df = df.sort_values('ChamberName')

    # df.to_csv(output_folder_path / f"ConsistencyMap_{timetag}.csv", index=False)

    # df = pd.DataFrame(filemap.items())
    # df.to_csv(output_folder_path / f"IndexesMap_{timetag}.csv", index=False,header=False)

                
