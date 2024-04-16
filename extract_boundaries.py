import awkward as ak
import uproot
import numpy as np
import sys

"""
Given an ntuple as input, it finds the eta partition boundaries and
prints them out on the screen following the naming convention in
https://github.com/fraivone/EfficiencyAnalyzer

usage 
python3 extract_boundaries.py <ntuple_path>
"""


branches_prophit = ["mu_propagated_station", "mu_propagated_region","mu_propagated_chamber","mu_propagated_layer","mu_propagated_etaP","mu_propagated_EtaPartition_centerX","mu_propagated_EtaPartition_centerY","mu_propagated_EtaPartition_rMax","mu_propagated_EtaPartition_rMin","mu_propagated_EtaPartition_phiMax","mu_propagated_EtaPartition_phiMin"]
upfile = uproot.open(sys.argv[1])
tree = upfile["muNtupleProducer/MuDPGTree;11"]
gemPropHit = tree.arrays(filter_name=branches_prophit)
gemPropHit["etaID"] =  (2**17 - 1) & gemPropHit[:].mu_propagated_chamber << 11 | (gemPropHit[:].mu_propagated_etaP << 6) | (gemPropHit[:].mu_propagated_station << 4) | (gemPropHit[:].mu_propagated_layer << 1) | (abs(gemPropHit[:].mu_propagated_region - 1) // 2)

outs = []

for station in [1,2]:
    stationProphit = gemPropHit[gemPropHit["mu_propagated_station"] == station]
    outs.append(f"Found {np.unique(ak.flatten(stationProphit['etaID']).to_numpy()).shape} unique etapartitions for GE{station}1")

    flattenedEtaID = ak.flatten(stationProphit["etaID"])
    flattenedPhiMax = ak.flatten(stationProphit["mu_propagated_EtaPartition_phiMax"])
    flattenedPhiMin = ak.flatten(stationProphit["mu_propagated_EtaPartition_phiMin"])
    flattenedRMax = ak.flatten(stationProphit["mu_propagated_EtaPartition_rMax"])
    flattenedRMin = ak.flatten(stationProphit["mu_propagated_EtaPartition_rMin"])

    dtype_arr1 = np.dtype([('etaID', np.int64), ('phiMax', np.float64), ('phiMin', np.float64), ('rMax', np.float64), ('rMin', np.float64)])
    allBoundaries = np.array(list(zip(flattenedEtaID,flattenedPhiMax,flattenedPhiMin,flattenedRMax,flattenedRMin)),dtype=dtype_arr1)
    uniqueBoundaries = np.unique(allBoundaries)

    for e in uniqueBoundaries:
        print(f"boundariesGE{station}1[{e['etaID']}]=[{e['phiMax']},{e['phiMin']},{e['rMax']},{e['rMin']}]")

for s in outs: print(s)
