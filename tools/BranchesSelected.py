## To be unpacked from the n-tuples
RechitBranchesToUnpack = [
    "gemRecHit_region",
    "gemRecHit_chamber",
    "gemRecHit_layer",
    "gemRecHit_etaPartition",
    "gemRecHit_g_phi",
    "gemRecHit_firstClusterStrip",
    "gemRecHit_station",
    "gemRecHit_cluster_size",
]
ProphitBranchesToUnpack = [
    "mu_propagated_charge",
    "mu_propagated_station",
    "mu_propagated_region",
    "mu_propagated_chamber",
    "mu_propagated_layer",
    "mu_propagated_etaP",
    "mu_propagated_strip",
    "mu_propagated_isME11",
    "mu_propagatedGlb_phi",
    "mu_propagatedGlb_r",
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

## To be stored in the output ROOT file
## separated by scope as some have to be added to those unpacked from the ntuples
ROOT_PropHitBranches = [
    "mu_propagatedLoc_x",
    "mu_propagatedLoc_y",
    #"mu_propagatedGlb_phi",
    #"mu_propagatedGlb_r",
    "mu_propagated_station",
    "mu_propagated_region",
    "mu_propagated_chamber",
    "mu_propagated_layer",
    "mu_propagated_etaP",
    "mu_propagated_strip",
    "mu_propagated_pt",
    "mu_propagatedGlb_errR",
    "mu_propagatedGlb_errPhi",
    "mu_propagatedLoc_dirX",
    "mu_propagatedLoc_dirY",
    "mu_propagatedGlb_x",
    "mu_propagatedGlb_y"
]
ROOT_MatchedHitBranches = [
    "residual_phi",
    "residual_rdphi"
]
ROOT_eventBranches = [
    "mu_propagated_lumiblock"
]
