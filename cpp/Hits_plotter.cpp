#include "VFATEffPlotter.h"
#include "TChain.h"
int main(){
    TCanvas *c1 = new TCanvas();
    TChain *chain = new TChain("muNtupleProducer/MuDPGTree");
    chain->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/Muon/360019_ZMu/221106_020350/0000/MuDPGNtuple_1.root");
    chain->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/Muon/360019_ZMu/221106_020350/0000/MuDPGNtuple_2.root");
    chain->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/Muon/360019_ZMu/221106_020350/0000/MuDPGNtuple_3.root");
    chain->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/Muon/360019_ZMu/221106_020350/0000/MuDPGNtuple_4.root");
    chain->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/Muon/360019_ZMu/221106_020350/0000/MuDPGNtuple_5.root");
    chain->Draw("gemRecHit_g_y:gemRecHit_g_x>>htemp(1000,-300,-100,1000,0,70)","gemRecHit_region==1 & gemRecHit_chamber==18 & gemRecHit_layer==1","");
    auto h = gDirectory->Get("htemp");
    h->Draw("COLZ");
    c1->SaveAs("ZMu_360019.png");
    chain->Draw("gemRecHit_g_z>>h(100,500,700)","gemRecHit_region==1 & gemRecHit_chamber==18 & gemRecHit_layer==1","");
    c1->SaveAs("ZMu_z_360019.png");



    TChain *chain2 = new TChain("muNtupleProducer/MuDPGTree");
    chain2->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/CRAB_UserFiles/b_gemReReco_hv_re3_230213_023322_0000/230409_084453/0000/MuDPGNtuple_1.root");
    chain2->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/CRAB_UserFiles/b_gemReReco_hv_re3_230213_023322_0000/230409_084453/0000/MuDPGNtuple_3.root");
    chain2->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/CRAB_UserFiles/b_gemReReco_hv_re2_230117_015414_0000/230409_105226/0000/MuDPGNtuple_115.root");
    chain2->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/CRAB_UserFiles/b_gemReReco_hv_re2_230117_015414_0000/230409_105226/0000/MuDPGNtuple_128.root");
    chain2->Add("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/CRAB_UserFiles/b_gemReReco_hv_re2_230117_015414_0000/230409_105226/0000/MuDPGNtuple_125.root");
    chain2->Draw("gemRecHit_g_y:gemRecHit_g_x>>htemp(1000,-300,-100,1000,0,70)","gemRecHit_region==1 & gemRecHit_chamber==18 & gemRecHit_layer==1","");
    h = gDirectory->Get("htemp");
    h->Draw("COLZ");
    c1->SaveAs("reRECO_360019.png");
    chain2->Draw("gemRecHit_g_z>>h(100,500,700)","gemRecHit_region==1 & gemRecHit_chamber==18 & gemRecHit_layer==1","");
    c1->SaveAs("reRECO_z_360019.png");
}
