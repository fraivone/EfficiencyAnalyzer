import argparse
import glob
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import copy
from Utils import EOS_OUTPUT_PATH, EOS_INDEX_FILE
from Statistics import generateClopperPearsonInterval
from PlottingFunctions import axs_36chambersEff_style
import hist
import re

def clear_axes(axs):
    axs.cla()
clear_axes_array = np.vectorize(clear_axes)

def AggregateDFByEta(path_dict):
    tmp_dfs = []
    for eta in path_dict:
        df = pd.read_csv(path_dict[eta])
        df["etaPartition"] = eta
        tmp_dfs.append(df)
    return pd.concat(tmp_dfs)

### ASSUMING HIST.txt with RANGE
RANGE = [-4,4]

##### General
station = 1 ## GE11 only
plt.rc("grid", linestyle="-.", color=(0.64, 0.64, 0.64))
timestamp = time.strftime("%-y%m%d_%H%M")
#####

##### Parser
parser = argparse.ArgumentParser(description="Study on the residuals")
parser.add_argument("inputs", help="Input csv(s)", nargs="*")
parser.add_argument(
    "--folder_name", type=str, help="Output folder name", required=False, default="test")

args = parser.parse_args()
inputs = args.inputs

output_folder_path = Path(EOS_OUTPUT_PATH, args.folder_name)
output_folder_path.mkdir(parents=True, exist_ok=True)
copy(EOS_INDEX_FILE, output_folder_path)






for folder in inputs:
    print(folder )
    all_RMS_files = glob.glob(folder+"*RMS*")
    all_hist_files = glob.glob(folder+"*hist*")
    fig,axs = plt.subplots(1, 1, figsize=(30,10))
    FitResult = []
    ### PLOT TO COMPARE RMS
    # 
    # df_posi = AggregateDFByEta({eta:list(filter(lambda x: f'Eta{eta}' in x, [k for k in all_RMS_files if "Posi" in k]))[0] for eta in range(1,9)})
    # df_nega = AggregateDFByEta({eta:list(filter(lambda x: f'Eta{eta}' in x, [k for k in all_RMS_files if "Nega" in k]))[0] for eta in range(1,9)})
    
    # chamber_list = list(set(list(df_posi.columns)).intersection(list(df_nega.columns)))
    # chamber_list = sorted(chamber_list)
    # n = len(chamber_list)
    
    # axs.errorbar(list(range(n)), [df_posi[chamber_name].iloc[0] for chamber_name in chamber_list], xerr=np.zeros(n), yerr=np.zeros(n), fmt='o', alpha = 0.7,color='blue',label="PosiMuons")
    # axs.errorbar(list(range(n)), [df_nega[chamber_name].iloc[0] for chamber_name in chamber_list], xerr=np.zeros(n), yerr=np.zeros(n), fmt='o', alpha = 0.7,color='red',label="NegaMuons")
    # axs.legend()
    # axs.set_xticks(list(range(n)), chamber_list,rotation=90)
    # axs.set_ylabel("Residuals RMS")
    # fig.tight_layout()
    # axs.grid(which='major', axis='x', linestyle='-.')
    # fig.savefig(f"{folder}/A_ResidualsRMS.png")
    # fig.savefig(f"{folder}/A_ResidualsRMS.pdf")
    ### END PLOT TO COMPARE RMS


    ### FIT AND PLOT FIT RESULTS
    fig_fit, axs_fit = plt.subplots(2, 8, figsize=(80, 10),sharex=True,gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0, wspace=0)
    chamber_list = sorted(list(set([ re.findall(r"GE11-[PM]-\d{2}L[12]",k)[0] for k in all_hist_files])))
    
        
    # for size in ["Long","Short"]:
    #     print(f"Processing {size} chamber")
    #     for layer in [1,2]:
    #         for region in [-1,1]:
    for region in [-1,1]:
        for layer in [1,2]:
            for size in [[9,11,13],[27,29,31]]:
                for eta in range(1,9):
                    print(f"Fitting region {region} Layer {layer} Eta {eta} Size {size}", end="\r")
                    
                    rest = 0 if size == "Long" else 1
                    endcap = "M" if region == -1 else "P" 

                    posi_muons = list(filter(lambda x: f"Eta{eta}" in x in x and f"-{endcap}-" in x and f"L{layer}" in x and int(x.split("/")[-1].split(".txt")[-2][-9:-7]) in size, [k for k in all_hist_files if "Posi" in k]))  # and f"L{layer}" and int(x.split("/")[-1].split(".txt")[-2][-9:-7]) % 2 == rest
                    nega_muons = list(filter(lambda x: f"Eta{eta}" in x in x and f"-{endcap}-" in x and f"L{layer}" in x and int(x.split("/")[-1].split(".txt")[-2][-9:-7]) in size, [k for k in all_hist_files if "Nega" in k])) 
                    
                    posi_residuals = np.sum([np.loadtxt(k) for k in posi_muons], axis = 0 )
                    nega_residuals = np.sum([np.loadtxt(k) for k in nega_muons], axis = 0 )

                    h_nega = hist.Hist.new.Regular(len(nega_residuals), RANGE[0], RANGE[1], name=f"Eta{eta} Residuals",label=r"Residuals R$\Delta$$\phi$ (cm)").Double()
                    h_posi = hist.Hist.new.Regular(len(posi_residuals), RANGE[0], RANGE[1], name=f"Eta{eta} Residuals",label=r"Residuals R$\Delta$$\phi$ (cm)").Double()
                    bin_size = h_nega.axes[0].centers[1] - h_nega.axes[0].centers[0]
                
                    ## fill histograms
                    h_nega[:] = nega_residuals
                    h_posi[:] = posi_residuals
                    ## extract RMS and AVG
                    RMS_nega = np.sqrt(sum(h_nega.values()*h_nega.axes[0].centers**2) / sum(h_nega) )
                    RMS_posi = np.sqrt(sum(h_posi.values()*h_posi.axes[0].centers**2) / sum(h_posi) )
                    AVG_nega = sum(h_nega.values()*h_nega.axes[0].centers) / sum(h_nega)
                    AVG_posi = sum(h_posi.values()*h_posi.axes[0].centers) / sum(h_posi)
                        
                    axs_fit[0][eta-1].set_title(f"$i\eta = {eta}$ residuals", fontsize=16, fontweight='bold', color='black')

                    artist, _ = h_nega[hist.loc(AVG_nega-RMS_nega):hist.loc(AVG_nega+RMS_nega + 1.1*bin_size)].plot_pull("gauss",eb_c="black",fp_c="black",fp_alpha=0.8,eb_label= "Nega Muons",fit_fmt = r"{name} = {value:.3g} $\pm$ {error:.3g}",ax_dict={"main_ax":axs_fit[0][eta-1],"pull_ax":axs_fit[1][eta-1]},pp_num=0,bar_color= "grey",bar_alpha=0.5)
                    ## getting sigma and mu
                    sigma_nega = [ k for k in artist[0]._label.split("\n") if "sigma" in k][0].split("=")
                    sigma_nega = (float(sigma_nega[1].split("$\\pm$")[0]),float(sigma_nega[1].split("$\\pm$")[-1]))
                    mu_nega = [ k for k in artist[0]._label.split("\n") if "mean" in k][0].split("=")
                    mu_nega = (float(mu_nega[1].split("$\\pm$")[0]),float(mu_nega[1].split("$\\pm$")[-1]))
                    FitResult.append (  [region, layer,'_'.join(str(x) for x in size),eta,-1,mu_nega[0],mu_nega[1],sigma_nega[0],sigma_nega[1]]  )
                    axs_fit[0][eta-1].set_xlim(min(AVG_posi-RMS_posi,AVG_nega-RMS_nega),max(AVG_posi+RMS_posi,AVG_nega+RMS_nega))

                    artist, _ = h_posi[hist.loc(AVG_posi-RMS_posi):hist.loc(AVG_posi+RMS_posi + 1.1*bin_size)].plot_pull("gauss",eb_c="red",fp_c="red",fp_alpha=0.8,eb_label = "Posi Muons",fit_fmt = r"{name} = {value:.3g} $\pm$ {error:.3g}",ax_dict={"main_ax":axs_fit[0][eta-1],"pull_ax":axs_fit[1][eta-1]},pp_num=0,bar_color= "red",bar_alpha=0.5)
                    ## getting sigma and mu
                    sigma_posi = [ k for k in artist[0]._label.split("\n") if "sigma" in k][0].split("=")
                    sigma_posi = (float(sigma_posi[1].split("$\\pm$")[0]),float(sigma_posi[1].split("$\\pm$")[-1]))
                    mu_posi = [ k for k in artist[0]._label.split("\n") if "mean" in k][0].split("=")
                    mu_posi = (float(mu_posi[1].split("$\\pm$")[0]),float(mu_posi[1].split("$\\pm$")[-1]))
                    FitResult.append (  [region, layer,'_'.join(str(x) for x in size),eta,+1,mu_posi[0],mu_posi[1],sigma_posi[0],sigma_posi[1]]  )
                    axs_fit[0][eta-1].set_xlim(min(AVG_posi-RMS_posi,AVG_nega-RMS_nega),max(AVG_posi+RMS_posi,AVG_nega+RMS_nega))

                fig_fit.tight_layout()
                fig_fit.savefig(f"{folder}/1aFit_Region{region}_Layer{layer}_{'_'.join(str(x) for x in size)}.png")
                fig_fit.savefig(f"{folder}/1aFit_Region{region}_Layer{layer}_{'_'.join(str(x) for x in size)}.pdf")
                clear_axes_array(axs_fit)
        
    ### END AND PLOT FIT RESULTS
    pd.DataFrame(FitResult,columns=["Region","Layer","Size","EtaP","Muon Charge","Fit Mean","Fit Mean Error","Fit Sigma","Fit Sigma Error"]).to_csv(f"{folder}/ShortOnly_FitResults.txt", index=False)






    # fig,axs = plt.subplots(1, 1, figsize=(30,10))
    # y = [sigma_posi.get(k)[0] if sigma_posi.get(k) is not None else np.nan for k in chamber_list]
    # y_err = [sigma_posi.get(k)[1] if sigma_posi.get(k) is not None else np.nan for k in chamber_list]
    # axs.errorbar(list(range(n)), y, xerr=np.zeros(n), yerr=y_err, fmt='o', color='black',label="PosiMuons")
    # axs.errorbar(list(range(n)), [sigma_nega.get(k)[0] if sigma_nega.get(k) is not None else np.nan for k in chamber_list], xerr=np.zeros(n), yerr=[sigma_nega.get(k)[1] if sigma_nega.get(k) is not None else np.nan for k in chamber_list], fmt='o', color='red',label="NegaMuons")
    # axs.set_ylim(min(y)*0.8,max(y)*1.3)
    # axs.legend()
    # axs.set_xticks(list(range(n)), chamber_list,rotation=90)
    # axs.set_ylabel("Fit sigma")
    # fig.tight_layout()
    # axs.grid(which='major', axis='x', linestyle='-.')
    # fig.savefig(f"{folder}/A_ResidualsSigma.png")
    # fig.savefig(f"{folder}/A_ResidualsSigma.pdf")
    # axs.cla()

    # y_posi = [mu_posi.get(k)[0] if mu_posi.get(k) is not None else np.nan for k in chamber_list]
    # y_posi_err = [mu_posi.get(k)[1] if mu_posi.get(k) is not None else np.nan for k in chamber_list]
    # y_nega = [mu_nega.get(k)[0] if mu_nega.get(k) is not None else np.nan for k in chamber_list]
    # y_nega_err = [mu_nega.get(k)[1] if mu_nega.get(k) is not None else np.nan for k in chamber_list]
    # axs.errorbar(list(range(n)), y_posi, xerr=np.zeros(n), yerr=y_posi_err, fmt='o', color='blue',label="PosiMuons")
    # axs.errorbar(list(range(n)), y_nega, xerr=np.zeros(n), yerr=y_nega_err, fmt='o', color='red',label="NegaMuons")
    # axs.set_ylim(1.3* min (min(y_nega), min(y_posi)) ,1.3* max (max(y_nega), max(y_posi)))
    # axs.legend()
    # axs.set_xticks(list(range(n)), chamber_list,rotation=90)
    # axs.set_ylabel("Fit Mean")
    # axs.grid(which='major', axis='x', linestyle='-.')
    # fig.savefig(f"{folder}/A_ResidualsMean.png")
    # fig.savefig(f"{folder}/A_ResidualsMean.pdf")
    
    