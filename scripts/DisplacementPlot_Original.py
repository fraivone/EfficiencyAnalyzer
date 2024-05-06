import argparse
import glob
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import copy
from Utils import OUTPUT_PATH, PHPINDEX_FILE
from Statistics import generateClopperPearsonInterval
from PlottingFunctions import axs_36chambersEff_style
import hist


mapfile = {
    "-5.0":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131046/",
    "-3.0":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131051/",
    "-2.5":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131048/",
    "-2.2":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131054/",
    "-2.0":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131040/",
    "-1.8":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131055/",
    "-1.0":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131042/",
    "-0.55":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131047/",
    "0.0":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131052/",
    "+0.55":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131041/",
    "+1.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131045/",
    "+1.80":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131056/",
    "+2.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131044/",
    "+2.20":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131053/",
    "+2.50":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131049/",
    "+3.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131050/",
    "+5.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306131043/",

    }

##### General
plt.rc("grid", linestyle="-.", color=(0.64, 0.64, 0.64))
timestamp = time.strftime("%-y%m%d_%H%M")
#####


output_folder_path = Path(OUTPUT_PATH, "367758")
output_folder_path.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, output_folder_path)

current_folder = Path(output_folder_path,f"Displacement")
current_folder.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,current_folder)

perChamber = Path(current_folder,f"PerChamber")
perChamber.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,perChamber)


displacement_array = []
df_collector = {}
fig_long,axs_long = plt.subplots(1, 1, figsize=(30,10))
fig_short,axs_short = plt.subplots(1, 1, figsize=(30,10))
### DATA FETCHING AND SUMMARY PLOTS
for displ in mapfile:
    try:
        df = pd.read_csv(glob.glob(mapfile[displ]+"FitResults.txt")[0],sep=",")
    except:
        print(f"Couldn't find data for displacement = {displ}")
        continue
    df = df.sort_values(ascending=False,by='Muon Charge')
    df["Fit Mean Error Squared"] = df["Fit Mean Error"]**2
    all_ch = df.Chamber.to_list()
    ch_long = [k for k in all_ch if int(k[-4:-2]) % 2 == 0]
    ch_short = [k for k in all_ch if int(k[-4:-2]) % 2 == 1]
    df_long = df[df["Chamber"].isin(ch_long)]
    df_short = df[df["Chamber"].isin(ch_short)]

    df_longL1 = df[df["Chamber"].isin([k for k in ch_long if "L1" in k])]
    df_longL2 = df[df["Chamber"].isin([k for k in ch_long if "L2" in k])]
    df_shortL1 = df[df["Chamber"].isin([k for k in ch_short if "L1" in k])]
    df_shortL2 = df[df["Chamber"].isin([k for k in ch_short if "L2" in k])]

    a = df.groupby('Chamber',as_index=False)
    df["Peak Separation"]= a["Fit Mean"].diff(axis=0)
    df["Peak Separation Error"]= a["Fit Mean Error Squared"].cumsum()
    # df["Peak Separation Error"]= ((df['Fit Mean Error']**2).sum()/2).where((df['Chamber'] < 0) | (df['B'] > 0), df['A'] / df['B'])
    df = df.sort_values(ascending=True,by=['Chamber',"Muon Charge"])
    df_collector[displ] = df
    
    
    df_long = df[df["Chamber"].isin(ch_long)]
    df_short = df[df["Chamber"].isin(ch_short)]
    

    axs_long.errorbar(list(range(len(df_long))),df_long["Peak Separation"].to_list(),yerr = df_long["Peak Separation Error"].to_list(),fmt='o',label = f"Long displacement = {displ} ")    
    axs_long.set_ylabel("Peak Separation (cm)")
    axs_long.set_xticks(list(range(len(df_long))),df_long["Chamber"].to_list(),rotation=90)

    axs_short.errorbar(list(range(len(df_short))),df_short["Peak Separation"].to_list(),yerr = df_short["Peak Separation Error"].to_list(),fmt='o',label = f"Short displacement = {displ} ")    
    axs_short.set_ylabel("Peak Separation (cm)")
    axs_short.set_xticks(list(range(len(df_short))),df_short["Chamber"].to_list(),rotation=90)

fig_long.tight_layout()
axs_long.legend()
axs_long.grid(True, which='both')
axs_long.axhline(y=0, color='k')
axs_long.axvline(x=0, color='k')
fig_long.savefig(f"{current_folder}/SummarySeparation_Long.png")
fig_short.tight_layout()
axs_short.legend()
axs_short.grid(True, which='both')
axs_short.axhline(y=0, color='k')
axs_short.axvline(x=0, color='k')
fig_short.savefig(f"{current_folder}/SummarySeparation_Short.png")
### END DATA FETCHING AND SUMMARY PLOTS

### OTHER PLOTS
fig,axs = plt.subplots(1, 1, figsize=(10,10))
displacements = [float(k) for k in df_collector.keys()]

plot_coordinate = {
    "NegativeEndcap":"M",
    "PositiveEndcap":"P",}

for title in plot_coordinate:
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L1')) & (df_collector[k]['Chamber'].isin(ch_long))]["Peak Separation"].mean() for k in df_collector],fmt='-o',label = f"Long Layer 1")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L2')) & (df_collector[k]['Chamber'].isin(ch_long))]["Peak Separation"].mean() for k in df_collector],fmt='-o',label = f"Long Layer 2")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L1')) & (df_collector[k]['Chamber'].isin(ch_short))]["Peak Separation"].mean() for k in df_collector],fmt='-o',label = f"Short Layer 1")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L2')) & (df_collector[k]['Chamber'].isin(ch_short))]["Peak Separation"].mean() for k in df_collector],fmt='-o',label = f"Short Layer 2")    
    
    axs.set_ylabel("Peak Separation (cm)")
    axs.set_xlabel("Propagation Displacement (cm)")
    fig.tight_layout()
    axs.legend()
    axs.grid(True, which='both')
    axs.axhline(y=0, color='k')
    axs.axvline(x=0, color='k')
    fig.savefig(f"{current_folder}/{title}_AVFPeakSeparation.png")
    axs.cla()

    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L1')) & (df_collector[k]['Chamber'].isin(ch_long)) & (df_collector[k]['Muon Charge'] == +1)]["Fit Sigma"].mean() for k in df_collector],fmt='-o',label = f"PosiMuons Long Layer 1")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L2')) & (df_collector[k]['Chamber'].isin(ch_long)) & (df_collector[k]['Muon Charge'] == +1)]["Fit Sigma"].mean() for k in df_collector],fmt='-o',label = f"PosiMuons Long Layer 2")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L1')) & (df_collector[k]['Chamber'].isin(ch_short))& (df_collector[k]['Muon Charge'] == +1)]["Fit Sigma"].mean() for k in df_collector],fmt='-o',label = f"PosiMuons Short Layer 1")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L2')) & (df_collector[k]['Chamber'].isin(ch_short))& (df_collector[k]['Muon Charge'] == +1)]["Fit Sigma"].mean() for k in df_collector],fmt='-o',label = f"PosiMuons Short Layer 2")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L1')) & (df_collector[k]['Chamber'].isin(ch_long)) & (df_collector[k]['Muon Charge'] == -1)]["Fit Sigma"].mean() for k in df_collector],fmt='-o',label = f"NegaMuons Long Layer 1")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L2')) & (df_collector[k]['Chamber'].isin(ch_long)) & (df_collector[k]['Muon Charge'] == -1)]["Fit Sigma"].mean() for k in df_collector],fmt='-o',label = f"NegaMuons Long Layer 2")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L1')) & (df_collector[k]['Chamber'].isin(ch_short))& (df_collector[k]['Muon Charge'] == -1)]["Fit Sigma"].mean() for k in df_collector],fmt='-o',label = f"NegaMuons Short Layer 1")    
    axs.errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber'].str.contains(plot_coordinate[title])) & (df_collector[k]['Chamber'].str.contains('L2')) & (df_collector[k]['Chamber'].isin(ch_short))& (df_collector[k]['Muon Charge'] == -1)]["Fit Sigma"].mean() for k in df_collector],fmt='-o',label = f"NegaMuons Short Layer 2")    
    
    axs.set_ylabel("AVG Sigma")
    axs.set_xlabel("Propagation Displacement (cm)")
    fig.tight_layout()
    axs.legend()
    axs.grid(True, which='both')
    axs.axhline(y=0, color='k')
    axs.axvline(x=0, color='k')
    fig.savefig(f"{current_folder}/{title}_AVGSigma.png")
    axs.cla()

fig,axs = plt.subplots(2, 1, figsize=(10,10))
for ch in all_ch:
    axs[0].errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==+1) ]["Fit Sigma"].iloc[0] for k in df_collector],yerr = [df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==+1) ]["Fig Sigma Error"].iloc[0] for k in df_collector], fmt='-o',label = f"PosiMuons")    
    axs[0].errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==-1) ]["Fit Sigma"].iloc[0] for k in df_collector],yerr = [df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==-1) ]["Fig Sigma Error"].iloc[0] for k in df_collector], fmt='-o',label = f"NegaMuons")    
    axs[1].errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==+1) ]["Fit Mean"].iloc[0] for k in df_collector],yerr = [df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==+1) ]["Fit Mean Error"].iloc[0] for k in df_collector], fmt='-o',label = f"PosiMuons")    
    axs[1].errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==-1) ]["Fit Mean"].iloc[0] for k in df_collector],yerr = [df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==-1) ]["Fit Mean Error"].iloc[0] for k in df_collector], fmt='-o',label = f"NegaMuons")    
    axs[0].set_ylabel("Fit Sigma")
    axs[0].set_xlabel("Propagation Displacement (cm)")
    axs[0].legend()
    axs[0].grid(True, which='both')
    axs[0].axhline(y=0, color='k')
    axs[0].axvline(x=0, color='k')

    axs[1].set_ylabel("Fit Mean")
    axs[1].set_xlabel("Propagation Displacement (cm)")
    axs[1].grid(True, which='both')
    axs[1].axhline(y=0, color='k')
    axs[1].axvline(x=0, color='k')
    axs[1].legend()
    
    fig.tight_layout()
    fig.savefig(f"{perChamber}/{ch}.png")
    axs[0].cla()
    axs[1].cla()



    

    # df['Volume Diff'] = df.groupby('Name')['Volume'].apply(lambda x: x.diff(1)).combine_first(df['Volume'])


#     short_ML1.append(df_short[[ k for k in df_short.columns if "L1" in k and "M" in k] ].mean(axis=1)[0])
#     short_ML2.append(df_short[[ k for k in df_short.columns if "L2" in k and "M" in k] ].mean(axis=1)[0])
#     short_PL1.append(df_short[[ k for k in df_short.columns if "L1" in k and "P" in k] ].mean(axis=1)[0])
#     short_PL2.append(df_short[[ k for k in df_short.columns if "L2" in k and "P" in k] ].mean(axis=1)[0])
#     long_ML1.append(df_long[[ k for k in df_long.columns if "L1" in k and "M" in k] ].mean(axis=1)[0])
#     long_ML2.append(df_long[[ k for k in df_long.columns if "L2" in k and "M" in k] ].mean(axis=1)[0])
#     long_PL1.append(df_long[[ k for k in df_long.columns if "L1" in k and "P" in k] ].mean(axis=1)[0])
#     long_PL2.append(df_long[[ k for k in df_long.columns if "L2" in k and "P" in k] ].mean(axis=1)[0])
#     displacement_array.append(float(displ))

# fig,axs = plt.subplots(1, 1, figsize=(10,10))
# axs.plot(displacement_array,short_ML1,'-o',label="Short ML1")    
# axs.plot(displacement_array,short_ML2,'-o',label="Short ML2")    
# axs.plot(displacement_array,short_PL1,'-o',label="Short PL1")    
# axs.plot(displacement_array,short_PL2,'-o',label="Short PL2")  
# axs.set_xlabel("GE11 propagation dispacement wrt 130X_dataRun3_Prompt_v3")
# axs.set_ylabel("AVG residuals sigma")
# axs.legend()
# fig.savefig(f"{current_folder}/Short.png")
# axs.cla()
# axs.plot(displacement_array,long_ML1,'-o',label="Long ML1")    
# axs.plot(displacement_array,long_ML2,'-o',label="Long ML2")    
# axs.plot(displacement_array,long_PL1,'-o',label="Long PL1")    
# axs.plot(displacement_array,long_PL2,'-o',label="Long PL2")  
# axs.set_xlabel("GE11 propagation dispacement wrt 130X_dataRun3_Prompt_v3")
# axs.set_ylabel("AVG residuals sigma")
# axs.legend()
# fig.savefig(f"{current_folder}/Long.png")
# axs.cla()
