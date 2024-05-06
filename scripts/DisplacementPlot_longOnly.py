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

def clear_axes(axs):
    axs.cla()
clear_axes_array = np.vectorize(clear_axes)

mapfile = {
    "-5.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150827/",
    "-3.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150832/",
    "-2.50":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150829/",
    "-2.20":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150835/",
    "-2.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150821/",
    "-1.80":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150836/",
    "-1.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150823/",
    "-0.55":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150828/",
    "+0.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150833/",
    "+0.55":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150822/",
    "+1.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150826/",
    "+1.80":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150837/",
    "+2.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150825/",
    "+2.20":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150834/",
    "+2.50":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150830/",
    "+3.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150831/",
    "+5.00":"/eos/user/f/fivone/www/P5_Operations/Run3/367758/Residuals_2306150824/",

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

perChamber = Path(current_folder,f"LongOnly")
perChamber.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,perChamber)


displacement_array = []
df_collector = {}
### DATA FETCHING AND SUMMARY PLOTS
for displ in mapfile:
    try:
        df = pd.read_csv(glob.glob(mapfile[displ]+"LongOnly_FitResults.txt")[0],sep=",")
    except:
        print(f"Couldn't find data for displacement = {displ}")
        continue
    df = df.sort_values(ascending=False,by=['Muon Charge','EtaP'])
    df["Fit Mean Error Squared"] = df["Fit Mean Error"]**2
   
    df_long = df[df["Size"] == "8_10_12"]
    df_short = df[df["Size"] == "26_28_30"]
    a = df.groupby(["EtaP","Size","Layer","Region"],as_index=False)
    
        
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df["Peak Separation"]= a["Fit Mean"].diff(axis=0)
    df["Peak Separation Error"]= a["Fit Mean Error Squared"].cumsum()
    # print(f"Displacement {displ} {df}")
    # df = df.sort_values(ascending=True,by=['Chamber',"Muon Charge"])
    df_collector[displ] = df
    
### END DATA FETCHING AND SUMMARY PLOTS

### OTHER PLOTS
fig, axs = plt.subplots(2, 8, figsize=(80, 20),sharex=True,gridspec_kw={'height_ratios': [1, 1]})
displacements = [float(k) for k in df_collector.keys()]

colors = plt.cm.tab10.colors
data = np.zeros((8,17))


last_plot = []
for region in [-1,1]:
    for eta in range(1,9):
        index = 0
        for size in ["8_10_12","26_28_30"]:
            for layer in [1,2]:
                ## peak separation plot
                for _index,displ in enumerate(mapfile.keys()):
                    if displ not in df_collector.keys(): continue
                    y = df_collector[displ][(df_collector[displ]['EtaP']==eta) & (df_collector[displ]['Layer'] == layer) & (df_collector[displ]['Size'] == size) & (df_collector[displ]['Muon Charge'] == -1) & (df_collector[displ]['Region'] == region)]["Peak Separation"].mean()
                    data[8-eta][_index] = y
                    
                    

                data_y = [df_collector[displ][(df_collector[displ]['EtaP']==eta) & (df_collector[displ]['Layer'] == layer) & (df_collector[displ]['Size'] == size) & (df_collector[displ]['Muon Charge'] == -1) & (df_collector[displ]['Region'] == region)]["Peak Separation"].mean() for displ in df_collector]
                data_yerr = [df_collector[displ][(df_collector[displ]['EtaP']==eta) & (df_collector[displ]['Layer'] == layer) & (df_collector[displ]['Size'] == size) & (df_collector[displ]['Muon Charge'] == -1) & (df_collector[displ]['Region'] == region)]["Peak Separation Error"].mean() for displ in df_collector]
                
                displacements
                slope, intercept = np.polyfit(displacements, data_y, 1)
                best_fit_line = slope * np.asarray(displacements) + intercept
                
                axs[0][eta-1].errorbar(displacements,data_y,yerr=data_yerr,fmt='o',label = f"{size} Layer {layer}",color=colors[index])    
                axs[0][eta-1].plot(displacements, best_fit_line,  color=colors[index])
                
                last_plot.append([region,layer,size,eta,-intercept/slope])

                ## fit sigma plot
                data_y = [df_collector[displ][(df_collector[displ]['EtaP']==eta) & (df_collector[displ]['Layer'] == layer) & (df_collector[displ]['Size'] == size) & (df_collector[displ]['Muon Charge'] == -1) & (df_collector[displ]['Region'] == region)]["Fit Sigma"].mean() for displ in df_collector]
                data_yerr = [df_collector[displ][(df_collector[displ]['EtaP']==eta) & (df_collector[displ]['Layer'] == layer) & (df_collector[displ]['Size'] == size) & (df_collector[displ]['Muon Charge'] == -1) & (df_collector[displ]['Region'] == region)]["Fit Sigma Error"].mean() for displ in df_collector]
                axs[1][eta-1].errorbar(displacements,data_y,yerr=data_yerr,fmt='-o',label = f"{size} Layer {layer}",color=colors[index])    
                index += 2
        

        axs[0][eta-1].set_title(f"$i\eta = {eta}$ AVG Seperation", fontsize=20, fontweight='bold', color='black')
        axs[0][eta-1].set_ylabel("Peak Separation (cm)")        
        axs[0][eta-1].legend()
        axs[0][eta-1].grid(True, which='both')
        axs[0][eta-1].axhline(y=0, color='k')
        axs[0][eta-1].axvline(x=0, color='k')
        
        axs[1][eta-1].set_title(f"$i\eta = {eta}$ AVG Sigma", fontsize=20, fontweight='bold', color='black')
        axs[1][eta-1].set_ylabel("AVG Sigma")
        axs[1][eta-1].set_xlabel("Propagation Displacement (cm)")
        axs[1][eta-1].legend()
        axs[1][eta-1].grid(True, which='both')
        axs[1][eta-1].axhline(y=0, color='k')
        axs[1][eta-1].axvline(x=0, color='k')
    
    fig.tight_layout()
    fig.savefig(f"{current_folder}/LongOnly_Region{region}_PeakSeparation.png")
    fig.savefig(f"{current_folder}/LongOnly_Region{region}_PeakSeparation.pdf")
    clear_axes_array(axs)
    
    ## plot 2d hist
    f,a = plt.subplots(1, 1, figsize=(10,10))
    im = a.imshow(data,extent=[-5, 5, 0.5, 8.5],cmap="seismic",vmin=-0.3, vmax=0.3)
    color_bar = f.colorbar(im, ax=a, pad=0.01,shrink=0.65)
    color_bar.ax.text(2.7,0,"Peak Separation (cm)",rotation=90,fontsize=16)
    
    step = 10 / 17 ## x range / number of x bins
    a.set_xticks([-5+step/2 + k*step for k in range(17)], [k for k in mapfile.keys()])
    a.grid()
    a.set_ylabel("Eta Partition")
    a.set_xlabel("Displacement")
    f.tight_layout()
    f.savefig(f"{current_folder}/Region{region}_2DHist.png")

last_df = pd.DataFrame(last_plot,columns=["region","layer","size","eta","Zero Crossing Displacement"])
fig, axs = plt.subplots(1, 2, figsize=(20, 10),sharey=True)
axs[0].set_title(f"Region -1 - Zero crossing displacement ", fontsize=20, fontweight='bold', color='black')
axs[0].plot([last_df[(last_df["region"] == -1) & (last_df["layer"] == 1) & (last_df["size"] == "8_10_12")& (last_df["eta"] == eta)]["Zero Crossing Displacement"].iloc[0] for eta in range(1,9)] ,  list(range(1,9)), linestyle='dashed', marker='o', label="8_10_12 ly1")
axs[0].plot([last_df[(last_df["region"] == -1) & (last_df["layer"] == 2) & (last_df["size"] == "8_10_12")& (last_df["eta"] == eta)]["Zero Crossing Displacement"].iloc[0] for eta in range(1,9)] ,  list(range(1,9)), linestyle='dashed', marker='o', label="8_10_12 ly2")
axs[0].plot([last_df[(last_df["region"] == -1) & (last_df["layer"] == 1) & (last_df["size"] == "26_28_30")& (last_df["eta"] == eta)]["Zero Crossing Displacement"].iloc[0] for eta in range(1,9)] , list(range(1,9)), linestyle='dashed', marker='o', label="26_28_30 ly1")
axs[0].plot([last_df[(last_df["region"] == -1) & (last_df["layer"] == 2) & (last_df["size"] == "26_28_30")& (last_df["eta"] == eta)]["Zero Crossing Displacement"].iloc[0] for eta in range(1,9)] , list(range(1,9)), linestyle='dashed', marker='o', label="26_28_30 ly2")
axs[0].legend()
axs[0].set_ylabel("Eta P")
axs[0].set_xlabel("ZeroCrossing Displacement (cm)")
axs[0].grid()
axs[0].set_xlim(0.2,4.2)

axs[1].set_title(f"Region 1 - Zero crossing displacement ", fontsize=20, fontweight='bold', color='black')
axs[1].plot([last_df[(last_df["region"] == 1) & (last_df["layer"] == 1) & (last_df["size"] == "8_10_12")& (last_df["eta"] == eta)]["Zero Crossing Displacement"].iloc[0] for eta in range(1,9)] , list(range(1,9)),  linestyle='dashed', marker='o', label="8_10_12 ly1")
axs[1].plot([last_df[(last_df["region"] == 1) & (last_df["layer"] == 2) & (last_df["size"] == "8_10_12")& (last_df["eta"] == eta)]["Zero Crossing Displacement"].iloc[0] for eta in range(1,9)] , list(range(1,9)),  linestyle='dashed', marker='o', label="8_10_12 ly2")
axs[1].plot([last_df[(last_df["region"] == 1) & (last_df["layer"] == 1) & (last_df["size"] == "26_28_30")& (last_df["eta"] == eta)]["Zero Crossing Displacement"].iloc[0] for eta in range(1,9)] ,list(range(1,9)),  linestyle='dashed', marker='o', label="26_28_30 ly1")
axs[1].plot([last_df[(last_df["region"] == 1) & (last_df["layer"] == 2) & (last_df["size"] == "26_28_30")& (last_df["eta"] == eta)]["Zero Crossing Displacement"].iloc[0] for eta in range(1,9)] ,list(range(1,9)),  linestyle='dashed', marker='o', label="26_28_30 ly2")
axs[1].legend()
axs[1].set_ylabel("Eta P")
axs[1].set_xlabel("ZeroCrossing Displacement (cm)")
axs[1].grid()
axs[1].set_xlim(-4.2,-0.2)
fig.tight_layout()
fig.savefig(f"{current_folder}/ZeroCrossingDisplacement_Long.png")
# fig,axs = plt.subplots(2, 1, figsize=(10,10))
# for ch in all_ch:
#     axs[0].errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==+1) ]["Fit Sigma"].iloc[0] for k in df_collector],yerr = [df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==+1) ]["Fig Sigma Error"].iloc[0] for k in df_collector], fmt='-o',label = f"PosiMuons")    
#     axs[0].errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==-1) ]["Fit Sigma"].iloc[0] for k in df_collector],yerr = [df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==-1) ]["Fig Sigma Error"].iloc[0] for k in df_collector], fmt='-o',label = f"NegaMuons")    
#     axs[1].errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==+1) ]["Fit Mean"].iloc[0] for k in df_collector],yerr = [df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==+1) ]["Fit Mean Error"].iloc[0] for k in df_collector], fmt='-o',label = f"PosiMuons")    
#     axs[1].errorbar(displacements,[df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==-1) ]["Fit Mean"].iloc[0] for k in df_collector],yerr = [df_collector[k][(df_collector[k]['Chamber']==ch) & (df_collector[k]['Muon Charge']==-1) ]["Fit Mean Error"].iloc[0] for k in df_collector], fmt='-o',label = f"NegaMuons")    
#     axs[0].set_ylabel("Fit Sigma")
#     axs[0].set_xlabel("Propagation Displacement (cm)")
#     axs[0].legend()
#     axs[0].grid(True, which='both')
#     axs[0].axhline(y=0, color='k')
#     axs[0].axvline(x=0, color='k')

#     axs[1].set_ylabel("Fit Mean")
#     axs[1].set_xlabel("Propagation Displacement (cm)")
#     axs[1].grid(True, which='both')
#     axs[1].axhline(y=0, color='k')
#     axs[1].axvline(x=0, color='k')
#     axs[1].legend()
    
#     fig.tight_layout()
#     fig.savefig(f"{perChamber}/{ch}.png")
#     axs[0].cla()
#     axs[1].cla()



    

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
