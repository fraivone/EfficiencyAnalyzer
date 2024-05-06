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

### ASSUMING HIST.txt with RANGE
RANGE = [-4,4]

##### General
station = 1 ## GE11 only
plt.rc("grid", linestyle="-.", color=(0.64, 0.64, 0.64))
timestamp = time.strftime("%-y%m%d_%H%M")
#####

##### Parser
parser = argparse.ArgumentParser(description="Study on the residuals")
parser.add_argument("inputs", help="Input csv(s)", nargs=2)
parser.add_argument(
    "--folder_name", type=str, help="Output folder name", required=False, default="test")
parser.add_argument(
    "--labels", type=str, help="label for the 2 distributions", required=False,nargs=2, default=None)
args = parser.parse_args()

inputs = args.inputs

output_folder_path = Path(OUTPUT_PATH, args.folder_name)
output_folder_path.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, output_folder_path)

output_folder_residualratio = Path(output_folder_path,f"Residuals_Ratio")
output_folder_residualratio.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,output_folder_residualratio)

current_folder = Path(output_folder_residualratio,timestamp)
current_folder.mkdir(parents=True, exist_ok=True)
if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE,current_folder)


file = open(Path(current_folder,"Info.txt"),'w')
file.write(f"Residuals ratio between\n")
file.write(f"{inputs[0].replace('/eos/user/f/fivone/www/', 'https://fivone4pfa.web.cern.ch/')} \n")
file.write(f"{inputs[1].replace('/eos/user/f/fivone/www/', 'https://fivone4pfa.web.cern.ch/')} \n")
file.close()

df0 = pd.read_csv(glob.glob(inputs[0]+"ResidualsRMS*")[0],sep=",")
df1 = pd.read_csv(glob.glob(inputs[1]+"ResidualsRMS*")[0],sep=",")
name0 = inputs[0][-11:-1] if args.labels is None else args.labels[0]
name1 = inputs[1][-11:-1] if args.labels is None else args.labels[1]

chamber_list = list(set(list(df0.columns)).intersection(list(df1.columns)))
chamber_list = sorted(chamber_list)


h0 = hist.Hist(hist.axis.StrCategory(chamber_list),label="ChamberIndex")
h1 = hist.Hist(hist.axis.StrCategory(chamber_list),label="ChamberIndex")
max_y = max(df0.max(axis=1)[0],df1.max(axis=1)[0])
for k in chamber_list:
    h0[hist.loc(k)] = df0[k].iloc[0]
    h1[hist.loc(k)] = df1[k].iloc[0]

fig,axs = plt.subplots(1, 1, figsize=(30,10))
main_ax_artists, sublot_ax_arists = h0.plot_ratio(
                h1,
                rp_ylabel=f"Ratio {name0} / {name1}",
                rp_num_label=f"{name0}",
                rp_denom_label=f"{name1}",
                rp_uncert_draw_type="line",  # line or bar
            )
main_ax_artists[0][0][1].lines[2][0].set_segments([]) ## remove error bars h1
main_ax_artists[1][0][1].lines[2][0].set_segments([]) ## remove error bars h0
sublot_ax_arists[1].lines[2][0].set_segments([])      ## remove error bars ratio plot
sublot_ax_arists[0]._axes.set_xticklabels(chamber_list, rotation=90) ## rotate ratio labels ticks x axis
sublot_ax_arists[0]._axes.set_ylim(0,max_y+1) ## set ratio inf lim to 0
main_ax_artists[0][0][0]._axes.set_ylim(0,int(max_y+1)) ## set main plot limy
main_ax_artists[0][0][0]._axes.set_ylabel("Residuals RMS (cm)") ## set main plot y title
fig.savefig(f"{current_folder}/Ratio.png")
fig.savefig(f"{current_folder}/Ratio.pdf")
axs.clear()
fig.clear()


fig, axs = plt.subplots(2, 1, figsize=(10, 10),sharex=True,gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(hspace=0.001, bottom=0.2)
h0 = hist.Hist(hist.axis.StrCategory(chamber_list),label="ChamberIndex")
h1 = hist.Hist(hist.axis.StrCategory(chamber_list),label="ChamberIndex")

sigma0 = {}
sigma1 = {}
for ch in chamber_list:
    f0 = glob.glob(inputs[0]+f"*{ch}*txt")
    f1 = glob.glob(inputs[1]+f"*{ch}*txt")
    if len(f1)==1 and len(f0)==1:
        print(ch)
        f0 = f0[0]
        f1 = f1[0]
        array0 = np.loadtxt(f0, dtype='int16')
        array1 = np.loadtxt(f1, dtype='int16')

        if np.sum(array0) ==0 or np.sum(array1) ==0:
            continue
        
        h0 = hist.Hist.new.Regular(len(array0), RANGE[0], RANGE[1], name="ch Residuals",label=r"Residuals R$\Delta$$\phi$ (cm)").Double()
        h1 = hist.Hist.new.Regular(len(array1), RANGE[0], RANGE[1], name="ch Residuals",label=r"Residuals R$\Delta$$\phi$ (cm)").Double()
        bin_size = h0.axes[0].centers[1] - h0.axes[0].centers[0]
        
        h0[:] = array0[:]
        h1[:] = array1[:]
        h0.plot(ax=axs[0],label=name0)
        h1.plot(ax=axs[0],label=name1)
        axs[0].legend()
        

        RMS0 = np.sqrt(sum(h0.values()*h0.axes[0].centers**2) / sum(h0) )
        RMS1 = np.sqrt(sum(h1.values()*h1.axes[0].centers**2) / sum(h1) )
        AVG0 = sum(h0.values()*h0.axes[0].centers) / sum(h0)
        AVG1 = sum(h1.values()*h1.axes[0].centers) / sum(h1)
        
        # Calculate the pulls, pulls errors ignoring numpy warning when dividing by 0
        with np.errstate(divide='ignore', invalid='ignore'):
            pulls = (h0.values() - h1.values()) / np.sqrt(h0.variances() + h1.variances())
            error_pulls = np.sqrt(h0.variances() + h1.variances()) / np.sqrt(h0.values()**2 + h1.values()**2)
        # plot the error_pull
        import warnings
        # num == 0 avoids to plot bands with colors. However causes a division by zero. Here ignoring the warning
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in double_scalars")
        hist.plot.plot_pull_array(h0,
                      pulls=pulls,
                      ax=axs[1],
                      bar_kwargs={"color":"grey","alpha":0.},
                      pp_kwargs = {"num":0}
                        )
        # plot the pull errors
        axs[1].errorbar(h0.axes[0].centers, pulls, yerr=error_pulls, fmt='none', ecolor='black', capsize=3)

        fig.savefig(f"{current_folder}/_Pull_{ch}.png")
        fig.savefig(f"{current_folder}/_Pull_{ch}.pdf")
        axs[0].cla()
        axs[1].cla()
        artist, _ = h0[hist.loc(AVG0-RMS0):hist.loc(AVG0+RMS0 + 1.1*bin_size)].plot_pull("gauss",fit_fmt = r"{name} = {value:.3g} $\pm$ {error:.3g}",ax_dict={"main_ax":axs[0],"pull_ax":axs[1]},pp_num=0,bar_color= "grey",bar_alpha=0.5)
        sigma = [ k for k in artist[0]._label.split("\n") if "sigma" in k][0].split("=")
        sigma0[ch] = (float(sigma[1].split("$\\pm$")[0]),float(sigma[1].split("$\\pm$")[-1]))
        fig.savefig(f"{inputs[0]}/Fit_{ch}.png")
        fig.savefig(f"{inputs[0]}/Fit_{ch}.pdf")
        axs[0].cla()
        axs[1].cla()
        
        artist, _ = h1[hist.loc(AVG1-RMS1):hist.loc(AVG1+RMS1 + 1.1*bin_size)].plot_pull("gauss",fit_fmt = r"{name} = {value:.3g} $\pm$ {error:.3g}",ax_dict={"main_ax":axs[0],"pull_ax":axs[1]},pp_num=0,bar_color= "grey",bar_alpha=0.5)
        sigma = [ k for k in artist[0]._label.split("\n") if "sigma" in k][0].split("=")
        sigma1[ch] = (float(sigma[1].split("$\\pm$")[0]),float(sigma[1].split("$\\pm$")[-1]))
        fig.savefig(f"{inputs[1]}/Fit_{ch}.png")
        fig.savefig(f"{inputs[1]}/Fit_{ch}.pdf")
        axs[0].cla()
        axs[1].cla()
        warnings.resetwarnings()
    else:
        raise ValueError(f"Failed to find txt candidates for chamber {ch}")

fig,axs = plt.subplots(1, 1, figsize=(30,10))
n = len(sigma0)
axs.errorbar(list(range(n)), [value[0] for key, value in sigma0.items()], xerr=np.zeros(n), yerr=[value[1] for key, value in sigma0.items()], fmt='o', color='blue',label=name0)
axs.errorbar(list(range(n)), [value[0] for key, value in sigma1.items()], xerr=np.zeros(n), yerr=[value[1] for key, value in sigma1.items()], fmt='o', color='red',label=name1)
axs.legend()
axs.set_xticks(list(range(n)), list(sigma0.keys()),rotation=90)
axs.set_ylabel("Fit sigma")
fig.savefig(f"{current_folder}/Sigma.png")
fig.savefig(f"{current_folder}/Sigma.pdf")

sigma0 = {key:value[0] for key,value in sigma0.items() }
pd.DataFrame(sigma0,index=[0]).to_csv(f"{inputs[0]}/ResidualsSigma.txt", index=False)
sigma1 = {key:value[0] for key,value in sigma1.items() }
pd.DataFrame(sigma1,index=[0]).to_csv(f"{inputs[1]}/ResidualsSigma.txt", index=False)
#####
