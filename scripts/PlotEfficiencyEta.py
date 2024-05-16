import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import copy
from Utils import OUTPUT_PATH, PHPINDEX_FILE
from Statistics import AddConfidenceIntervals
from PlottingFunctions import axs_8EtasEff_style

"""
ONLY FOR GE11
In GE21 the conversion VFAT --> EtaPartition is not as easy.
Therefore the input csv has to be already grouped by EtaPartition
"""


aggregation_functions = {
        "Station": "first",
        "Region": "first",
        "Layer": "first",
        "Chamber": "first",
        "ChamberType": "first",
        "EtaPartition": "first",
        "matchedRecHit": "sum",
        "propHit": "sum",
        }


def VFAT2iEta(VFATN):
    vfatPosition = 0
    try:
        vfatPosition = int(VFATN)
    except:
        print(f"VFAT Number provided {VFATN} is not a number.\nExiting...")
        

    if vfatPosition <0 or vfatPosition>23:
        print("Invalid VFAT position.\nExiting...")

    iEta = (8 - vfatPosition%8)
    iPhi = vfatPosition//8
    return iEta

def plotGE11EfficiencyEta(df, ax, _color, label, title):
    ## add etaPartition from the VFAT number
    dfGE11 = df.copy()
    dfGE11["EtaPartition"] = dfGE11.apply(lambda x: VFAT2iEta(x["VFAT"]),axis=1)
    
    global aggregation_functions
    dfGE11 = dfGE11.groupby(["EtaPartition"]).aggregate(aggregation_functions)
    dfGE11 = AddConfidenceIntervals(dfGE11)  
    
    ax.bar(
        dfGE11["EtaPartition"],
        dfGE11["eff_upper_limit"] - dfGE11["eff_lower_limit"],
        bottom=dfGE11["eff_lower_limit"],
        width=0.5,
        color=_color,
        zorder=0,
        align="center",
        label=label,
        alpha=0.5,
    )
    ax.errorbar(
        dfGE11["EtaPartition"],
        dfGE11["avg_eff"],
        yerr=[dfGE11["eff_low_error"], dfGE11["eff_up_error"]],
        ecolor=_color,
        fmt="none",
        capsize=9,
    )
    ax.set_title(
        title,
        
        fontweight="bold",
        size=24,
    )
    return ax



##### General
plt.rc("grid", linestyle="-.", color=(0.64, 0.64, 0.64))
timestamp = time.strftime("%-y%m%d_%H%M")
#####

##### Parser
parser = argparse.ArgumentParser(description="Efficiency plotter from csv")
parser.add_argument("inputs", help="Input csv(s)", nargs="*")
parser.add_argument("--folder_name", type=str, help="Output folder name, appended to OUTPUT_PATH. Defaults to 'test'", required=False, default="test")
parser.add_argument("--labels", type=str, help="Label with which the runs should be listed in the legend (according to inputs order). If not provided, input names will be used", required=False, nargs="*")
args = parser.parse_args()

if __name__ == '__main__':
    output_folder_path = Path(OUTPUT_PATH, args.folder_name)
    label_list = args.labels if args.labels is not None else args.inputs
    colors = plt.cm.rainbow(np.linspace(0, 1, len(label_list)))
    print(output_folder_path)
    if len(label_list) != len(args.inputs):
        print("Parsed inputs and labels are different in number...\nExiting ..")
        sys.exit(0)

    output_folder_path.mkdir(parents=True, exist_ok=True)
    if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, output_folder_path)
    #####

    fig_Eta, axs_EtaEfficiency = plt.subplots(nrows=2, ncols=2, figsize=(39, 20), layout="constrained")
    axs_EtaEfficiency = axs_EtaEfficiency.flatten()

    ### GE11
    station = 1
    for index, file_path in enumerate(args.inputs):
        df = pd.read_csv(file_path, sep=",")
        df = df[df["propHit"] != 0]
        for idx, (region, layer) in enumerate([(-1, 1), (1, 1), (-1, 2), (1, 2)]):
            temp_df = df[(df.Region == region) & (df.Layer == layer)]
            axs_EtaEfficiency[idx] = plotGE11EfficiencyEta(temp_df[temp_df.Station == station], axs_EtaEfficiency[idx], colors[index], label_list[index], f"GE{'+' if region>0 else '-'}{station}1 Ly{layer}",)

    ## apply plot style
    axs_EtaEfficiency = np.array(list(map(axs_8EtasEff_style, axs_EtaEfficiency)))
    
    
    fig_Eta.savefig(Path(output_folder_path, f"GE11EffEta_{timestamp}.png"), dpi=200)
    fig_Eta.savefig(Path(output_folder_path, f"GE11EffEta_{timestamp}.pdf"))
