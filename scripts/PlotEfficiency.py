import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import copy
from Utils import OUTPUT_PATH, PHPINDEX_FILE
from Statistics import generateClopperPearsonInterval
from PlottingFunctions import axs_36chambersEff_style,axs_4GE21modulesEff_style,GE21PlottableChambers

aggregation_functions = {
        "Station": "first",
        "Region": "first",
        "Layer": "first",
        "Chamber": "first",
        "ChamberType": "first",
        "matchedRecHit": "sum",
        "propHit": "sum",
        }

def AddConfidenceIntervals(efficiency_df):
    efficiency_df["eff_lower_limit"] = efficiency_df.apply(
        lambda x: generateClopperPearsonInterval(x["matchedRecHit"], x["propHit"])[0],axis=1)
        
    efficiency_df["eff_upper_limit"] = efficiency_df.apply(
        lambda x: generateClopperPearsonInterval(x["matchedRecHit"], x["propHit"])[1],axis=1)
    
    efficiency_df["avg_eff"] = efficiency_df.apply(lambda x: x["matchedRecHit"] / x["propHit"], axis=1)
    
    efficiency_df["eff_low_error"] = efficiency_df["avg_eff"] - efficiency_df["eff_lower_limit"]
    efficiency_df["eff_up_error"] = efficiency_df["eff_upper_limit"] - efficiency_df["avg_eff"]
    
    return efficiency_df


def plotGE11wheel(df_GE11wheel, ax, _color, label, title):
    global aggregation_functions
    df_GE11wheel = df_GE11wheel.groupby(df_GE11wheel["Chamber"]).aggregate(aggregation_functions)
    df_GE11wheel = AddConfidenceIntervals(df_GE11wheel)

    ax.bar(
        df_GE11wheel["Chamber"],
        df_GE11wheel["eff_upper_limit"] - df_GE11wheel["eff_lower_limit"],
        bottom=df_GE11wheel["eff_lower_limit"],
        width=0.5,
        color=_color,
        zorder=0,
        align="center",
        label=label,
        alpha=0.5,
    )
    ax.errorbar(
        df_GE11wheel["Chamber"],
        df_GE11wheel["avg_eff"],
        yerr=[df_GE11wheel["eff_low_error"], df_GE11wheel["eff_up_error"]],
        ecolor=_color,
        fmt="none",
        capsize=9,
    )
    ax.set_title(title, fontweight="bold", size=24)
    return ax

def plotGE21chamber(df_GE21chamber, ax, _color, label, title):
    global aggregation_functions
    df = df_GE21chamber.groupby(["ChamberType"]).aggregate(aggregation_functions)
    df = AddConfidenceIntervals(df)
    
    ax.bar(
        df["ChamberType"],
        df["eff_upper_limit"] - df["eff_lower_limit"],
        bottom=df["eff_lower_limit"],
        width=0.5,
        color=_color,
        zorder=0,
        align="center",
        label=label,
        alpha=0.5,
    )
    ax.errorbar(
        df["ChamberType"],
        df["avg_eff"],
        yerr=[df["eff_low_error"], df["eff_up_error"]],
        ecolor=_color,
        fmt="none",
        capsize=9,
    )
    ax.set_title(title,fontweight="bold",size=24)
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

    fig_GE11efficiency, axs_GE11efficiency = plt.subplots(nrows=2, ncols=2, figsize=(39, 20), layout="constrained")
    axs_GE11efficiency = axs_GE11efficiency.flatten()
    
    fig_GE21efficiency, axs_GE21efficiency = plt.subplots(nrows=len(GE21PlottableChambers), ncols=1, figsize=(10, 15), layout="constrained")
    axs_GE21efficiency = axs_GE21efficiency.flatten()

    ### GE11
    station = 1
    for index, file_path in enumerate(args.inputs):
        df = pd.read_csv(file_path, sep=",")
        df = df[df["propHit"] != 0]
        for idx, (region, layer) in enumerate([(-1, 1), (1, 1), (-1, 2), (1, 2)]):
            temp_df = df[(df.Region == region) & (df.Layer == layer)]
            axs_GE11efficiency[idx] = plotGE11wheel(temp_df[temp_df.Station == station], axs_GE11efficiency[idx], colors[index], label_list[index], f"GE{'+' if region>0 else '-'}{station}1 Ly{layer}",)

    ### GE21
    for idx, chID in enumerate(GE21PlottableChambers):
        query_ch = f"(Station=={chID[0]} & Region=={chID[1]} & Chamber=={chID[2]} & Layer=={chID[3]})"
        temp_df = df.query(query_ch)
        plotGE21chamber(temp_df, axs_GE21efficiency[idx], colors[0], label_list[0], f"GE{chID[0]}1-{'P' if chID[1]>0 else 'M'}-{chID[2]}L{chID[3]}")


    ## apply plot style
    axs_GE11efficiency = np.array(list(map(axs_36chambersEff_style, axs_GE11efficiency)))
    axs_GE21efficiency = np.array(list(map(axs_4GE21modulesEff_style, axs_GE21efficiency)))
    
    fig_GE11efficiency.savefig(Path(output_folder_path, f"GE11Eff_{timestamp}.png"), dpi=200)
    fig_GE11efficiency.savefig(Path(output_folder_path, f"GE11Eff_{timestamp}.pdf"))
    fig_GE21efficiency.savefig(Path(output_folder_path, f"GE21Eff_{timestamp}.png"), dpi=200)
    fig_GE21efficiency.savefig(Path(output_folder_path, f"GE21Eff_{timestamp}.pdf"))
