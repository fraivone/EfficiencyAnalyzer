import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import copy
from Utils import EOS_OUTPUT_PATH, EOS_INDEX_FILE
from Statistics import generateClopperPearsonInterval
from PlottingFunctions import axs_36chambersEff_style

##### General
station = 1 ## GE11 only
plt.rc("grid", linestyle="-.", color=(0.64, 0.64, 0.64))
timestamp = time.strftime("%-y%m%d_%H%M")
#####

##### Parser
parser = argparse.ArgumentParser(description="Efficiency plotter from csv")
parser.add_argument("inputs", help="Input csv(s)", nargs="*")
parser.add_argument(
    "--folder_name", type=str, help="Output folder name", required=False, default="test"
)
parser.add_argument(
    "--labels",
    type=str,
    help="Label with which the runs should be listed in the legend (according to inputs order). If not provided, input names will be used",
    required=False,
    nargs="*",
)
args = parser.parse_args()

output_folder_path = Path(EOS_OUTPUT_PATH, args.folder_name)
label_list = args.labels if args.labels is not None else args.inputs
color = plt.cm.rainbow(np.linspace(0, 1, len(label_list)))

if len(label_list) != len(args.inputs):
    print("Parsed inputs and labels are different in number...\nExiting ..")
    sys.exit(0)

output_folder_path.mkdir(parents=True, exist_ok=True)
copy(EOS_INDEX_FILE, output_folder_path)
#####

fig_efficiency, axs_efficiency = plt.subplots(
    nrows=2, ncols=2, figsize=(39, 20), layout="constrained"
)
axs_efficiency = axs_efficiency.flatten()

for index, file_path in enumerate(args.inputs):
    df = pd.read_csv(file_path, sep=",")
    df = df[df["propHit"] != 0]
    for idx, (region, layer) in enumerate([(-1, 1), (1, 1), (-1, 2), (1, 2)]):
        temp_df = df[(df.Station == 1) & (df.Region == region) & (df.Layer == layer)]
        aggregation_functions = {
            "Station": "first",
            "Region": "first",
            "Layer": "first",
            "Chamber": "first",
            "matchedRecHit": "sum",
            "propHit": "sum",
        }
        temp_df = temp_df.groupby(df["Chamber"]).aggregate(aggregation_functions)
        temp_df["eff_lower_limit"] = temp_df.apply(
            lambda x: generateClopperPearsonInterval(x["matchedRecHit"], x["propHit"])[
                0
            ],
            axis=1,
        )
        
        temp_df["eff_upper_limit"] = temp_df.apply(
            lambda x: generateClopperPearsonInterval(x["matchedRecHit"], x["propHit"])[
                1
            ],
            axis=1,
        )
        temp_df["avg_eff"] = temp_df.apply(
            lambda x: x["matchedRecHit"] / x["propHit"], axis=1
        )
        temp_df["eff_low_error"] = temp_df["avg_eff"] - temp_df["eff_lower_limit"]
        temp_df["eff_up_error"] = temp_df["eff_upper_limit"] - temp_df["avg_eff"]

        axs_efficiency[idx].bar(
            temp_df["Chamber"],
            temp_df["eff_upper_limit"] - temp_df["eff_lower_limit"],
            bottom=temp_df["eff_lower_limit"],
            width=0.5,
            color=color[index],
            zorder=0,
            align="center",
            label=label_list[index],
            alpha=0.5,
        )
        axs_efficiency[idx].errorbar(
            temp_df["Chamber"],
            temp_df["avg_eff"],
            yerr=[temp_df["eff_low_error"], temp_df["eff_up_error"]],
            ecolor=color[index],
            fmt="none",
            capsize=9,
        )
        axs_efficiency[idx].set_title(
            f"GE{'+' if region>0 else '-'}{station}1 Ly{layer}",
            fontweight="bold",
            size=24,
        )

axs_efficiency = np.array(list(map(axs_36chambersEff_style, axs_efficiency)))
fig_efficiency.savefig(Path(output_folder_path, f"Eff_{timestamp}.png"), dpi=200)
fig_efficiency.savefig(Path(output_folder_path, f"Eff_{timestamp}.pdf"))
