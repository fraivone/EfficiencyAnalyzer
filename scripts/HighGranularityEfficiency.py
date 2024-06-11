import argparse
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pathlib import Path
from shutil import copy
from Utils import OUTPUT_PATH, PHPINDEX_FILE
import time


newcmp = cm.get_cmap('viridis', 256)
## set color_under to white
newcmp.set_under(np.array([1, 1, 1, 1]))

##### Parser
parser = argparse.ArgumentParser(description="High Granularity Efficiency Plotter from ROOT file")
parser.add_argument("inputs", help="Input root", nargs="*")
parser.add_argument("--folder_name", type=str, help="Output folder name, appended to OUTPUT_PATH. Defaults to 'test'", required=False, default="test")
parser.add_argument("--outputprefix", type=str, help="Prefix to the output file names. Defaults to execution's yymmddHHmm",default=time.strftime("%-y%m%d%H%M_"))
parser.add_argument("--binSizeX", type=float, help="Step size for X bins in mm. Defaults to 10", required=False, default=10)
parser.add_argument("--binSizeY", type=float, help="Step size for Y bins in mm. Defaults to 10", required=False, default=10)
parser.add_argument("--NhitsCutOff", type=int, help="Minimum number of propagated hits per bin. If not met, the bin will result empty. Defaults to 3", required=False, default=3)
parser.add_argument("--plotHits", action="store_true", help="Enable plotting of propHits and recHits on top of the efficiency. Defaults to false.", required=False, default=False)
args = parser.parse_args()


unpackedBranches = ["mu_propagated_station","mu_propagated_region","mu_propagated_layer","residual_phi","mu_propagatedGlb_x","mu_propagatedGlb_y"]
uproot_batch_size = "3000 MB" #either a string with byte specifier or number of entries as int
stepSizeX = args.binSizeX
stepSizeY = args.binSizeY
NpropHitsCutOff = args.NhitsCutOff
plotHits = args.plotHits
if __name__ == '__main__':
    output_folder_path = Path(OUTPUT_PATH, args.folder_name)
    prefix = args.outputprefix


    output_folder_path.mkdir(parents=True, exist_ok=True)
    print(OUTPUT_PATH)
    if PHPINDEX_FILE is not None: copy(PHPINDEX_FILE, output_folder_path)

    ## Binning&Plotting ranges
    ## For GE21 depend on the installed chambers
    ## Plotting the whole wheel would be distracting
    #station,region,layer,x_min,x_max,y_min,y_max
    ranges = [
        [1,1,1,-255,255,-255,255],   ## GE11 P Ly1
        [1,1,2,-255,255,-255,255],   ## GE11 P Ly2
        [1,-1,1,-255,255,-255,255],  ## GE11 M Ly1
        [1,-1,2,-255,255,-255,255],  ## GE11 M Ly2
        [2,-1,1,50,320,-300, -5],    ## GE21 M Ly1
        [2,-1,2,50,320,-300, -5],    ## GE21 M Ly2
        [2,1,1,50,230,-300, -80],    ## GE21 P Ly1
        [2,1,2,50,230,-300, -80],    ## GE21 P Ly2
    ]
    df_ranges = pd.DataFrame(ranges, columns = ["station","region","layer","x_min","x_max","y_min","y_max"])

    all_wheels = [(s, r, l) for s in [1,2] for r in [-1,1] for l in [1,2]]

    # The files will be processed iteratively
    # The histograms have to be summed at the end
    # Here a structured numpy array comes in handy
    # Structured data type
    dtype = [('Station', 'int'), ('Region', 'int'), ('Layer', 'int'), ("histograms", "object")]
    # Create a structured array
    histsProphit = np.zeros(len(all_wheels), dtype=dtype)
    histsMatchedhit = np.zeros(len(all_wheels), dtype=dtype)
    # initialize the structured array
    for i, (s,r,l) in enumerate(all_wheels):
        histsProphit[i] =(s, r, l, [])
        histsMatchedhit[i] =(s, r, l, [])


    ## loop through datafiles
    for df_hits in uproot.iterate( [k+":tree" for k in args.inputs], filter_name=unpackedBranches, step_size=uproot_batch_size, library="pd"):
        df_hits.rename(columns={"mu_propagatedGlb_x":"glb_x","mu_propagatedGlb_y":"glb_y"},inplace=True)
        
        for station, region, layer in all_wheels:
            range_selection = (df_ranges["station"]==station) & (df_ranges["region"]==region) & (df_ranges["layer"]==layer)
            _, _, _, x_min, x_max, y_min,y_max = tuple(df_ranges[range_selection].iloc[0])
            nBinsX = int( 10*(x_max - x_min)/stepSizeX ) # 10 converts the stepsize from mm to cm
            nBinsY = int( 10*(y_max - y_min)/stepSizeY ) # 10 converts the stepsize from mm to cm
            
            df = df_hits[ (df_hits["mu_propagated_station"] == station) & (df_hits["mu_propagated_region"] == region)  & (df_hits["mu_propagated_layer"] == layer)]
            matched_df = df[ ~df["residual_phi"].isna() ]
            
            prophits, edx, edy = np.histogram2d(
                df["glb_x"].to_numpy(),
                df["glb_y"].to_numpy(),
                bins=(nBinsX,nBinsY),
                range=[[x_min, x_max], [y_min, y_max]],
            )
            matchedhits, edx, edy = np.histogram2d(
                matched_df["glb_x"].to_numpy(),
                matched_df["glb_y"].to_numpy(),
                bins=(nBinsX,nBinsY),
                range=[[x_min, x_max], [y_min, y_max]],
            )
            histsProphit[(histsProphit["Station"]==station) & (histsProphit["Region"]==region) & (histsProphit["Layer"]==layer)]["histograms"][0].append(prophits)
            histsMatchedhit[(histsMatchedhit["Station"]==station) & (histsMatchedhit["Region"]==region) & (histsMatchedhit["Layer"]==layer)]["histograms"][0].append(matchedhits)


    ## merge histos and spit out plots
    for s,r,l in all_wheels:
        title = f"GE{'+' if r>0 else '-'}{s}1_Ly{l}"

        pHist = sum(histsProphit[(histsProphit["Station"]==s) & (histsProphit["Region"]==r) & (histsProphit["Layer"]==l)]["histograms"][0])
        mHist = sum(histsMatchedhit[(histsProphit["Station"]==s) & (histsMatchedhit["Region"]==r) & (histsMatchedhit["Layer"]==l)]["histograms"][0])
        efficiency = np.ones_like(pHist)*-1
        np.divide(mHist,pHist, out=efficiency, where=pHist>=NpropHitsCutOff)

        range_selection = (df_ranges["station"]==s) & (df_ranges["region"]==r) & (df_ranges["layer"]==l)
        _, _, _, x_min, x_max, y_min,y_max = tuple(df_ranges[range_selection].iloc[0])

        targets = {"propHits":pHist,"matchedHits":mHist, "efficiency":efficiency} if plotHits else {"efficiency":efficiency}
        for name,hist in targets.items():
            fig,ax = plt.subplots(figsize=(10,10), dpi=300)
            im = ax.imshow(
                hist.T,
                extent=[x_min, x_max, y_min, y_max],
                aspect='equal',
                interpolation="nearest",
                origin="lower",
                cmap=newcmp,
                vmin = 0. if name == "efficiency" else 1, ## efficiency(hits) plot is white if <NpropHitsCutOff (<1) prophits in bin
                vmax = 1. if name == "efficiency" else None
            )
    
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.1, 0.04, 0.8])
            fig.colorbar(im, cax=cbar_ax, label=name)
            ax.set_title(title)
            ax.set_xlabel("Glb x")
            ax.set_ylabel("Glb y")
            ax.text(0.78, 0.93, f'X bin size = {int(stepSizeX)}mm\nY bin size = {int(stepSizeY)}mm\n>={NpropHitsCutOff} prophits per bin',fontsize=11, transform=ax.transAxes)
            fig.savefig(Path(output_folder_path,f"{name}_{prefix}{title}.pdf"),dpi=300, facecolor=fig.get_facecolor())
            fig.savefig(Path(output_folder_path,f"{name}_{prefix}{title}.png"),dpi=300, facecolor=fig.get_facecolor())
            plt.close(fig)
            
        print(f"processed {title}\tAVG Nmatched/Nprop = {round(mHist.mean(),2)}/{round(pHist.mean(),2)}")
    
    
