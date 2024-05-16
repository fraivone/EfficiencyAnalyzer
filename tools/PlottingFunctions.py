import awkward as ak
import numpy as np
import numba
from numba import jit, int32, int64, float64
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from dataclasses import dataclass
from typing import List
import pandas as pd


GE21PlottableChambers = [ #(Station,region,chamber,layer)
                          [2,1,16,1],  #DEMONSTRATOR
                          [2,-1,16,1],
                          [2,-1,18,1]
]


@jit(float64[:](float64[:], int64), cache=True, nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins + 1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@jit(int32(float64, float64[:]), cache=True, nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]
    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1  # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0:  ## hist underflow
        return 0
    elif bin >= n:  ## hist overflow
        return n - 1
    else:
        return bin

@jit(cache=True, nopython=True)
def Fill_Histo_Generic(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges

@jit(cache=True, nopython=True)
def Fill_Histo_Residuals(compatible_hits_array, hist_range, nbins):
    hist_data = np.zeros((1, 2, 36, 2, nbins), dtype=np.uint16)
    bin_edges = get_bin_edges(hist_range, nbins)

    for evt_idx in range(len(compatible_hits_array)):
        for hit_idx in range(len(compatible_hits_array[evt_idx].gemRecHit_region)):
            st = compatible_hits_array[evt_idx].gemRecHit_station[hit_idx]
            re = (compatible_hits_array[evt_idx].gemRecHit_region[hit_idx] + 1) // 2  ## transforming -1 -> 0, 1 -> 1 to respect the indexing
            ch = compatible_hits_array[evt_idx].gemRecHit_chamber[hit_idx]
            la = compatible_hits_array[evt_idx].gemRecHit_layer[hit_idx]
            residual = compatible_hits_array[evt_idx].residual_rdphi[hit_idx]
            current_bin = compute_bin(residual, bin_edges)
            hist_data[st - 1, re, ch - 1, la - 1, current_bin] += 1
            # if current_bin is not None:
            #     print(type(current_bin))
            #     print(type(la))
            #     hist_data[(0,0,0,0,0)] += 1
    return hist_data, bin_edges


def Store_Binned_Residuals(histogram_data, bin_edges, output_folder,output_name_prefix="",enable_plot=False):
    RMS = {}
    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = round(bin_edges[1] - bin_edges[0], 4)
    fig, ax = plt.subplots(1, figsize=(10, 10), layout="constrained")
    for st in [0]:
        for region in [0, 1]:
            for chamber in range(36):
                for layer in range(2):
                    name = f"GE11-{'P' if region==1 else 'M'}-{chamber+1:02d}L{layer+1}"                        
                    entries = sum(histogram_data[st,region,chamber,layer,:])
                    if entries != 0:
                        avg = sum([x[i] * histogram_data[st, region, chamber, layer, i] for i in range(len(x))]) / entries
                        rms = np.sqrt(sum([ (x[i]**2) * histogram_data[st, region, chamber, layer, i] for i in range(len(x))]) / entries)
                    else:
                        avg = 0
                        rms = 0
                    RMS[name] = rms
                    if enable_plot:
                        ax.hist(x,weights=histogram_data[st, region, chamber, layer, :],bins=bin_edges,alpha=0.5,color="g")
                        ax.set_title(f"{name} residuals", fontweight="bold", size=24)
                        ax.set_xlabel(r"R$\Delta$$\phi$ (cm)", loc="right", size=20)
                        ax.set_ylabel(f"Entries/{bin_width}cm", loc="center", size=20)
                        ax.text(0.2,0.95,f"Entries: {entries}\nAVG {avg:.2f}\nRMS {rms:.2f}",transform=ax.transAxes,fontsize=20,fontweight="bold",va="top")
                        ax.grid()
                        fig.savefig(f"{output_folder}/{output_name_prefix}{name}.png")
                        # print(f"Plotted chamber {name}")
                        ax.cla()
                    np.savetxt(f"{output_folder}/{output_name_prefix}{name}_hist.txt", histogram_data[st, region, chamber, layer, :], delimiter=",")
    df = pd.DataFrame(RMS,index=[0])
    df.to_csv(f"{output_folder}/{output_name_prefix}ResidualsRMS.txt", index=False)
    
    return ax



@dataclass
class ArrayOfRecords_HistogramBins:
    x_low_lim: float
    x_high_lim: float
    y_low_lim: float
    y_high_lim: float
    fields: List[str]
    ArrayOfRecords = None
    _bins = None
    _range = None
    _base_array = None

    def __post_init__(self):
        self._base_array =  np.zeros(( 1, # stations
                                       2, # regions
                                       2, # layers
                                       int(self.x_high_lim - self.x_low_lim),  # UDF (usually x axis of the final plot)
                                       int(self.y_high_lim - self.y_low_lim)), # UDF (usually y axis of the final plot)
                                       dtype=np.int32)
        ## Array of records has as fields self.fields, as array _base_array
        self.ArrayOfRecords = ak.zip({k: v for k, v in zip(self.fields, [self._base_array for k in self.fields])})
        self._bins = (int(self.x_high_lim - self.x_low_lim), int(self.y_high_lim-self.y_low_lim))
        self._range = np.array([(self.x_low_lim, self.x_high_lim), (self.y_low_lim, self.y_high_lim)])

    def AddEntriesFromBinCounts(self,field: str,array: np.ndarray):
        if array.shape == self.ArrayOfRecords[field].to_numpy().shape:
            sum_result = np.add(self.ArrayOfRecords[field].to_numpy(), array.astype(np.int32), casting="unsafe")
            self.ArrayOfRecords[field] = sum_result
        else:
            raise ValueError(f"Parsed array has shape {array.shape} non-addable to {self.__repr__()} of shape {self.ArrayOfRecords[field].to_numpy().shape} ")

    def plot(self,field,x_ticks,y_ticks,xaxis_label,yaxis_label,color_map,ax,normalization_factor=None):
        station = 1 ## GE11 only
        for idx, (region,layer) in enumerate([(-1,1),(1,1),(-1,2),(1,2)]):
            plot_content = self.ArrayOfRecords[field][station - 1, (region+1)//2, layer -1]
            title = f"GE{'+' if region>0 else '-'}{station}1 Ly{layer} {field}"
            ax[idx].set_title(title, fontweight="bold", size=24)
            ax[idx].set_xlabel(xaxis_label, loc="right", size=20)
            ax[idx].set_ylabel(yaxis_label, loc="center", size=20)
            ax[idx].set_xticks(x_ticks)
            ax[idx].set_yticks(y_ticks)
            if normalization_factor is not None:
                plot_content = plot_content*normalization_factor
            im = ax[idx].imshow(( plot_content.to_numpy()).T, 
                         origin='lower', 
                         aspect="auto",
                         interpolation="none",
                         extent=[self.x_low_lim, self.x_high_lim, self.y_low_lim, self.y_high_lim], 
                                cmap=color_map,vmin = ak.min(plot_content), vmax = ak.max(plot_content))  
            ax[idx].grid()
        return im

    def __getitem__(self, key):
        return self.ArrayOfRecords[key]

def axs_36chambersEff_style(axs):
    axs.legend(loc="lower left", prop={"size": 20})
    axs.set_xlabel("Chamber", loc="right", size=20)
    axs.set_ylabel("Muon Efficiency", loc="center", size=20)
    # Major ticks
    axs.set_xticks(np.arange(1, 37, 1))
    axs.set_yticks(np.arange(0, 1.2, 0.2))
    # Labels for major ticks
    axs.set_xticklabels(np.arange(1, 37, 1), size=15)
    axs.set_yticklabels(np.arange(0, 1.2, 0.2), size=15)
    axs.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    # Minor ticks
    axs.set_xticks(np.arange(0.5, 37.5, 1), minor=True)
    # Gridlines based on minor ticks
    axs.grid(which="major")
    return axs

def axs_8EtasEff_style(axs):
    axs.legend(loc="lower left", prop={"size": 20})
    axs.set_xlabel("EtaPartition", loc="right", size=20)
    axs.set_ylabel("Muon Efficiency", loc="center", size=20)
    # Major ticks
    axs.set_xticks(np.arange(1, 9, 1))
    axs.set_yticks(np.arange(0, 1.2, 0.2))
    # Labels for major ticks
    axs.set_xticklabels(np.arange(1, 9, 1), size=15)
    axs.set_yticklabels(np.arange(0, 1.2, 0.2), size=15)
    axs.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    # Minor ticks
    axs.set_xticks(np.arange(0.5, 9.5, 1), minor=True)
    # Gridlines based on minor ticks
    axs.grid(which="major")
    return axs



def axs_4GE21modulesEff_style(axs):
    axs.legend(loc="best", prop={"size": 20})
    axs.set_xlabel("Module", loc="right", size=20)
    axs.set_ylabel("Muon Efficiency", loc="center", size=20)
    # Major ticks
    axs.set_xticks(np.arange(1, 5, 1))
    axs.set_yticks(np.arange(0, 1.2, 0.2))
    # Labels for major ticks
    axs.set_xticklabels(np.arange(1, 5, 1), size=15)
    axs.set_yticklabels(np.arange(0, 1.2, 0.2), size=15)
    axs.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    # Minor ticks
    axs.set_xticks(np.arange(0.5, 5.5, 1), minor=True)
    # Gridlines based on minor ticks
    axs.grid(which="major")
    return axs


def axs_8etasEff_style(axs):
    axs.legend(loc="lower left", prop={"size": 20})
    axs.set_xlabel("etaPartition", loc="right", size=20)
    axs.set_ylabel("Muon Efficiency", loc="center", size=20)
    # Major ticks
    axs.set_xticks(np.arange(1, 9, 1))
    axs.set_yticks(np.arange(0.6, 1.2, 0.2))
    # Labels for major ticks
    axs.set_xticklabels(np.arange(1, 9, 1), size=15)
    axs.set_yticklabels(np.arange(0.6, 1.2, 0.2), size=15)
    axs.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    # Minor ticks
    axs.set_xticks(np.arange(0.5, 9.5, 1), minor=True)
    # Gridlines based on minor ticks
    axs.grid(which="major")
    return axs


@numba.jit(cache=True, nopython=True)
def unpackVFATStatus_toBin(gemOHStatus,hist_bins):
    VFATMasked = hist_bins.copy()
    VFATMissing = hist_bins.copy()
    VFATZSd = hist_bins.copy()
    for evt_idx in range(len(gemOHStatus)):
        for status_idx in range(len(gemOHStatus[evt_idx].gemOHStatus_chamber)):
            st = gemOHStatus[evt_idx].gemOHStatus_station[status_idx]
            re = (gemOHStatus[evt_idx].gemOHStatus_region[status_idx] + 1) // 2  ## transforming -1 -> 0, 1 -> 1 to respect the indexing
            ch = gemOHStatus[evt_idx].gemOHStatus_chamber[status_idx]
            layer = gemOHStatus[evt_idx].gemOHStatus_layer[status_idx]
            for vfat in range(24):
                VFATMasked[(st - 1, re, layer - 1, ch - 1, vfat)] += np.logical_not((gemOHStatus[evt_idx].gemOHStatus_VFATMasked[status_idx]>>vfat) & 0b1)
                VFATMissing[(st - 1, re, layer - 1, ch - 1, vfat)] += (gemOHStatus[evt_idx].gemOHStatus_VFATMissing[status_idx]>>vfat) & 0b1
                VFATZSd[(st - 1, re, layer - 1, ch - 1, vfat)] += (gemOHStatus[evt_idx].gemOHStatus_VFATZS[status_idx]>>vfat) & 0b1
    return VFATMasked,VFATMissing,VFATZSd

@numba.jit(cache=True, nopython=True)
def OHStatus_toBin(gemOHStatus,hist_bins):
    OHHasStatus = hist_bins.copy()
    OHErrors = hist_bins.copy()
    OHWarnings = hist_bins.copy()
    for evt_idx in range(len(gemOHStatus)):
        for status_idx in range(len(gemOHStatus[evt_idx].gemOHStatus_chamber)):
            st = gemOHStatus[evt_idx].gemOHStatus_station[status_idx]
            re = (gemOHStatus[evt_idx].gemOHStatus_region[status_idx] + 1) // 2  ## transforming -1 -> 0, 1 -> 1 to respect the indexing
            ch = gemOHStatus[evt_idx].gemOHStatus_chamber[status_idx]
            layer = gemOHStatus[evt_idx].gemOHStatus_layer[status_idx]
            lumiblock = gemOHStatus[evt_idx].gemOHStatus_lumiblock[status_idx]
            
            OHHasStatus[(st - 1, re, layer - 1, lumiblock, ch-1)] += 1
            if gemOHStatus[evt_idx].gemOHStatus_errors[status_idx] > 0:
                OHErrors[(st - 1, re, layer - 1, lumiblock, ch-1)] += 1
            if gemOHStatus[evt_idx].gemOHStatus_warnings[status_idx] > 0:
                OHWarnings[(st - 1, re, layer - 1, lumiblock, ch-1)] += 1

    return OHHasStatus,OHErrors,OHWarnings


if __name__=='__main__':
    k = ArrayOfRecords_HistogramBins(0,20,0,30, ["A","B"])
    print(k.ArrayOfRecords["A"].type)
