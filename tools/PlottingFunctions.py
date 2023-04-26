import awkward as ak
import numpy as np
from numba import jit, int32, int64, float64
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


@jit(float64[:](int64[:], int64), cache=True, nopython=True)
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


def Plot_Binned_Residuals(histogram_data, bin_edges, output_folder):
    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = round(bin_edges[1] - bin_edges[0], 2)
    fig, ax = plt.subplots(1, figsize=(10, 10), layout="constrained")
    for st in [0]:
        for region in [0, 1]:
            for chamber in range(36):
                for layer in range(2):
                    name = f"GE11-{'P' if region==1 else 'M'}-{chamber+1:02d}L{layer+1}"
                    ax.hist(x,weights=histogram_data[st, region, chamber, layer, :],bins=bin_edges,alpha=0.5,color="g")
                    avg = sum([x[i] * histogram_data[st, region, chamber, layer, i] for i in range(len(x))]) / len(x)
                    ax.set_title(f"{name} residuals", fontweight="bold", size=24)
                    ax.set_xlabel(r"Residual R$\Delta$$\phi$ (cm)", loc="right", size=20)
                    ax.set_ylabel(f"Entries/{bin_width}cm", loc="center", size=20)
                    ax.text(0.2,0.95,f"Entries: {sum(histogram_data[st,region,chamber,layer,:])}\nAVG {avg:.2f}\nAVG",transform=ax.transAxes,fontsize=20,fontweight="bold",va="top")
                    ax.grid()
                    fig.savefig(f"{output_folder}/{name}.png")
                    print(f"Plotted chamber {name}")
                    ax.cla()
    return ax


def plotArray_2D(
    array,
    mask,
    x_field,
    y_field,
    x_lim,
    y_lim,
    x_ticks,
    y_ticks,
    title,
    xaxis_label,
    yaxis_label,
    color_map,
    ax,
    normalization_factor=None,
):
    if mask is None:
        plotting_x = ak.flatten(array[f"{x_field}"]).to_numpy()
        plotting_y = ak.flatten(array[f"{y_field}"]).to_numpy()
    else:
        plotting_x = ak.flatten(array[f"{x_field}"][mask]).to_numpy()
        plotting_y = ak.flatten(array[f"{y_field}"][mask]).to_numpy()

    ax.set_title(title, fontweight="bold", size=24)
    ax.set_xlabel(xaxis_label, loc="right", size=20)
    ax.set_ylabel(yaxis_label, loc="center", size=20)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    H, xedges, yedges, im = ax.hist2d(plotting_x,plotting_y,bins=(int(x_lim[-1] - x_lim[0]), int(y_lim[-1] - y_lim[0])),cmap=color_map,range=np.array([(x_lim[0], x_lim[-1]), (y_lim[0], y_lim[-1])]))
    if normalization_factor is not None:
        H_normalized = H / normalization_factor  # the max value of the histogrm is 1
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(H_normalized,extent=extent,cmap=color_map,interpolation="none",origin="lower")
    ax.grid()
    return im


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
