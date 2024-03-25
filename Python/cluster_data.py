"""
    WEC Power AI/ML
    Copyright 2024 (C)

    Anthony Truelove MASc, P.Eng.
    email:  gears1763@tutanota.com
    github: gears1763-2

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    CONTINUED USE OF THIS SOFTWARE CONSTITUTES ACCEPTANCE OF THESE TERMS.
"""


"""
    A script for clustering the feature and power arrays for some initial insight into
    the structure of the data.
"""


import matplotlib.pyplot as plt

import numpy as np

import sklearn.cluster as skl_cl
import sklearn.decomposition as skl_dc
import sklearn.manifold as skl_mf
import sklearn.preprocessing as skl_pp


if __name__ == "__main__":
    #   load data
    input_clustering = np.load("data/feature_array.npy")
    target_clustering = np.load("data/power_array.npy")
    
    size = input_clustering.shape[0]
    dimension = input_clustering.shape[1]
    
    
    
    #   join data into single array, with the target (power) column log10 scaled,
    #   for clustering
    input_clustering = np.append(
        input_clustering,
        np.log10(target_clustering).reshape(-1, 1),
        axis=1
    )
    
    size = input_clustering.shape[0]
    dimension = input_clustering.shape[1]
    
    
    
    #   define column labels
    column_label_list = [
        "Water Depth [m]",
        "Significant Wave Height [m]",
        "Wave Peak Period [s]",
        "Float Diameter [m]",
        "Power Takeoff Stiffness [N/m]",
        r"$\log_{10}($Power Takeoff Damping [N.s/m]$)$",
        r"$\Pi_1 = \log_{10}\left(\overline{k}^2(D^2 - d^2)\right)$",
        r"$\Pi_2 = \log_{10}\left(\frac{b}{\rho g H_s^2T_p}\right)$",
        r"$\Pi_3 = \log_{10}\left(\frac{b}{2\sqrt{(k + k_D)m}}\right)$",
        r"$\Pi_4 = T_p\sqrt{\frac{k + k_D}{m}}$",
        r"$\log_{10}($Expected Power Output [kW]$)$"
    ]
    
    
    
    #   normalize inputs for clustering
    min_max_scaler = skl_pp.MinMaxScaler()
    min_max_scaler.fit(input_clustering)
    
    input_clustering_norm = min_max_scaler.transform(input_clustering)
    
    
    
    #   extract principal components for clustering
    PCA = skl_dc.PCA(n_components=dimension)
    PCA.fit(input_clustering_norm)
    
    input_clustering_PCA = PCA.transform(input_clustering_norm)
    
    
    
    #   get k-means clustering of data
    k = dimension
    
    k_means_clustering = skl_cl.KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=500,
        verbose=1,
        algorithm="elkan"
    )
    
    k_means_clustering.fit(input_clustering_PCA)
    
    
    
    #   assign data to clusters
    cluster_assignment = k_means_clustering.predict(input_clustering_PCA)
    idx_cluster_sort = np.argsort(cluster_assignment)
    
    
    
    #   make some k-means cluster plots
    for i in range(0, len(column_label_list)):
        for j in range(i + 1, len(column_label_list)):
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, zorder=1)
            plt.scatter(
                input_clustering[idx_cluster_sort, i],
                input_clustering[idx_cluster_sort, j],
                s=16,
                c=cluster_assignment[idx_cluster_sort],
                cmap="jet",
                zorder=2
            )
            plt.colorbar(
                label=r"Cluster (k-means, $k={}$) [ ]".format(k),
                ticks=[i for i in range(0, k)]
            )
            plt.xlabel(column_label_list[i])
            plt.ylabel(column_label_list[j])
            plt.savefig(
                "../LaTeX/images/mining/k-means_clustering_{}_{}.png".format(i, j),
                format="png",
                dpi=128,
                bbox_inches="tight"
            )
            
            plt.close()
    
    
    
    #   get t-SNE projection (2D) of data
    tSNE = skl_mf.TSNE(
        n_components=2,
        verbose=1
    )
    
    input_clustering_tSNE = tSNE.fit_transform(input_clustering_norm)
    
    
    
    #   make t-SNE plot (colouring by log10(power))
    power_sort_idx = np.argsort(input_clustering[:, dimension - 1])
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        input_clustering_tSNE[power_sort_idx, 0],
        input_clustering_tSNE[power_sort_idx, 1],
        s=16,
        c=input_clustering[power_sort_idx, dimension - 1],
        cmap="jet",
        zorder=2
    )
    plt.colorbar(
        label=column_label_list[dimension - 1]
    )
    plt.xlabel("t-SNE Component 0")
    plt.ylabel("t-SNE Component 1")
    plt.savefig(
        "../LaTeX/images/mining/t-SNE_clustering.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    plt.close()
