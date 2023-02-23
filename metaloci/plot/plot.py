import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np


def plot_kk(mlo):

    plt.figure(figsize=(10, 10))
    options = {"node_size": 50, "edge_color": "black", "linewidths": 0.1, "width": 0.05}

    nx.draw(
        mlo.kk_graph,
        mlo.kk_nodes,
        node_color=range(len(mlo.kk_nodes)),
        cmap=plt.cm.coolwarm,
        **options,
    )

    if mlo.midpoint is not None:

        plt.scatter(
            mlo.kk_nodes[mlo.midpoint - 1][0],
            mlo.kk_nodes[mlo.midpoint - 1][1],
            s=80,
            facecolors="none",
            edgecolors="r",
        )

    xs = []
    ys = []

    for _, val in mlo.kk_nodes.items():

        xs.append(val[0])
        ys.append(val[1])

    sns.lineplot(x=xs, y=ys, sort=False, lw=2, color="black", legend=False, zorder=1)

    return plt


def mixed_matrices_plot(mlobject):

    if mlobject.flat_matrix is None:

        print("Flat matrix not found in metaloci object. Run get_subset_matrix() first.")
        exit()

    fig_matrix, (ax1, _) = plt.subplots(1, 2, figsize=(20, 5))

    # Plot of the mixed matrix (top triangle is the original matrix,
    # lower triange is the subsetted matrix)
    ax1.imshow(
        mlobject.mixed_matrices,
        cmap="YlOrRd",
        vmax=np.nanquantile(mlobject.flat_matrix[mlobject.kk_top_indexes], 0.99),
    )
    ax1.patch.set_facecolor("black")

    # Density plot of the subsetted matrix
    sns.histplot(
        data=mlobject.flat_matrix[mlobject.kk_top_indexes].flatten(),
        stat="density",
        alpha=0.4,
        kde=True,
        legend=False,
        kde_kws={"cut": 3},
        **{"linewidth": 0},
    )

    fig_matrix.tight_layout()
    fig_matrix.suptitle(f"Matrix for {mlobject.region} (cutoff: {mlobject.kk_cutoff})")

    return mlobject, fig_matrix
