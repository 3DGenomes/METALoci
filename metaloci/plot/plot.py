import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import os


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
