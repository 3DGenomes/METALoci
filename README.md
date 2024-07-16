
# METALoci

#### Spatially auto-correlated signals in 3D genomes.

METALoci relies on spatial autocorrelation analysis, classically employed in geostatistics, to describe how the variation of a variable depends on space at a global and local scales (e.g., identifying contamination hotspots within a city). METALoci repurposes this type of analysis to quantify spatial genome hubs of similar epigenetic properties. Briefly, the overall flowchart of METALoci consists of four steps:

* First, a genome-wide Hi-C normalized matrix is taken as input and the top interactions selected.

* Second, the selected interactions are used to build a graph layout (equivalent to a physical map) using the Kamada-Kawai algorithm with nodes representing bins in the Hi-C matrix and the 2D distance between the nodes being inversely proportional to their normalized Hi-C interaction frequency.

* Third, epigenetic/genomic signals, measured as coverage per genomic bin (e.g., ChIP-Seq signal for H3K27ac), are next mapped into the nodes of the graph layout.

* The fourth and final step involves the use of a measure of autocorrelation (specifically, the Local Moran’s I or LMI) to identify nodes and their neighborhoods with an enrichment of similar epigenetic/genomic signals.

METALoci is compatible with .cool, .mcool and .hic Hi-C formats; and with .bed signal files. The signal used in METALoci
may be any numerical signal (as long as it is in a .bed file, with the location of such signal).

#### Have a look at the [documentation](https://metaloci.readthedocs.io)!

## Install metaloci from PyPI:

```bash
conda create -n metaloci -c bioconda python==3.9 bedtools==2.31.1
conda activate metaloci
pip install metaloci
```

If you are experiencing any unexpected results with METALoci (e.g. your signal after binning is 0 for every bin), we 
suggest to update the version of **awk** you are using. The recommended version is 5.1.0 or newer.

In Ubuntu, you can do this with:

```bash
sudo apt install gawk
```

## Contributors

METALoci is currently being developed at the [MarciusLab](http://www.marciuslab.org) by Iago Maceda, 
Marc A. Marti-Renom and Leo Zuber, with the contribution of other members of the lab.

