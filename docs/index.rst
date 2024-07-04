.. METALoci documentation master file, created by
   sphinx-quickstart on Tue Jul  2 15:11:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to METALoci's documentation!
====================================

*Spatially auto-correlated signals in 3D genomes.*

====================================

METALoci relies on spatial autocorrelation analysis, classically employed in geostatistics, to describe how the 
variation of a variable depends on space at a global and local scales (e.g., identifying contamination hotspots 
within a city). METALoci repurposes this type of analysis to quantify spatial genome hubs of similar epigenetic 
properties. Briefly, the overall flowchart of METALoci consists of four steps:

* First, a genome-wide Hi-C normalized matrix is taken as input and the top interactions selected.

* Second, the selected interactions are used to build a graph layout (equivalent to a physical map) using the ``Kamada-Kawai algorithm`` with nodes representing bins in the Hi-C matrix and the 2D distance between the nodes being inversely proportional to their normalized Hi-C interaction frequency.

* Third, epigenetic/genomic signals, measured as coverage per genomic bin (e.g., ChIP-Seq signal for H3K27ac), are next mapped into the nodes of the graph layout.

* The fourth and final step involves the use of a measure of autocorrelation (specifically, the Local Moran’s I or ``LMI``) to identify nodes and their neighborhoods with an enrichment of similar epigenetic/genomic signals.

METALoci is compatible with ``.cool``, ``.mcool`` and ``.hic`` Hi-C formats; and with ``.bed`` signal files. The signal 
used in METALoci may be any numerical signal (as long as it is in a .bed file, with the location of such signal).

METALoci is meant to be use in a command line interface (CLI). Several scripts are available to run METALoci in a
step-by-step fashion. Refer to the :ref:`tutorial` for more information about how to use it.

The ``METALoci Python API`` is also available for more advanced users.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   tutorial
   cli_usage
   METALoci Python API <api>


   


