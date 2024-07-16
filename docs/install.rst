.. _install:

Installation
============

*METALoci has only been tested in Ubuntu and Manjaro Linux distributions. MacOS support is not guaranteed.*

Install metaloci from PyPI:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda create -n metaloci -c bioconda python==3.9 bedtools==2.31.1
    conda activate metaloci
    pip install metaloci

If you are experiencing any unexpected results with METALoci (e.g. your signal after binning is 0 for every bin), we 
suggest to update the version of **awk** you are using. The recommended version is 5.1.0 or newer.

In Ubuntu, you can do this with:

.. code-block:: bash

    sudo apt install gawk



