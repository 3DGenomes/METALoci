.. _install:

Installation
============

*METALoci has only been tested in Ubuntu and Manjaro Linux distributions. MacOS support is not guaranteed.*

Install metaloci from PyPI:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda create -n metaloci -c bioconda python==3.12 bedtools
    conda activate metaloci
    pip install metaloci

Do you have trouble compiling dependencies? Perhaps you need an upgraded version of libcurl.

In Ubuntu, you can install it with:

.. code-block:: bash
    
    sudo apt install -y libcurl4-openssl-dev


If you are experiencing any unexpected results with METALoci (e.g. your signal after binning is 0 for every bin), we 
suggest to update the version of **awk** you are using. The recommended version is 5.1.0 or newer.

In Ubuntu, you can do this with:

.. code-block:: bash

    sudo apt install gawk



