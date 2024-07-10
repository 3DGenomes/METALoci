.. _install:

Installation
============

*METALoci has only been tested in Ubuntu and Manjaro Linux distributions. MacOS support is not guaranteed.*

METALoci requires bedtools to be installed and accesible from the conda environment you will use. You can install it with:

.. code-block:: bash

   conda install bedtools

Install metaloci from PyPI:

.. code-block:: bash

    conda create -n metaloci python==3.9
    conda activate metaloci
    pip install metaloci

If you are experiencing any unexpected results with METALoci, we suggest to update the version of awk you are using. The recommended version is 5.1.0 or newer.

In Ubuntu, you can do this with:

.. code-block:: bash

    sudo apt install gawk



