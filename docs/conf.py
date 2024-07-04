# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import sphinx_rtd_theme

# Add the docs directory to sys.path
# sys.path.insert(0, os.path.abspath('.'))

# Dynamically compute the path to your project's root directory
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the 'src' directory to sys.path to make the modules discoverable by Sphinx
# sys.path.insert(0, os.path.join(project_root, 'src'))

sys.path.insert(0, os.path.abspath('..'))

project = 'METALoci'
copyright = '2024'
author = 'Iago Maceda, Marc A. Marti-Renom, Leo Zuber'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinxarg.ext'
]

os.environ['SPHINX_BUILD'] = 'true'

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

def skip_member(app, what, name, obj, skip, options):
    # Modules to exclude entirely, including submodules
    excluded_modules = {"metaloci.tests"}
    # Specific functions to exclude
    excluded_functions = {"metaloci.misc.misc.check_diagonal",
                          "metaloci.misc.misc.remove_folder",
                          "metaloci.misc.misc.natural_sort",
                          "metaloci.plot.plot.get_x_axis_label_signal_plot",
                          "metaloci.plot.plot.get_color_alpha",
                          "metaloci.spatial_stats.lmi.coord_to_id",
                          "metaloci.tools.figure.populate_args",
                          "metaloci.tools.layout.populate_args",
                          "metaloci.tools.ml.populate_args",
                          "metaloci.tools.prep.populate_args",
                          "metaloci.utility_scripts.bts.populate_args",
                          "metaloci.utility_scripts.gene_selector.populate_args",
                          "metaloci.utility_scripts.sniffer.populate_args",
                          }

    # Check if the object's module is in the excluded list or is a submodule of an excluded module
    if hasattr(obj, '__module__'):

        module_name = obj.__module__
        # Check if the module or any of its parent modules should be excluded
        if any(module_name == mod or module_name.startswith(mod + '.') for mod in excluded_modules):

            return True
        # Check if the full name (module + object name) should be excluded
        full_name = f"{module_name}.{name}"

        if full_name in excluded_functions:

            return True

    # Additionally, check if the name alone matches any excluded functions
    if name in excluded_functions:

        return True

    # If none of the conditions match, follow the original decision
    return skip

def setup(app):

    app.connect("autodoc-skip-member", skip_member)