import sys
from unittest.mock import MagicMock

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "libcll"
copyright = "2024, Nai Xuan Ye"
author = "Nai Xuan Ye"
release = "1.0.0"
version = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Napoleon settings

# MOCK_MODULES = [
#     'libcll'
#     'libcll.datasets.CLBaseDataset',
#     'libcll.strategies.Strategy',
# ]

# sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
]


templates_path = ["_templates"]
# How to represents typehints
autodoc_typehints = "signature"

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

html_theme_options = {"collapse_navigation": False}

# -- Options for EPUB output
epub_show_urls = "footnote"

numpydoc_show_class_members = False
