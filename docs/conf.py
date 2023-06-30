# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys; sys.path.append('..') # ! add rl4co path 
sys.path.insert(0, '../rl4co') # ! add rl4co path
import torch

project = 'rl4co'
copyright = 'Federico Berto, Chuanbo Hua, Junyoung Park'
author = 'Federico Botu, Chuanbo Hua, Junyoung Park, Minsu Kim, Hyeonah Kim, Jiwoo Son, Haeyeon Kim, Joungho Kim, Jinkyoo Park'
release = 'v0.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser', # Markdown parser
    'sphinx_rtd_theme', # Read the Docs theme
    'sphinx.ext.autodoc', # Autodoc
    'nbsphinx', # Jupyter Notebookq support
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# SECTION: original theme set
# html_theme = 'alabaster'
# html_static_path = ['_static']

# SECTION: set sphinx_rtd_theme
# html_theme = "sphinx_rtd_theme"
# html_static_path = ['_static']
# html_theme_options = {
#     'navigation_depth': 4,
# }

# SECTION: set rl4co theme
html_theme = "rl4co"
html_theme_path = ["_theme"]

# SECTION: set different parser for rst and md
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
