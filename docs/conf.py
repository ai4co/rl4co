# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import rl4co

project = "rl4co"
copyright = "Federico Berto, Chuanbo Hua, Junyoung Park"
author = "Federico Berto, Chuanbo Hua, Junyoung Park, Minsu Kim, Hyeonah Kim, Jiwoo Son, Haeyeon Kim, Joungho Kim, Jinkyoo Park"


release = rl4co.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# Most taken from: https://github.com/Lightning-AI/lightning/blob/master/docs/source-pytorch/conf.py
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_toolbox.collapse",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.video",
    "sphinxcontrib.katex",
    "myst_parser",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "sphinx_togglebutton",
    "sphinxcontrib.collections",
]


# Include the folder from the main repo containing the notebooks
collections = {
    "my_files": {
        "driver": "copy_folder",
        "source": "../notebooks/",
        "target": "",
        "ignore": ["*.ckpt"],
    }
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     "torch": ("https://pytorch.org/docs/stable/", None),
#     "torchmetrics": ("https://torchmetrics.readthedocs.io/en/stable/", None),
#     "tensordict": ("https://pytorch-labs.github.io/tensordict/", None),
#     "torchrl": ("https://pytorch.org/rl/", None),
#     "torchaudio": ("https://pytorch.org/audio/stable/", None),
#     "torchtext": ("https://pytorch.org/text/stable/", None),
#     "torchvision": ("https://pytorch.org/vision/stable/", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
# }


autosummary_generate = True

autodoc_member_order = "groupwise"

autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}

# Sphinx will add “permalinks” for each heading and description environment as paragraph signs that
#  become visible when the mouse hovers over them.
# This value determines the text for the permalink; it defaults to "¶". Set it to None or the empty
#  string to disable permalinks.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_add_permalinks
html_permalinks = True
html_permalinks_icon = "¶"

# True to prefix each section label with the name of the document it is in, followed by a colon.
#  For example, index:Introduction for a section called Introduction that appears in document index.rst.
#  Useful for avoiding ambiguity when the same section heading appears in different documents.
# http://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
autosectionlabel_prefix_document = True
