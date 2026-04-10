"""Sphinx configuration for ml-sgs-turbulence documentation."""

import sys
from pathlib import Path

# Add package to path so autodoc works without installation
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# -- Project information -------------------------------------------------------
project = "ml-sgs-turbulence"
copyright = "2026, Sambit Kumar Panda, Todd Jones, Muhammad Shahzad, Bryan Lawrence, Anna-Louise Ellis"
author = "Sambit Kumar Panda et al."
release = "1.0.0"

# -- General configuration -------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy / Google docstrings
    "sphinx.ext.viewcode",      # link to source
    "sphinx.ext.intersphinx",   # cross-reference NumPy, PyTorch docs
    "myst_parser",              # Markdown support
    "nbsphinx",                 # Jupyter notebooks
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- HTML output -------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}

# -- Intersphinx -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "torch":  ("https://pytorch.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}
