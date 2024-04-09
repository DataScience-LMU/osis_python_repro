"""Documentation configuration file for the project."""

import os
import sys

project = 'OSIS Reproducability Python Temlate'
copyright = '2024, Emanuel Sommer, Tobias Weber, Lisa Wimmer'
author = 'Emanuel Sommer, Tobias Weber, Lisa Wimmer'

sys.path.insert(0, os.path.abspath('.'))

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', '.github']

# sys.path.append(os.path.abspath('../src'))

html_theme = 'classic'
html_static_path = ['_static']

# autodoc
# autosummary_generate = True
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': True,
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}
# autodoc_mock_imports = ["src"]
