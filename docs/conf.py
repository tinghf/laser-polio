extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]
source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
project = "LASER Polio (PHASER)"
year = "2025"
author = "Institute for Disease Modeling"
copyright = f"{year}, Gates Foundation"
version = release = "0.2.30"

pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": ("https://github.com/InstituteforDiseaseModeling/laser-polio/issues/%s", "#%s"),
    "pr": ("https://github.com/InstituteforDiseaseModeling/laser-polio/pull/%s", "PR #%s"),
}

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    # "githuburl": "https://github.com/InstituteforDiseaseModeling/laser-polio/",
}

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_short_title = f"{project}-{version}"

# Napoleon settings (Napolean converts Google-style docstrings to reStructuredText)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True  # from Cookiecutter template, False is the default
napoleon_use_param = False  # from Cookiecutter template, True is the default
napoleon_use_rtype = False  # from Cookiecutter template, True is the default
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

mathjax3_config = {"TeX": {"Macros": {"small": ["{\\scriptstyle #1}", 1]}}}

# Prevent the following warning:
# sphinx/builders/linkcheck.py:86: RemovedInSphinx80Warning: The default value for 'linkcheck_report_timeouts_as_broken' will change to False in Sphinx 8, meaning that request timeouts will be reported with a new 'timeout' status, instead of as 'broken'. This is intended to provide more detail as to the failure mode. See https://github.com/sphinx-doc/sphinx/issues/11868 for details.
#   warnings.warn(deprecation_msg, RemovedInSphinx80Warning, stacklevel=1)
linkcheck_report_timeouts_as_broken = False

# Ignore the (not-yet-live) PyPI page during linkcheck
linkcheck_ignore = [
    r"https://pypi\.org/project/laser-polio/?",
]
