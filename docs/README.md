# 📑 RL4CO Docs

We are using [Sphinx](https://www.sphinx-doc.org/en/master/) with [Napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) to build the documentation.
Moreover, we set [Google style](https://google.github.io/styleguide/pyguide.html) to follow with type convention.

- [Napoleon formatting with Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [ReStructured Text (reST)](https://docs.pylonsproject.org/projects/docs-style-guide/)
- [Paragraph-level markup](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#paragraphs)

See following short example of a sample function taking one position string and optional

```python
from typing import Optional


def my_func(param_a: int, param_b: Optional[float] = None) -> str:
    """Sample function.

    Args:
        param_a: first parameter
        param_b: second parameter

    Return:
        sum of both numbers

    Example::

        >>> my_func(1, 2)
        3

    Note:
        If you want to add something.
    """
    p = param_b if param_b else 0
    return str(param_a + p)
```

## 🗂️ File structures

```
.
├── _build/ - output website, only for local building
├── _content/ - content for docs pages in Markdown format 
├── _theme/
│   └── rl4co/ - website theme files
├── conf.py - main config file for building
├── index.md - content for index page
├── make.bat - building script for Windows
├── Makefile - building script for Unix
├── README.md
└── requirements.txt - requirement python packages for Read the Docs building
```

##  ⚙️ Build docs locally

**Step 1**. Install requirement packages (from root folder): `pip install -r docs/requirement.txt`;

**Step 2**. Run the building script:

- **Windows**: run `make.bat`;
- **Linux/macOS**: run `make html`;

The generated docs will be under the `_build` folder. You can open `docs/build/html/index.html` in your browser to check the docs.

We need to have LaTeX installed for rendering math equations. You can for example install TeXLive with the necessary extras by doing one of the following:

- **Windows/macOS**: check the [Tex Live install guide](https://www.tug.org/texlive/windows.html) for Windows/macOS;
- **Ubuntu (Linux)**: run `sudo apt-get update && sudo apt-get install -y texlive-latex-extra dvipng texlive-pictures`;
- Use the [RTD docker image](https://hub.docker.com/r/readthedocs/build);

## ⚙️ Build in Read the Docs

In the root of this repository, there is `.readthedocs.yaml` which will be loaded by the Read the Docs to build the docs.   Please refer to the [configuration file v2 guide from Read the Docs](https://docs.readthedocs.io/en/stable/config-file/v2.html) for details information of variables.

## 💡 Notes for contents
<details>
<summary>Markdown and RST support</summary>
RST is originally supported by the Sphinx. With the extension `myst_parser` it can support Markdown contents. Follow [this guide](https://www.sphinx-doc.org/en/master/usage/markdown.html) to learn more. 

In the meantime, we can still use RST within  Markdown files by 
````
```{eval-rst}
RST CONTENTS
```
````
</details>
<details>
<summary>Jupyter notebook support</summary>
With the extension `nbsphinx`, Sphinx can support Jupyter notebook. Follow [this guide](https://docs.readthedocs.io/en/stable/guides/jupyter.html) to learn more.

Indexing a Jupyter notebook is the same with a Markdown file in RST:
```
.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   _content/start/installation
   _content/start/quickstart_notebook
```
</details>
<details>
<summary>API docs auto generator</summary>
With Sphinx's `automodule` we can easily get the API docs:
```
.. automodule:: rl4co.data.generate_data
   :members:
   :undoc-members:
```
When deploy in Read the Docs, make sure putting the package to `requirement.txt` mentioned before. 
</details>

## 📚 References

We base the above guide on the official [PyTorch Lightning Docs](https://github.com/Lightning-AI/lightning/tree/master/docs).