# RL4CO Docs

We are using Sphinx with Napoleon extension.
Moreover, we set Google style to follow with type convention.

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

## Building Docs

When updating the docs, make sure to build them first locally and visually inspect the html files in your browser for
formatting errors. In certain cases, a missing blank line or a wrong indent can lead to a broken layout.
Run these commands

```bash
git submodule update --init --recursive
make docs
```

and open `docs/build/html/index.html` in your browser.

When you send a PR the continuous integration will run tests and build the docs.

Notes:

- You need to have LaTeX installed for rendering math equations. You can for example install TeXLive with the necessary extras by doing one of the following:
  - on Ubuntu (Linux) run `sudo apt-get update && sudo apt-get install -y texlive-latex-extra dvipng texlive-pictures`
  - use the [RTD docker image](https://hub.docker.com/r/readthedocs/build)
- You need to have pandoc installed for rendering Jupyter Notebooks. On Ubuntu (Linux), you can run: `sudo apt-get install pandoc`


## References

We base the above guide on the official [PyTorch Lightning Docs](https://github.com/Lightning-AI/lightning/tree/master/docs).