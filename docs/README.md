# RL4CO Documentation

We use [MkDocs](https://www.mkdocs.org/) to generate the documentation with the [MkDocs Material theme](https://squidfunk.github.io/mkdocs-material/).

## Development

From the root directory:

1. Install RL4CO locally

```bash
pip install -e ".[dev,graph,routing,docs]"
```

note that `docs` is the extra requirement for the documentation.


2. To build the documentation, run:

```bash
mkdocs serve
```

### Hooks

We are using the [hooks.py](hooks.py) for additional modifications. MkDocs for instance cannot detect files that are not in the same directory as an `__init__.py` (as described [here](https://stackoverflow.com/questions/75232397/mkdocs-unable-to-find-modules)) so we are automatically creating and deleting such files with our script
