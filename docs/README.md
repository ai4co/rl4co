# RL4CO Documentation

We use [MkDocs](https://www.mkdocs.org/) to generate the documentation with the [MkDocs Material theme](https://squidfunk.github.io/mkdocs-material/).

## Development

From root directory - install RL4CO locally:

```bash
pip install -e ".[dev,graph,routing]"
```

Then, install the dependencies:

```bash
pip install -r docs/requirements.txt
```

To build the documentation, run:

```bash
mkdocs serve
```