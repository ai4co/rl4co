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

### Hooks

You may have noticed that we do not have an `index.md` file. This is because we are using [hooks.py](hooks.py) to copy the root `README.md` to `index.md`. We are also copying the examples folder and deleting it upon stopping the server.
