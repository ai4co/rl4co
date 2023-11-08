# Contributing to RL4CO

Thank you for your interest in contributing to our project. We welcome contributions from anyone and are grateful for even the smallest of fixes! Please read this document to learn how to contribute to RL4CO.

## Getting Started

To get started with contributing, please follow these steps:

1. Fork the repository and clone it to your local machine.
2. Install any necessary dependencies.
3. Create a new branch for your changes: `git checkout -b my-branch-name`.
4. Make your desired changes or additions.
5. Run the tests to ensure everything is working as expected with: `pytest tests`
6. Commit your changes: `git commit -m "Descriptive commit message"`.
7. Push to the branch: `git push origin my-branch-name`.
8. Submit a pull request to the `main` branch of the original repository.


## Code Style

Please make sure to follow the established code style guidelines for this project. Consistent code style helps maintain readability and makes it easier for others to contribute to the project.

To enforce this we use [`pre-commit`](https://pre-commit.com/) to run [`black`](https://black.readthedocs.io/en/stable/index.html) and [`ruff`](https://beta.ruff.rs/docs/) on every commit.

`pre-commit` is part of our requirements in the `pyproject.toml` file so you should already have it installed. If you don't, you can install the library via pip with:

```bash
$ pip install -e .

# And then install the `pre-commit` hooks with:

$ pre-commit install

# output:
pre-commit installed at .git/hooks/pre-commit
```

Or you could just run `make dev-install` to install the dependencies and the hooks.

If you are not familiar with the concept of [git hooks](https://git-scm.com/docs/githooks) and/or [`pre-commit`](https://pre-commit.com/) please read the documentation to understand how they work.

As an introduction of the actual workflow, here is an example of the process you will encounter when you make a commit:

Let's add a file we have modified with some errors, see how the pre-commit hooks run `black` and fails.
`black` is set to automatically fix the issues it finds:

```bash
$ git add rl4co/models/my_awesome_model.py
$ git commit -m "commit message"
black....................................................................Failed
- hook id: black
- files were modified by this hook

reformatted rl4co/models/my_awesome_model.py

All done! ✨ 🍰 ✨
1 file reformatted.
```

You can see that `rl4co/models/my_awesome_model.py` is both staged and not staged for commit. This is because `black` has formatted it and now it is different from the version you have in your working directory. To fix this you can simply run `git add rl4co/models/my_awesome_model.py` again and now you can commit your changes.

```bash
$ git status
On branch pre-commit-setup
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
    modified:   rl4co/models/my_awesome_model.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
    modified:   rl4co/models/my_awesome_model.py
```

Now let's add the file again to include the latest commits and see how `ruff` fails.

```bash
$ git add rl4co/models/my_awesome_model.py
$ git commit -m "commit message"
black....................................................................Passed
ruff.....................................................................Failed
- hook id: ruff
- exit code: 1
- files were modified by this hook

Found 2 errors (2 fixed, 0 remaining).
```

Same as before, you can see that `rl4co/models/my_awesome_model.py` is both staged and not staged for commit. This is because `ruff` has formatted it and now it is different from the version you have in your working directory. To fix this you can simply run `git add rl4co/models/my_awesome_model.py` again and now you can commit your changes.

```bash
$ git add rl4co/models/my_awesome_model.py
$ git commit -m "commit message"
black....................................................................Passed
ruff.....................................................................Passed
fix end of files.........................................................Passed
[pre-commit-setup f00c0ce] testing
 1 file changed, 1 insertion(+), 1 deletion(-)
```

Now your file has been committed and you can push your changes.

At the beginning this might seem like a tedious process (having to add the file again after `black` and `ruff` have modified it) but it is actually very useful. It allows you to see what changes `black` and `ruff` have made to your files and make sure that they are correct before you commit them.

## Issue Tracker

If you encounter any bugs, issues, or have feature requests, please [create a new issue](https://github.com/ai4co/rl4co/issues/new/choose) on the project's GitHub repository. Provide a clear and descriptive title along with relevant details to help us address the problem or understand your request.

## Acknowledgements

We adapted these contributing guidelines from [this repo](https://github.com/AntonOsika/gpt-engineer/blob/main/.github/CONTRIBUTING.md).

## Licensing

By contributing to RL4CO, you agree that your contributions will be licensed under the [LICENSE](../LICENSE) file of the project.

---

Thank you for your interest in contributing to RL4CO! We appreciate your support and look forward to your contributions.

