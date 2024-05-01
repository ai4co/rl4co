---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: ["bug"]
assignees: fedebotu, cbhua
---

## Describe the bug

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior.

Please try to provide a minimal example to reproduce the bug. Error messages and stack traces are also helpful.

Please use the markdown code blocks for both code and stack traces.

```python
import src
```

```bash
Traceback (most recent call last):
  File ...
```

## System info

Specify the version information of the `rl4co` installation you use. You can do this by including the output of `show_versions()`:

```python
from rl4co.utils import show_versions
show_versions()
```
in your report. You can run this from the command line as

```bash
python -c 'from rl4co.utils import show_versions; show_versions()'
```

## Additional context

Add any other context about the problem here.


## Reason and Possible fixes

If you know or suspect the reason for this bug, paste the code lines and suggest modifications.

## Checklist

- [ ] I have checked that there is no similar issue in the repo (**required**)
- [ ] I have provided a minimal working example to reproduce the bug (**required**)
