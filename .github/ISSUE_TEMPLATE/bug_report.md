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

Report the characteristics of your environment such as how the library was installed (pip or source), Python version, and version of other relevant libraries.

> [!TIP] 
> You can use the script below to report relevant versions. Just make sure you are using the same Python environment where the bug occurred!

```bash
python -c "import rl4co, torch, lightning, torchrl, tensordict, numpy, sys; print('RL4CO:', \
 rl4co.__version__, '\nPyTorch:', torch.__version__, '\nPyTorch Lightning:', \
lightning.__version__, '\nTorchRL:',  torchrl.__version__, '\nTensorDict:',\
 tensordict.__version__, '\nNumpy:', numpy.__version__, '\nPython:', \
sys.version, '\nPlatform:', sys.platform)"
```

## Additional context

Add any other context about the problem here.


## Reason and Possible fixes

If you know or suspect the reason for this bug, paste the code lines and suggest modifications.

## Checklist

- [ ] I have checked that there is no similar issue in the repo (**required**)
- [ ] I have provided a minimal working example to reproduce the bug (**required**)
