from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td: TensorDict, actions=None, ax=None):
    raise NotImplementedError
