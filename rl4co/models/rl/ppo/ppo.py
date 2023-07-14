from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PPO(RL4COLitModule):
    """
    PPO -> TODO
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
