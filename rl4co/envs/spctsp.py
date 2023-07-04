from rl4co.envs.pctsp import PCTSPEnv
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SPCTSPEnv(PCTSPEnv):
    name = "spctsp"
    _stochastic = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def stochastic(self):
        return self._stochastic

    @stochastic.setter
    def stochastic(self, state: bool):
        if state is False:
            log.warning(
                "Deterministic mode should not be used for SPCTSP. Use PCTSP instead."
            )
