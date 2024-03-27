from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SMTWTPEnv(RL4COEnvBase):
    """
    Single Machine Total Weighted Tardiness Problem environment as described in DeepACO (https://arxiv.org/pdf/2309.14032.pdf)
    SMTWTP is a scheduling problem in which a set of jobs must be processed on a single machine.
    Each job i has a processing time, a weight, and a due date. The objective is to minimize the sum of the weighted tardiness of all jobs,
    where the weighted tardiness of a job is defined as the product of its weight and the duration by which its completion time exceeds its due date.
    At each step, the agent chooses a job to process. The reward is 0 unless the agent processes all the jobs.
    In that case, the reward is (-)objective value of the processing order: maximizing the reward is equivalent to minimizing the objective.

    Args:
        num_job: number of jobs
        min_time_span: lower bound of jobs' due time. By default, jobs' due time is uniformly sampled from (min_time_span, max_time_span)
        max_time_span: upper bound of jobs' due time. By default, it will be set to num_job / 2
        min_job_weight: lower bound of jobs' weights. By default, jobs' weights are uniformly sampled from (min_job_weight, max_job_weight)
        max_job_weight: upper bound of jobs' weights
        min_process_time: lower bound of jobs' process time. By default, jobs' process time is uniformly sampled from (min_process_time, max_process_time)
        max_process_time: upper bound of jobs' process time
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """

    name = "smtwtp"

    def __init__(
        self,
        num_job: int = 10,
        min_time_span: float = 0,
        max_time_span: float = None,  # will be set to num_job/2 by default. In DeepACO, it is set to num_job, which would be too simple
        min_job_weight: float = 0,
        max_job_weight: float = 1,
        min_process_time: float = 0,
        max_process_time: float = 1,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_job = num_job
        self.min_time_span = min_time_span
        self.max_time_span = num_job / 2 if max_time_span is None else max_time_span
        self.min_job_weight = min_job_weight
        self.max_job_weight = max_job_weight
        self.min_process_time = min_process_time
        self.max_process_time = max_process_time
        self._make_spec(td_params)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_job = td["action"]

        # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_job.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # Increase used time
        selected_process_time = td["job_process_time"][
            torch.arange(current_job.size(0)), current_job
        ]
        current_time = td["current_time"] + selected_process_time.unsqueeze(-1)

        # We are done there are no unvisited locations
        done = torch.count_nonzero(available, dim=-1) <= 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_job": current_job,
                "current_time": current_time,
                "action_mask": available,
                "reward": reward,
                "done": done,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialization
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["job_due_time"].shape[:-1]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        device = td["job_due_time"].device if td is not None else self.device
        self.to(device)

        td = self.generate_data(batch_size) if td is None else td

        init_job_due_time = td["job_due_time"]
        init_job_process_time = td["job_process_time"]
        init_job_weight = td["job_weight"]

        # Other variables
        current_job = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        current_time = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, self.num_job + 1), dtype=torch.bool, device=device
        )
        available[:, 0] = 0  # mask the starting dummy node

        return TensorDict(
            {
                "job_due_time": init_job_due_time,
                "job_weight": init_job_weight,
                "job_process_time": init_job_process_time,
                "current_job": current_job,
                "current_time": current_time,
                "action_mask": available,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params: TensorDict = None):
        self.observation_spec = CompositeSpec(
            job_due_time=BoundedTensorSpec(
                low=self.min_time_span,
                high=self.max_time_span,
                shape=(self.num_job + 1,),
                dtype=torch.float32,
            ),
            job_weight=BoundedTensorSpec(
                low=self.min_job_weight,
                high=self.max_job_weight,
                shape=(self.num_job + 1,),
                dtype=torch.float32,
            ),
            job_process_time=BoundedTensorSpec(
                low=self.min_process_time,
                high=self.max_process_time,
                shape=(self.num_job + 1,),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_job + 1,),
                dtype=torch.bool,
            ),
            current_time=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.num_job + 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def get_reward(self, td, actions) -> TensorDict:
        job_due_time = td["job_due_time"]
        job_weight = td["job_weight"]
        job_process_time = td["job_process_time"]

        batch_idx = torch.arange(
            job_process_time.shape[0], device=job_process_time.device
        ).unsqueeze(1)

        ordered_process_time = job_process_time[batch_idx, actions]
        ordered_due_time = job_due_time[batch_idx, actions]
        ordered_job_weight = job_weight[batch_idx, actions]
        presum_process_time = torch.cumsum(
            ordered_process_time, dim=1
        )  # ending time of each job
        job_tardiness = presum_process_time - ordered_due_time
        job_tardiness[job_tardiness < 0] = 0
        job_weighted_tardiness = ordered_job_weight * job_tardiness

        return -job_weighted_tardiness.sum(-1)

    def generate_data(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        # Sampling according to Ye et al. (2023)
        job_due_time = (
            torch.FloatTensor(*batch_size, self.num_job + 1)
            .uniform_(self.min_time_span, self.max_time_span)
            .to(self.device)
        )
        job_weight = (
            torch.FloatTensor(*batch_size, self.num_job + 1)
            .uniform_(self.min_job_weight, self.max_job_weight)
            .to(self.device)
        )
        job_process_time = (
            torch.FloatTensor(*batch_size, self.num_job + 1)
            .uniform_(self.min_process_time, self.max_process_time)
            .to(self.device)
        )

        # Rollouts begin at dummy node 0, whose features are set to 0
        job_due_time[:, 0] = 0
        job_weight[:, 0] = 0
        job_process_time[:, 0] = 0

        return TensorDict(
            {
                "job_due_time": job_due_time,
                "job_weight": job_weight,
                "job_process_time": job_process_time,
            },
            batch_size=batch_size,
        )

    @staticmethod
    def render(td, actions=None, ax=None):
        raise NotImplementedError("TODO: render is not implemented yet")
