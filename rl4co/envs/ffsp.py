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


class FFSPEnv(RL4COEnvBase):
    """Flexible Flow Shop Problem (FFSP) environment.
    The goal is to schedule a set of jobs on a set of machines such that the makespan is minimized.

    Args:
        num_stage: number of stages
        num_machine: number of machines in each stage
        num_job: number of jobs
        min_time: minimum processing time of a job
        max_time: maximum processing time of a job
        batch_size: batch size of the problem

    Note:
        - [IMPORTANT] This version of ffsp requires the number of machines in each stage to be the same
    """

    name = "ffsp"

    def __init__(
        self,
        num_stage: int,
        num_machine: int,
        num_job: int,
        min_time: float = 0.1,
        max_time: float = 1.0,
        batch_size: list = [50],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_stage = num_stage
        self.num_machine = num_machine
        self.num_machine_total = num_stage * num_machine
        self.num_job = num_job
        self.min_time = min_time
        self.max_time = max_time
        self.batch_size = batch_size

    def _step(self, td: TensorDict) -> TensorDict:
        # job_idx is the action from the model
        job_idx = td["job_idx"]
        time_idx = td["time_idx"]
        batch_idx = td["batch_idx"]
        machine_idx = td["machine_idx"][0]
        sub_time_idx = td["sub_time_idx"]

        schedule = td["schedule"]
        schedule[batch_idx, machine_idx, job_idx] = time_idx

        job_length = td["job_duration"][batch_idx, job_idx, machine_idx]

        machine_wait_step = td["machine_wait_step"]
        machine_wait_step[batch_idx, machine_idx] = job_length

        job_location = td["job_location"]
        job_location[batch_idx, job_idx] += 1

        job_wait_step = td["job_wait_step"]
        job_wait_step[batch_idx, job_idx] = job_length

        finish = (job_location[:, : self.num_job] == self.num_stage).all(dim=-1)
        done = finish.all()

        if done:
            end_schedule = schedule + td["job_duration"].permute(0, 2, 1)
            end_time_max, _ = end_schedule[:, :, : self.job_cnt].max(dim=-1)
            end_time_max, _ = end_time_max.max(dim=-1)
            reward = end_time_max
        else:
            ready = torch.flatten(finish)
            idx = torch.flatten(batch_idx)
            idx = idx[~ready]

            while ~ready.all():
                new_sub_time_idx = sub_time_idx[idx] + 1
                step_time_required = new_sub_time_idx == self.num_machine_total
                time_idx[idx] += step_time_required.long()
                new_sub_time_idx[step_time_required] = 0
                sub_time_idx[idx] = new_sub_time_idx
                new_machine_idx = td["machine_table"][0][new_sub_time_idx]
                machine_idx[idx] = new_machine_idx

                machine_wait_steps = machine_wait_step[idx, :]
                machine_wait_steps[step_time_required, :] -= 1
                machine_wait_steps[machine_wait_steps < 0] = 0
                machine_wait_step[idx, :] = machine_wait_steps

                job_wait_steps = job_wait_step[idx, :]
                job_wait_steps[step_time_required, :] -= 1
                job_wait_steps[job_wait_steps < 0] = 0
                job_wait_step[idx, :] = job_wait_steps

                machine_ready = machine_wait_step[idx, new_machine_idx] == 0

                new_stage_idx = td["stage_table"][0][new_sub_time_idx]
                job_ready_1 = job_location[idx, : self.num_job] == new_stage_idx[:, None]
                job_ready_2 = job_wait_step[idx, : self.num_job] == 0
                job_ready = (job_ready_1 & job_ready_2).any(dim=-1)

                ready = machine_ready & job_ready
                idx = idx[~ready]

            stage_idx = td["stage_table"][0][sub_time_idx]
            stage_machine_idx = td["stage_machine_table"][0][sub_time_idx]

            job_loc = job_location[:, : self.num_job]
            job_wait_time = job_wait_step[:, : self.num_job]

            job_in_stage = job_loc == stage_idx[:, None]
            job_not_waiting = job_wait_time == 0
            job_available = job_in_stage & job_not_waiting

            job_in_previous_stages = (job_loc < stage_idx[:, None]).any(dim=-1)
            job_waiting_in_stage = (job_in_stage & (job_wait_time > 0)).any(dim=-1)
            wait_allowed = job_in_previous_stages + job_waiting_in_stage + finish

            job_enable = torch.cat((job_available, wait_allowed[:, None]), dim=-1)
            job_mask = torch.full(
                size=(*self.batch_size, self.num_job + 1),
                dtype=torch.float32,
                device=self.device,
                fill_value=float("-inf"),
            )
            job_mask[job_enable] = 0

            reward = td["reward"]

        return TensorDict(
            {
                "next": {
                    "stage_table": td["stage_table"],
                    "machine_table": td["machine_table"],
                    "time_idx": time_idx,
                    "sub_time_idx": sub_time_idx,
                    "batch_idx": batch_idx,
                    "machine_idx": machine_idx,
                    "schedule": schedule,
                    "machine_wait_step": machine_wait_step,
                    "job_location": job_location,
                    "job_wait_step": job_wait_step,
                    "job_duration": td["job_duration"],
                    "reward": reward,
                    "finish": finish,
                    # Update variables
                    "job_mask": job_mask,
                    "stage_idx": stage_idx,
                    "stage_machine_idx": stage_machine_idx,
                }
            },
            td.shape,
        )

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        """
        Args:

        Returns:
            - stage_table [batch_size, num_stage * num_machine]
            - machine_table [batch_size, num_machine * num_stage]
            - stage_machine_idx [batch_size, num_stage * num_machine]
            - time_idx [batch_size]
            - sub_time_idx [batch_size]
            - batch_idx [batch_size]
            - machine_idx [batch_size]
            - schedule [batch_size, num_machine_total, num_job+1]
            - machine_wait_step [batch_size, num_machine_total]
            - job_location [batch_size, num_job+1]
            - job_wait_step [batch_size, num_job+1]
            - job_duration [batch_size, num_job+1, num_machine * num_stage]
        """
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["observation"].shape[:-2]

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        # Init stage and machine mapping table
        stage_table = (
            torch.arange(self.num_stage, dtype=torch.long, device=self.device)
            .repeat_interleave(self.num_machine)
            .repeat(*batch_size, 1)
        )
        machine_table = torch.arange(
            self.num_machine * self.num_stage, dtype=torch.long, device=self.device
        ).repeat(*batch_size, 1)
        stage_machine_table = torch.arange(
            self.num_machine, dtype=torch.long, device=self.device
        ).repeat(*batch_size, self.num_stage)

        # Init index record tensor
        time_idx = torch.zeros(size=(batch_size), dtype=torch.long, device=self.device)
        sub_time_idx = torch.zeros(
            size=(batch_size), dtype=torch.long, device=self.device
        )
        batch_idx = torch.arange(*batch_size)
        machine_idx = machine_table[..., sub_time_idx]

        # Scheduling status information
        schedule = torch.full(
            size=(*batch_size, self.num_machine_total, self.num_job + 1),
            dtype=torch.long,
            device=self.device,
            fill_value=-999999,
        )
        machine_wait_step = torch.zeros(
            size=(*batch_size, self.num_machine_total),
            dtype=torch.long,
            device=self.device,
        )
        job_location = torch.zeros(
            size=(*batch_size, self.num_job + 1),
            dtype=torch.long,
            device=self.device,
        )
        job_wait_step = torch.zeros(
            size=(*batch_size, self.num_job + 1),
            dtype=torch.long,
            device=self.device,
        )
        job_duration = torch.empty(
            size=(*batch_size, self.num_job + 1, self.num_machine * self.num_stage),
            dtype=torch.long,
            device=self.device,
        )
        job_duration[..., : self.num_job, :] = td["run_time"].view(
            *batch_size, self.num_job, -1
        )
        job_duration[..., self.num_job, :] = 0

        # Finish status information
        reward = torch.full(
            size=(self.batch_size),
            dtype=torch.float32,
            device=self.device,
            fill_value=float("-inf"),
        )
        finish = torch.full(
            size=(self.batch_size),
            dtype=torch.bool,
            device=self.device,
            fill_value=False,
        )

        return TensorDict(
            {
                # Mapping table information
                "stage_table": stage_table,
                "machine_table": machine_table,
                "stage_machine_table": stage_machine_table,
                # Index information
                "time_idx": time_idx,
                "sub_time_idx": sub_time_idx,
                "batch_idx": batch_idx,
                "machine_idx": machine_idx,
                # Scheduling status information
                "schedule": schedule,
                "machine_wait_step": machine_wait_step,
                "job_location": job_location,
                "job_wait_step": job_wait_step,
                "job_duration": job_duration,
                # Finish status information
                "reward": reward,
                "finish": finish,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params: TensorDict):
        self.observation_spec = CompositeSpec(
            time_idx=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
            sub_time_idx=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
            batch_idx=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
            machine_idx=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
            schedule=UnboundedDiscreteTensorSpec(
                shape=(self.num_machine_total, self.num_job + 1),
                dtype=torch.int64,
            ),
            machine_wait_step=UnboundedDiscreteTensorSpec(
                shape=(self.num_machine_total),
                dtype=torch.int64,
            ),
            job_location=UnboundedDiscreteTensorSpec(
                shape=(self.num_job + 1),
                dtype=torch.int64,
            ),
            job_wait_step=UnboundedDiscreteTensorSpec(
                shape=(self.num_job + 1),
                dtype=torch.int64,
            ),
            job_duration=UnboundedDiscreteTensorSpec(
                shape=(self.num_job + 1, self.num_machine * self.num_stage),
                dtype=torch.int64,
            ),
            shape=(),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def get_reward(self, td, actions) -> TensorDict:
        return td["reward"]

    def generate_data(self, batch_size) -> TensorDict:
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Init observation: running time of each job on each machine
        run_time = (
            torch.FloatTensor(*batch_size, self.num_job, self.num_machine, self.num_stage)
            .uniform_(self.min_time, self.max_time)
            .to(self.device)
        )

        return TensorDict(
            {
                "run_time": run_time,
            },
            batch_size=batch_size,
        )

    def render(self, td: TensorDict):
        raise NotImplementedError("TODO: render is not implemented yet")


if __name__ == "__main__":
    """
    num_stage: int,
    num_machine: int,
    num_job: int,
    min_time: float = 0.1,
    max_time: float = 1.0,
    pomo_size: int = 1,
    batch_size: list = [50],
    seed: int = None,
    device: str = "cpu",
    """
    env = FFSPEnv(
        num_stage=2,
        num_machine=3,
        num_job=4,
        min_time=2,
        max_time=10,
        batch_size=[5],
        seed=None,
        device="cpu",
    )
    td = env.reset()
    print(td)

    td["job_idx"] = torch.tensor([1, 1, 1, 1, 1])
    td = env._step(td)
    print(td)
    pass
