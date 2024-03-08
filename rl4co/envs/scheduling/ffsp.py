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
        min_time: int = 2,
        max_time: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_stage = num_stage
        self.num_machine = num_machine
        self.num_machine_total = num_stage * num_machine
        self.num_job = num_job
        self.min_time = min_time
        self.max_time = max_time

        # Init stage and machine mapping table
        self.stage_table = torch.arange(
            self.num_stage, dtype=torch.long, device=self.device
        ).repeat_interleave(self.num_machine)

        self.machine_table = torch.arange(
            self.num_machine * self.num_stage, dtype=torch.long, device=self.device
        )

        self.stage_machine_table = torch.arange(
            self.num_machine, dtype=torch.long, device=self.device
        ).repeat(self.num_stage)

    def _step(self, td: TensorDict) -> TensorDict:
        batch_size = td.batch_size

        # job_idx is the action from the model
        job_idx = td["action"]
        time_idx = td["time_idx"]
        batch_idx = td["batch_idx"]
        machine_idx = td["machine_idx"]
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

        done = (job_location[:, : self.num_job] == self.num_stage).all(dim=-1)

        update_dict = {
            "time_idx": time_idx,
            "sub_time_idx": sub_time_idx,
            "batch_idx": batch_idx,
            "machine_idx": machine_idx,
            "schedule": schedule,
            "machine_wait_step": machine_wait_step,
            "job_location": job_location,
            "job_wait_step": job_wait_step,
            "done": done,
        }

        if done.all():
            end_schedule = schedule + td["job_duration"].permute(0, 2, 1)
            end_time_max, _ = end_schedule[:, :, : self.num_job].max(dim=-1)
            end_time_max, _ = end_time_max.max(dim=-1)
            reward = end_time_max.to(torch.float32)
            update_dict["reward"] = reward

        else:
            ready = torch.flatten(done)
            idx = torch.flatten(batch_idx)
            idx = idx[~ready]

            while ~ready.all():
                new_sub_time_idx = sub_time_idx[idx] + 1
                step_time_required = new_sub_time_idx == self.num_machine_total
                time_idx[idx] += step_time_required.long()
                new_sub_time_idx[step_time_required] = 0
                sub_time_idx[idx] = new_sub_time_idx
                new_machine_idx = self.machine_table[new_sub_time_idx]
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

                new_stage_idx = self.stage_table[new_sub_time_idx]
                job_ready_1 = job_location[idx, : self.num_job] == new_stage_idx[:, None]
                job_ready_2 = job_wait_step[idx, : self.num_job] == 0
                job_ready = (job_ready_1 & job_ready_2).any(dim=-1)

                ready = machine_ready & job_ready
                idx = idx[~ready]

            stage_idx = self.stage_table[sub_time_idx]
            stage_machine_idx = self.stage_machine_table[sub_time_idx]

            job_loc = job_location[:, : self.num_job]
            job_wait_time = job_wait_step[:, : self.num_job]

            job_in_stage = job_loc == stage_idx[:, None]
            job_not_waiting = job_wait_time == 0
            job_available = job_in_stage & job_not_waiting

            job_in_previous_stages = (job_loc < stage_idx[:, None]).any(dim=-1)
            job_waiting_in_stage = (job_in_stage & (job_wait_time > 0)).any(dim=-1)
            wait_allowed = job_in_previous_stages + job_waiting_in_stage + done

            job_enable = torch.cat((job_available, wait_allowed[:, None]), dim=-1)
            job_mask = torch.full(
                size=(*batch_size, self.num_job + 1),
                dtype=torch.bool,
                device=self.device,
                fill_value=0,
            )
            job_mask[job_enable] = 1

            cost_matrix = (
                td["run_time"]
                .gather(
                    3,
                    stage_idx[:, None, None, None].expand(
                        *batch_size, self.num_job, self.num_machine, 1
                    ),
                )
                .squeeze(-1)
            )

            update_dict.update(
                {
                    "cost_matrix": cost_matrix,
                    "action_mask": job_mask,
                    "stage_idx": stage_idx,
                    "stage_machine_idx": stage_machine_idx,
                }
            )

        # Updated state
        td.update(update_dict)
        return td

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
            batch_size = self.batch_size if td is None else td.batch_size

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        # Init index record tensor
        time_idx = torch.zeros(size=(batch_size), dtype=torch.long, device=self.device)
        sub_time_idx = torch.zeros(
            size=(batch_size), dtype=torch.long, device=self.device
        )
        batch_idx = torch.arange(*batch_size)

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
            size=(batch_size),
            dtype=torch.float32,
            device=self.device,
            fill_value=float("-inf"),
        )
        done = torch.full(
            size=(batch_size),
            dtype=torch.bool,
            device=self.device,
            fill_value=False,
        )

        action_mask = torch.ones((*batch_size, self.num_job + 1), dtype=bool)
        action_mask[..., -1] = 0

        stage_idx = self.stage_table[sub_time_idx]
        machine_idx = self.machine_table[sub_time_idx]
        stage_machine_idx = self.stage_machine_table[sub_time_idx]

        cost_matrix = (
            td["run_time"]
            .gather(
                3,
                stage_idx[:, None, None, None].expand(
                    *batch_size, self.num_job, self.num_machine, 1
                ),
            )
            .squeeze(-1)
        )

        return TensorDict(
            {
                # Index information
                "stage_idx": stage_idx,
                "time_idx": time_idx,
                "sub_time_idx": sub_time_idx,
                "batch_idx": batch_idx,
                "machine_idx": machine_idx,
                "stage_machine_idx": stage_machine_idx,
                # Scheduling status information
                "schedule": schedule,
                "machine_wait_step": machine_wait_step,
                "job_location": job_location,
                "job_wait_step": job_wait_step,
                "job_duration": job_duration,
                # Finish status information
                "reward": reward,
                "done": done,
                "cost_matrix": cost_matrix,
                "action_mask": action_mask,
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
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def get_reward(self, td, actions) -> TensorDict:
        return td["reward"]

    def generate_data(self, batch_size) -> TensorDict:
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Init observation: running time of each job on each machine
        run_time = torch.randint(
            low=self.min_time,
            high=self.max_time,
            size=(*batch_size, self.num_job, self.num_machine, self.num_stage),
        ).to(self.device)

        return TensorDict(
            {
                "run_time": run_time,
            },
            batch_size=batch_size,
        )

    def render(self, td: TensorDict):
        raise NotImplementedError("TODO: render is not implemented yet")
