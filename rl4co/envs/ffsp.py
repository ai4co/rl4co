from platform import machine
from re import sub
from typing import Optional

import itertools
from sympy import poly_from_expr
import torch
from tensordict.tensordict import TensorDict

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs import RL4COEnvBase


class FFSPEnv(RL4COEnvBase):
    name = "ffsp"

    def __init__(
        self,
        num_stage: int,
        num_machine_list: list,
        num_job: int,
        min_time: float = 0.1,
        max_time: float = 1.0,
        pomo_size: int = 1,
        seed: int = None,
        device: str = "cpu",
    ):
        """Flexible Flow Shop Problem (FFSP) Environment
        Args:
        """
        super().__init__(seed=seed, device=device)
        self.num_stage = num_stage
        self.num_machine_list = num_machine_list
        self.num_machine_total = sum(num_machine_list)
        self.num_job = num_job
        self.min_time = min_time
        self.max_time = max_time
        self.pomo_size = pomo_size

        self.sm_indexer = _Stage_N_Machine_Index_Converter(self)

    @staticmethod
    def _step(self, td: TensorDict) -> TensorDict:
        """Update the states of the environment
        Args:
            - td <TensorDict>: tensor dictionary containing with the action
        """
        job_idx = td["job_idx"]
        time_idx = td["time_idx"]
        pomo_idx = td["pomo_idx"]
        batch_idx = td["batch_idx"]
        machine_idx = td["machine_idx"]
        sub_time_idx = td["sub_time_idx"]

        schedule = td["schedule"]
        schedule[batch_idx, pomo_idx, machine_idx, job_idx] = time_idx

        job_length = td["job_length"][batch_idx, job_idx, machine_idx]

        machine_wait_step = td["machine_wait_step"]
        machine_wait_step[batch_idx, pomo_idx, machine_idx] = job_length

        job_location = td["job_location"]
        job_location[batch_idx, pomo_idx, job_idx] += 1

        job_wait_step = td["job_wait_step"]
        job_wait_step[batch_idx, pomo_idx, job_idx] = job_length

        reward = torch.ones_like(done) * float("-inf")

        finish = (job_location[:, :, :self.job_num] == self.stage_num).all(dim=2)
        done = finish.all()

        if done:
            pass
        else:
            time_idx, pomo_idx, batch_idx, sub_time_idx, machine_idx, machine_wait_step, job_wait_step, job_location = self._move_to_next_machine(
                time_idx=time_idx,
                pomo_idx=pomo_idx,
                batch_idx=batch_idx,
                sub_time_idx=sub_time_idx,
                machine_idx=machine_idx,
                machine_wait_step=machine_wait_step,
                job_wait_step=job_wait_step,
                job_location=job_location,
                finish=finish,
            )
            self._update_step_state(
                sub_time_idx=sub_time_idx,
                pomo_idx=pomo_idx,
                job_location=job_location,
                job_wait_step=job_wait_step,
                finish=finish,
            )

        return TensorDict(
            {
                "next": {
                    "problem_list": td["problem_list"],
                    "time_idx": time_idx,
                    "pomo_idx": pomo_idx,
                    "batch_idx": batch_idx,
                    "machine_idx": machine_idx,
                    "schedule": schedule,
                    "machine_wait_step": machine_wait_step,
                    "job_location": job_location,
                    "job_wait_step": job_wait_step,
                    "job_durations": td["job_durations"],
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )

    def _move_to_next_machine(
            self,
            time_idx: torch.Tensor,
            pomo_idx: torch.Tensor,
            batch_idx: torch.Tensor,
            sub_time_idx: torch.Tensor,
            machine_idx: torch.Tensor,
            machine_wait_step: torch.Tensor,
            job_wait_step: torch.Tensor,
            job_location: torch.Tensor,
            finish: torch.Tensor,
        ):

        b_idx = torch.flatten(batch_idx)
        p_idx = torch.flatten(pomo_idx)
        ready = torch.flatten(finish)

        b_idx = b_idx[~ready]
        p_idx = p_idx[~ready]

        while ~ready.all():
            new_sub_time_idx = sub_time_idx[b_idx, p_idx] + 1
            step_time_required = new_sub_time_idx == self.num_machine_total
            time_idx[b_idx, p_idx] += step_time_required.long()
            new_sub_time_idx[step_time_required] = 0
            sub_time_idx[b_idx, p_idx] = new_sub_time_idx
            new_machine_idx = self.sm_indexer.get_machine_index(p_idx, new_sub_time_idx)
            machine_idx[b_idx, p_idx] = new_machine_idx

            machine_wait_steps = machine_wait_step[b_idx, p_idx, :]
            machine_wait_steps[step_time_required, :] -= 1
            machine_wait_steps[machine_wait_steps < 0] = 0
            machine_wait_step[b_idx, p_idx, :] = machine_wait_steps

            job_wait_steps = job_wait_step[b_idx, p_idx, :]
            job_wait_steps[step_time_required, :] -= 1
            job_wait_steps[job_wait_steps < 0] = 0
            job_wait_step[b_idx, p_idx, :] = job_wait_steps

            machine_ready = machine_wait_step[b_idx, p_idx, new_machine_idx] == 0

            new_stage_idx = self.sm_indexer.get_stage_index(new_sub_time_idx)
            job_ready_1 = (job_location[b_idx, p_idx, :self.num_job] == new_stage_idx[:, None])
            job_ready_2 = (job_wait_step[b_idx, p_idx, :self.num_job] == 0)
            job_ready = (job_ready_1 & job_ready_2).any(dim=1)

            ready = machine_ready & job_ready

            b_idx = b_idx[~ready]
            p_idx = p_idx[~ready]

        return time_idx, pomo_idx, batch_idx, sub_time_idx, machine_idx, machine_wait_step, job_wait_step, job_location

    def _update_step_state(
            self,
            sub_time_idx: torch.Tensor,
            pomo_idx: torch.Tensor,
            job_location: torch.Tensor,
            job_wait_step: torch.Tensor,
            finish: torch.Tensor,
        ):
        self.step_state.step_cnt += 1

        stage_idx = self.sm_indexer.get_stage_index(sub_time_idx)
        stage_machine_idx = self.sm_indexer.get_stage_machine_index(pomo_idx, sub_time_idx)

        job_loc = job_location[:, :, :self.num_job]
        job_wait_time = job_wait_step[:, :, :self.num_job]

        job_in_stage = job_loc == stage_idx[:, :, None]
        job_not_waiting = (job_wait_time == 0)
        job_available = job_in_stage & job_not_waiting

        job_in_previous_stages = (job_loc < stage_idx[:, :, None]).any(dim=2)
        job_waiting_in_stage = (job_in_stage & (job_wait_time > 0)).any(dim=2)
        wait_allowed = job_in_previous_stages + job_waiting_in_stage + finish

        job_ninf_mask = torch.full(size=(self.batch_size, self.pomo_size, self.num_job+1),
            fill_value=float('-inf'))
        job_enable = torch.cat((job_available, wait_allowed[:, :, None]), dim=2)
        job_ninf_mask[job_enable] = 0
        state_finished = finish
        return job_ninf_mask, state_finished

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["observation"].shape[:-2]

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        time_idx = torch.zeros(
            size=(self.batch_size, self.pomo_size), 
            dtype=torch.long
        ) # shape: (batch, pomo)

        sub_time_idx = torch.zeros(
            size=(self.batch_size, self.pomo_size), 
            dtype=torch.long
        ) # shape: (batch, pomo)

        pomo_idx = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        batch_idx = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        machine_idx = self.sm_indexer.get_machine_index(pomo_idx, sub_time_idx) # shape: (batch, pomo)

        schedule = torch.full(size=(self.batch_size, self.pomo_size, self.num_machine_total, self.num_job+1),
                                   dtype=torch.long, fill_value=-999999)# shape: (batch, pomo, machine, job+1)
        machine_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.num_machine_total),
                                             dtype=torch.long)# shape: (batch, pomo, machine)
        job_location = torch.zeros(size=(self.batch_size, self.pomo_size, self.num_job+1), dtype=torch.long)# shape: (batch, pomo, job+1)
        job_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.num_job+1), dtype=torch.long)# shape: (batch, pomo, job+1)
        job_durations = torch.empty(size=(self.batch_size, self.num_job+1, self.num_machine_total), dtype=torch.long) # shape: (batch, job+1, total_machine)
        job_durations[:, :self.num_job, :] = td["problem_list"]
        job_durations[:, self.num_job, :] = 0

        done = torch.full(size=(self.batch_size, self.pomo_size), dtype=torch.bool, fill_value=False)# shape: (batch, pomo)

        return TensorDict(
            {
                "problem_list": td["problem_list"],
                "time_idx": time_idx,
                "sub_time_idx": sub_time_idx,
                "pomo_idx": pomo_idx,
                "batch_idx": batch_idx,
                "machine_idx": machine_idx,
                "schedule": schedule,
                "machine_wait_step": machine_wait_step,
                "job_location": job_location,
                "job_wait_step": job_wait_step,
                "job_durations": job_durations,
                "done": done,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params: TensorDict):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            prize=BoundedTensorSpec(
                minimum=self.min_prize,
                maximum=self.max_prize,
                shape=(self.num_loc),
                dtype=torch.float32,
            ),
            prize_collect=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc),
                dtype=torch.bool,
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

    @staticmethod
    def get_reward(self, td, actions) -> TensorDict:
        job_durations_perm = td["job_durations"].permute(0, 2, 1)
        end_schedule = td["schedule"] + job_durations_perm[:, None, :, :]

        end_time_max, _ = end_schedule[:, :, :, :self.num_job].max(dim=3)
        end_time_max, _ = end_time_max.max(dim=2)

        return end_time_max

    def generate_data(self, batch_size) -> TensorDict:
        """
        Args:
            - batch_size <int> or <list>: batch size
        Returns:
            - td <TensorDict>: tensor dictionary containing the initial state
                - problem_list <Tensor> [batch_size, num_stage, num_job, num_machine]
        """
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Get random problems
        problems_INT_list = self.get_random_problems(
            batch_size=batch_size,
            num_stage=self.num_stage,
            num_machine_list=self.num_machine_list,
            num_job=self.num_job,
        )
        problem_list = []
        for idx_stage in range(self.num_stage):
            stage_problems_INT = problems_INT_list[idx_stage]
            stage_problems = stage_problems_INT.clone().type(torch.float)
            problem_list.append(stage_problems)
        
        # FIXME: doesn't work for different machine number for now
        # Shape: [batch_size, num_stage, num_job, num_machine]
        problem_list = torch.stack(problem_list, dim=-3)

        return TensorDict(
            {
                "problem_list": problem_list,
            },
            batch_size=batch_size,
        )

    def render(self, td: TensorDict):
        raise NotImplementedError("TODO: render is not implemented yet")

    def get_random_problems(
            self, 
            batch_size, 
            num_stage, 
            num_machine_list, 
            num_job, 
        ):
        '''Generate random problems for each stage.'''
        time_low = self.min_time
        time_high = self.max_time

        problems_INT_list = []
        for stage_num in range(num_stage):
            machine_cnt = num_machine_list[stage_num]
            stage_problems_INT = torch.randint(low=time_low, high=time_high, size=(batch_size, num_job, machine_cnt))
            problems_INT_list.append(stage_problems_INT)

        return problems_INT_list


class _Stage_N_Machine_Index_Converter:
    def __init__(self, env):
        assert env.machine_cnt_list == [4, 4, 4]
        assert env.pomo_size == 24

        machine_SUBindex_0 = torch.tensor(list(itertools.permutations([0, 1, 2, 3])))
        machine_SUBindex_1 = torch.tensor(list(itertools.permutations([0, 1, 2, 3])))
        machine_SUBindex_2 = torch.tensor(list(itertools.permutations([0, 1, 2, 3])))
        self.machine_SUBindex_table = torch.cat((machine_SUBindex_0, machine_SUBindex_1, machine_SUBindex_2), dim=1)
        # machine_SUBindex_table.shape: (pomo, total_machine)

        starting_SUBindex = [0, 4, 8]
        machine_order_0 = machine_SUBindex_0 + starting_SUBindex[0]
        machine_order_1 = machine_SUBindex_1 + starting_SUBindex[1]
        machine_order_2 = machine_SUBindex_2 + starting_SUBindex[2]
        self.machine_table = torch.cat((machine_order_0, machine_order_1, machine_order_2), dim=1)
        self.stage_table = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    def get_stage_index(self, sub_time_idx):
        return self.stage_table[sub_time_idx]

    def get_machine_index(self, POMO_IDX, sub_time_idx):
        # POMO_IDX.shape: (batch, pomo)
        # sub_time_idx.shape: (batch, pomo)
        return self.machine_table[POMO_IDX, sub_time_idx]
        # shape: (batch, pomo)

    def get_stage_machine_index(self, POMO_IDX, sub_time_idx):
        return self.machine_SUBindex_table[POMO_IDX, sub_time_idx]
