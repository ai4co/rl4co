from typing import Optional, Tuple

import numpy as np
import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase


class Configs:
    pass


configs = Configs()


class JSSPEnv(RL4COEnvBase):
    """Job Shop Scheduling Problem (JSSP) environment.
    As per the definition given in https://arxiv.org/pdf/2010.12367.pdf.
    The goal is to schedule a set of jobs on a set of machines such that the makespan is minimized.
    In this variation, the number of operations per job is equal to the number of machines.

    Args:

    Note:
        -
    """

    name = "jssp"

    def __init__(self, num_jobs, num_machines, **kwargs):
        super().__init__(**kwargs)

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_tasks = self.num_jobs * self.num_machines
        # the task id for first column
        self.first_col = torch.arange(
            start=0,
            end=self.num_tasks,
            step=num_machines,
            dtype=torch.int64,
            device=self.device,
        )
        # the task id for last column
        self.last_col = torch.arange(
            start=num_machines - 1,
            end=self.num_tasks,
            step=num_machines,
            dtype=torch.int64,
            device=self.device,
        )

        # initialize zero matrices for memory allocation
        self.machines = torch.zeros(
            (self.num_jobs, self.num_machines),
            dtype=torch.float32,
            device=self.device,
        )
        self.durations = torch.zeros(
            (self.num_jobs, self.num_machines),
            dtype=torch.float32,
            device=self.device,
        )
        self.durations_cp = torch.zeros(
            (self.num_jobs, self.num_machines),
            dtype=torch.float32,
            device=self.device,
        )

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    def _step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect
        if action not in self.partial_sol_sequeence:
            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.finished_mark[row, col] = 1
            op_duration = self.durations[row, col]
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = permissible_left_shift(
                action=action,
                durations=self.dur,
                machines=self.machines,
                machines_start_times=self.machines_start_times,
                operations_on_machines=self.operations_on_machines,
            )
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = startTime_a + op_duration

            self.LBs = end_time_lb(self.temp1, self.dur_cp)

            # adj matrix
            precd, succd = self.getNghbs(action, self.operations_on_machines)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if (
                flag and precd != action and succd != action
            ):  # Remove the old arc when a new operation inserts between two operations
                self.adj[succd, precd] = 0

        # prepare for return
        fea = np.concatenate(
            (
                self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
                self.finished_mark.reshape(-1, 1),
            ),
            axis=1,
        )
        reward = -(self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        """Reset the environment."""
        self.machines = td["machines"]
        self.durations = td["durations"]
        self.dur_cp = torch.copy(self.dur)
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # initialize adj matrix
        conj_nei_up_stream = torch.diag_embed(torch.ones(self.num_tasks - 1), offset=1)
        conj_nei_low_stream = torch.diag_embed(torch.ones(self.num_tasks + 1), offset=-1)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = torch.eye(self.num_tasks, dtype=torch.float32, device=self.device)
        self.adjacency = (
            self_as_nei + conj_nei_up_stream
        )  # TODO check conj_nei_low_stream

        # initialize features
        self.LBs = torch.cumsum(self.durations, dim=1)
        self.initial_quality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_end_time = torch.copy(self.initial_quality)
        self.finished_mark = torch.zeros_like(self.machines)

        features = torch.concatenate(
            [
                self.LBs.reshape(self.num_tasks, 1) / configs.et_normalize_coef,
                self.finished_mark.reshape(self.num_tasks, 1),
            ],
            dim=1,
        )

        # initialize feasible actions
        self.feasible_actions = self.first_col.to(dtype=torch.int64)

        # initialize mask
        self.mask = torch.zeros(
            size=(self.num_jobs,),
            dtype=torch.bool,
        )

        # start time of operations on machines
        self.machines_start_times = (
            torch.ones_like(self.durations.T, dtype=torch.int32) * -configs.high
        )
        # Ops ID on machines
        self.operations_on_machines = -self.num_jobs * np.ones_like(
            self.durations.T, dtype=torch.int32
        )

        self.starting_times = torch.zeros_like(self.durations, dtype=torch.float32)

        tensordict = TensorDict(
            {
                "adjacency": self.adjacency.unsqueeze(0),
                "features": features.unsqueeze(0),
                "feasible_actions": self.feasible_actions.unsqueeze(0),
                "mask": self.mask.unsqueeze(0),
            },
            batch_size=1,
        )
        return tensordict


def last_nonzero_indices(
    starting_times: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the last non-zero indices of the given 2D tensor along the columns (dim=2).

    Args:
        starting_times (torch.Tensor): 2D array with jobs starting times to find the last non-zero indices of.
            shape: (num_jobs, num_machines)
    Returns:
        Tuple of tensors containing the last non-zero indices of the given array along the given axis.
    """
    invalid_val = -1
    dim = 1
    mask = (starting_times != 0).to(dtype=torch.int32)
    val = starting_times.shape[dim] - torch.flip(mask, dims=[dim]).argmax(dim=dim) - 1
    yAxis = torch.where(mask.any(dim=dim), val, invalid_val)
    xAxis = torch.arange(starting_times.shape[0], dtype=torch.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet


def end_time_lb(starting_times: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """
    Calculate the lower bound of the end time of each job.
    It is equal to the duration for operations that have not yet started,
    otherwise it is the start time plus the cumulative sum of the durations of the operations on the same machine.

    Args:
        starting_times (torch.Tensor): batched 2D array containing the start time of each job.
            shape: (batch_size, num_jobs, num_machines)
        durations (torch.Tensor): batched 2D array containing the duration of each job.
            shape: (batch_size, num_jobs, num_machines)
    Returns:
        Tensor containing the lower bound of the end time of each job.
    """
    x, y = last_nonzero_indices(starting_times)
    durations[torch.where(starting_times != 0)] = 0
    durations[x, y] = starting_times[x, y]
    temp2 = torch.cumsum(durations, dim=1)
    temp2[np.where(starting_times != 0)] = 0
    ret = starting_times + temp2
    return ret


def permissible_left_shift(
    action, durations, machines, machines_start_times, operations_on_machines
):
    """
    Calculate the permissible left shift of the given action.
    It is equal to the duration for operations that have not yet started,
    otherwise it is the start time plus the cumulative sum of the durations of the operations on the same machine.

    Args:
        action (torch.Tensor): action taken by the agent
        durations (torch.Tensor): matrices with the duration of each task.
            For example, if durations[1, 2] = 3, it means that
            the task 2 of the job 1 takes 3 time units.
            shape (num_jobs, num_machines)
        machines (torch.Tensor): matrices with the indexes of the machines for each task.
            For example, if machines[1, 2] = 3, it means that
            the task 2 of the job 1 is executed on the machine 3.
            shape (num_jobs, num_machines)
        machines_start_times (torch.Tensor): matrices with the starting time of each task.
            For example, if machines_start_times[1, 2] = 3, it means that
            the task 2 of the job 1 starts at time 3.
            shape (num_jobs, num_machines)
        operations_on_machines (torch.Tensor): matrices with the indexes of the operations executed on each machine.
            For example, if operations_on_machines[1, 2] = 3, it means that
            the machine of index 1 executes the operation 3 at position 2.
            shape (num_machines, num_jobs)
    """
    op_ready_time, machine_ready_time = job_machines_ready_time(
        action, machines, durations, machines_start_times, operations_on_machines
    )
    op_duration = torch.take(durations, action)
    selected_machine = torch.take(machines, action) - 1
    start_times_for_selected_machine = machines_start_times[selected_machine]
    op_for_selected_machine = operations_on_machines[selected_machine]
    flag = False

    possible_pos = torch.where(op_ready_time < start_times_for_selected_machine)[0]
    # print('possible_pos:', possible_pos)
    if len(possible_pos) == 0:
        start_time = put_in_the_end(
            action,
            op_ready_time,
            machine_ready_time,
            start_times_for_selected_machine,
            op_for_selected_machine,
        )
    else:
        index_legal_pos, legal_pos, end_time_possible_pos = extract_legal_pos(
            op_duration,
            op_ready_time,
            durations,
            possible_pos,
            start_times_for_selected_machine,
            op_for_selected_machine,
        )
        # print('legal_pos:', legal_pos)
        if len(legal_pos) == 0:
            start_time = put_in_the_end(
                action,
                op_ready_time,
                machine_ready_time,
                start_times_for_selected_machine,
                op_for_selected_machine,
            )
        else:
            flag = True
            start_time = put_inbetween(
                action,
                index_legal_pos,
                legal_pos,
                end_time_possible_pos,
                start_times_for_selected_machine,
                op_for_selected_machine,
            )
    return start_time, flag


def put_in_the_end(
    action,
    op_ready_time,
    machine_ready_time,
    start_times_for_selected_machine,
    op_for_selected_machine,
):
    index = torch.where(start_times_for_selected_machine == -configs.high)[0][0]
    start_time = max(op_ready_time, machine_ready_time)
    start_times_for_selected_machine[index] = start_time
    op_for_selected_machine[index] = action
    return start_time


def extract_legal_pos(
    op_duration: torch.Tensor,
    op_ready_time: torch.Tensor,
    durations: torch.Tensor,
    possible_pos: torch.Tensor,
    start_times_for_selected_machine: torch.Tensor,
    op_for_selected_machine: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract legal positions for the given action.

    Args:
        op_duration (torch.Tensor): duration of the given action
        op_ready_time (torch.Tensor): ready time of the given action
        durations (torch.Tensor): matrices with the duration of each task.
            For example, if durations[1, 2] = 3, it means that
            the task 2 of the job 1 takes 3 time units.
            shape (num_jobs, num_machines)
        possible_pos (torch.Tensor): possible positions for the given action
        start_times_for_selected_machine (torch.Tensor): matrices with the starting time of each task.
            For example, if start_times_for_selected_machine[1, 2] = 3, it means that
            the task 2 of the job 1 starts at time 3.
            shape (num_jobs, num_machines)
        op_for_selected_machine (torch.Tensor): matrices with the indexes of the operations executed on each machine.
            For example, if op_for_selected_machine[1, 2] = 3, it means that
            the machine of index 1 executes the operation 3 at position 2.
            shape (num_machines, num_jobs)
    Returns:
        Tuple of tensors containing the index of the legal positions, the legal positions and the end time of the possible positions.
    """
    start_times_of_possible_pos = start_times_for_selected_machine[possible_pos]
    duration_possible_pos = torch.take(durations, op_for_selected_machine[possible_pos])
    earliest_start_time = max(
        op_ready_time,
        start_times_for_selected_machine[possible_pos[0] - 1]
        + torch.take(durations, [op_for_selected_machine[possible_pos[0] - 1]]),
    )
    end_time_possible_pos = torch.append(
        earliest_start_time, (start_times_of_possible_pos + duration_possible_pos)
    )[
        :-1
    ]  # end time for last ops don't care
    possible_gaps = start_times_of_possible_pos - end_time_possible_pos
    index_legal_pos = torch.where(op_duration <= possible_gaps)[0]
    legal_pos = torch.take(possible_pos, index_legal_pos)
    return index_legal_pos, legal_pos, end_time_possible_pos


def put_inbetween(
    action: torch.Tensor,
    index_legal_pos: torch.Tensor,
    legal_pos: torch.Tensor,
    end_time_possible_pos: torch.Tensor,
    start_times_for_selected_machine: torch.Tensor,
    op_for_selected_machine: torch.Tensor,
):
    earliest_idx = index_legal_pos[0]
    earliest_pos = legal_pos[0]
    start_time = end_time_possible_pos[earliest_idx]
    start_times_for_selected_machine[:] = torch.insert(
        start_times_for_selected_machine, earliest_pos, start_time
    )[:-1]
    op_for_selected_machine[:] = torch.insert(
        op_for_selected_machine, earliest_pos, action
    )[:-1]
    return start_time


def job_machines_ready_time(
    action: torch.Tensor,
    machines: torch.Tensor,
    durations: torch.Tensor,
    machines_start_times: torch.Tensor,
    operations_on_machines: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ready time on the job and machine for the selected operations (=actions).

    Args:
        action (torch.Tensor): action taken by the agent
            shape (1, )
        machines (torch.Tensor): matrices with the indexes of the machines for each task.
            For example, if machines[1, 2] = 3, it means that
            the task 2 of the job 1 is executed on the machine 3.
            shape (num_jobs, num_machines)
        durations (torch.Tensor): matrices with the duration of each task.
            For example, if durations[1, 2] = 3, it means that
            the task 2 of the job 1 takes 3 time units.
            shape (num_jobs, num_machines)
        machines_start_times (torch.Tensor): matrices with the starting time of each task.
            For example, if machines_start_times[1, 2] = 3, it means that
            the task 2 of the job 1 starts at time 3.
            shape (num_jobs, num_machines)
        operations_on_machines (torch.Tensor): matrices with the indexes of the operations executed on each machine.
            For example, if operations_on_machines[1, 2] = 3, it means that
            the machine of index 1 executes the operation 3 at position 2.
            shape (num_machines, num_jobs)

    Returns:
        ops_ready_time (torch.Tensor): ready time of the selected operations
            shape (1, )
        machines_ready_time (torch.Tensor): ready time of the machines
            shape (1, )
    """
    selected_machine = torch.take(machines, action) - 1
    previous_op_in_job = action - 1 if action % machines.shape[1] != 0 else None

    if previous_op_in_job is not None:
        duration_previous_op_in_job = torch.take(durations, previous_op_in_job)
        machine_previous_op_in_job = torch.take(machines, previous_op_in_job) - 1
        ops_ready_time = (
            machines_start_times[machine_previous_op_in_job][
                np.where(
                    operations_on_machines[machine_previous_op_in_job]
                    == previous_op_in_job
                )
            ]
            + duration_previous_op_in_job
        ).item()
    else:
        ops_ready_time = 0
    # cal machine_ready_time
    previous_op_in_machine = (
        operations_on_machines[selected_machine][
            torch.where(operations_on_machines[selected_machine] >= 0)
        ][-1]
        if len(torch.where(operations_on_machines[selected_machine] >= 0)[0]) != 0
        else None
    )
    if previous_op_in_machine is not None:
        duration_previous_op_in_machine = torch.take(durations, previous_op_in_machine)
        machines_ready_time = (
            machines_start_times[selected_machine][
                np.where(machines_start_times[selected_machine] >= 0)
            ][-1]
            + duration_previous_op_in_machine
        ).item()
    else:
        machines_ready_time = 0

    return ops_ready_time, machines_ready_time


def get_action_nbghs(
    action: torch.Tensor, op_id_on_mchs: torch.Tensor
) -> Tuple[int, int]:
    """Get the predecessor and successor of the given action in the given schedule.

    Args:
        action (torch.Tensor): The action to get the predecessor and successor of.
        op_id_on_mchs (torch.Tensor): 2D array containing the operation IDs on each machine.
    Returns:
        Tuple of ints containing the predecessor and successor of the given action.
    """
    coordAction = np.where(op_id_on_mchs == action)
    precd = op_id_on_mchs[
        coordAction[0],
        coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1],
    ].item()
    succdTemp = op_id_on_mchs[
        coordAction[0],
        coordAction[1] + 1
        if coordAction[1].item() + 1 < op_id_on_mchs.shape[-1]
        else coordAction[1],
    ].item()
    succd = action if succdTemp < 0 else succdTemp
    return precd, succd
