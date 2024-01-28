from typing import Optional, Tuple

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec

from rl4co.envs.common.base import RL4COEnvBase


class Configs:
    low = 1
    high = 99
    et_normalize_coef = 1000
    rewardscale = 0.0
    init_quality_flag = False


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
        self.batch_size = torch.Size([1])
        adjacency_spec = DiscreteTensorSpec(
            n=2,
            shape=torch.Size((1, num_jobs * num_machines, num_jobs * num_machines)),
            device=self.device,
            dtype=torch.int64,
        )
        features_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size((1, num_jobs * num_machines, 2)),
            device=self.device,
        )
        feasible_actions_spec = DiscreteTensorSpec(
            n=num_jobs * num_machines,
            shape=torch.Size(
                (
                    1,
                    num_jobs,
                )
            ),
            device=self.device,
            dtype=torch.int64,
        )
        action_mask_spec = DiscreteTensorSpec(
            n=2,
            shape=torch.Size(
                (
                    1,
                    num_jobs,
                )
            ),
            device=self.device,
            dtype=torch.bool,
        )

        self.observation_spec = CompositeSpec(
            adjacency=adjacency_spec,
            features=features_spec,
            feasible_actions=feasible_actions_spec,
            action_mask=action_mask_spec,
            shape=self.batch_size,
        )

        self.action_spec = DiscreteTensorSpec(
            n=num_jobs,
            shape=self.batch_size,
            device=self.device,
            dtype=torch.int64,
        )

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
        if len(self.partial_sol_sequence) == self.num_tasks:
            return torch.tensor(True)
        return torch.tensor(False)

    def _step(self, td: TensorDict) -> TensorDict:
        job_idx = td["action"].squeeze()
        # action is an int 0 - num_jobs
        # the actual action is the operation id of the selected job
        # taken from the feasible actions:
        action = self.feasible_actions[job_idx].clone()  # range 0 - num_tasks

        if action not in self.partial_sol_sequence:
            # UPDATE BASIC INFO:
            row = action // self.num_machines
            col = action % self.num_machines
            self.finished_mark[row, col] = 1
            op_duration = self.durations[row, col]
            self.partial_sol_sequence.append(action)

            # UPDATE STATE:
            # permissible left shift
            start_time, flag = permissible_left_shift(
                action=action,
                durations=self.durations,
                machines=self.machines,
                machines_start_times=self.machines_start_times,
                operations_on_machines=self.operations_on_machines,
            )
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.feasible_actions[action // self.num_machines] += 1
            else:
                self.mask[action // self.num_machines] = 1

            self.ending_times[row, col] = start_time + op_duration

            self.LBs = end_time_lb(self.ending_times, self.durations_cp)

            # adj matrix
            precd, succd = get_action_nbghs(action, self.operations_on_machines)
            self.adjacency[action] = 0
            self.adjacency[action, action] = 1
            if action not in self.first_col:
                self.adjacency[action, action - 1] = 1
            self.adjacency[action, precd] = 1
            self.adjacency[succd, action] = 1
            if (
                flag and precd != action and succd != action
            ):  # Remove the old arc when a new operation inserts between two operations
                self.adjacency[succd, precd] = 0

        # prepare for return
        features = torch.concatenate(
            (
                self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
                self.finished_mark.reshape(-1, 1),
            ),
            dim=1,
        )
        reward = -(self.LBs.max() - self.max_end_time)
        if reward == 0:
            reward = torch.tensor(configs.rewardscale)
            self.positive_reward += reward
        self.max_end_time = self.LBs.max()

        tensordict = TensorDict(
            {
                "adjacency": self.adjacency.unsqueeze(0),
                "features": features.unsqueeze(0),
                "feasible_actions": self.feasible_actions.unsqueeze(0),
                "action_mask": ~self.mask.unsqueeze(0),
                "done": self.done().unsqueeze(0),
                "reward": reward.unsqueeze(0),
            },
            batch_size=1,
        )

        return tensordict

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        """Reset the environment."""
        if td is None:
            td = uniform_instance_gen(
                self.num_jobs, self.num_machines, configs.low, configs.high
            )

        self.machines = td["machines"].squeeze(0)
        self.durations = td["durations"].squeeze(0)
        self.durations_cp = self.durations.clone()
        # record action history
        self.partial_sol_sequence = []
        self.flags = []
        self.positive_reward = 0

        # initialize adj matrix
        conj_nei_up_stream = torch.diag_embed(torch.ones(self.num_tasks - 1), offset=-1)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        self_as_nei = torch.eye(self.num_tasks, dtype=torch.float32, device=self.device)
        self.adjacency = self_as_nei + conj_nei_up_stream

        # initialize features
        self.LBs = torch.cumsum(self.durations, dim=1)
        self.initial_quality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_end_time = self.initial_quality.clone()
        self.finished_mark = torch.zeros_like(self.machines)

        features = torch.concatenate(
            [
                self.LBs.reshape(self.num_tasks, 1) / configs.et_normalize_coef,
                self.finished_mark.reshape(self.num_tasks, 1),
            ],
            dim=1,
        )

        # initialize feasible actions
        self.feasible_actions = self.first_col.to(dtype=torch.int64).clone()

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
        self.operations_on_machines = -self.num_jobs * torch.ones_like(
            self.durations.T, dtype=torch.int32
        )

        self.ending_times = torch.zeros_like(self.durations, dtype=torch.float32)

        tensordict = TensorDict(
            {
                "adjacency": self.adjacency.unsqueeze(0),
                "features": features.unsqueeze(0),
                "feasible_actions": self.feasible_actions.unsqueeze(0),
                "action_mask": ~self.mask.unsqueeze(0),
            },
            batch_size=1,
        )
        return tensordict

    def get_reward(self, td, actions):
        return self.positive_reward.unsqueeze(0)

    def render(self, *args, **kwargs):
        """Display a gantt chart of the solution."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(50, 25))
        plt.title("Gantt Chart")
        plt.xlabel("Time")
        plt.ylabel("Machine")
        plt.yticks(range(self.num_machines), range(1, self.num_machines + 1))
        plt.grid(True)

        durAlongMchs = torch.take(
            self.durations, self.operations_on_machines.to(dtype=torch.long)
        )

        for machine in range(self.num_machines):
            for job in range(self.num_jobs):
                # job_idx = task // self.num_machines
                # start_time_idx = (self.operations_on_machines == task).nonzero(as_tuple=True)
                # mac_idx = start_time_idx[0]
                task = self.operations_on_machines[machine, job]
                job_num = task // self.num_machines
                plt.barh(
                    y=machine,
                    left=self.machines_start_times[machine, job],
                    width=durAlongMchs[machine, job],
                    color="C{}".format(job_num),
                    label="Job {}".format(job_num),
                )
                # add the duration on the bar
                text_to_add = [
                    f"Task {task}:",
                    f"{self.machines_start_times[machine, job].item()}",
                    f"{(self.machines_start_times[machine, job] + durAlongMchs[machine, job]).item()}",
                ]
                for idx, line in enumerate(text_to_add):
                    plt.text(
                        self.machines_start_times[machine, job],
                        machine - idx * 0.2,
                        line,
                        ha="left",
                        va="bottom",
                    )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


def last_nonzero_indices(
    ending_times: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the last non-zero indices of the given 2D tensor along the columns (dim=2).

    Args:
        ending_times (torch.Tensor): 2D array with jobs starting times to find the last non-zero indices of.
            shape: (num_jobs, num_machines)
    Returns:
        Tuple of tensors containing the last non-zero indices of the given array along the given axis.
    """
    invalid_val = -1
    dim = 1
    mask = (ending_times != 0).to(dtype=torch.int32)
    val = ending_times.shape[dim] - torch.flip(mask, dims=[dim]).argmax(dim=dim) - 1
    yAxis = torch.where(mask.any(dim=dim), val, invalid_val)
    xAxis = torch.arange(ending_times.shape[0], dtype=torch.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet


def end_time_lb(ending_times: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """
    Calculate the lower bound of the end time of each job.
    It is equal to the duration for operations that have not yet started,
    otherwise it is the start time plus the cumulative sum of the durations of the operations on the same machine.

    Args:
        ending_times (torch.Tensor): batched 2D array containing the start time of each job.
            shape: (batch_size, num_jobs, num_machines)
        durations (torch.Tensor): batched 2D array containing the duration of each job.
            shape: (batch_size, num_jobs, num_machines)
    Returns:
        Tensor containing the lower bound of the end time of each job.
    """
    x, y = last_nonzero_indices(ending_times)  # get the last operation of each job
    durations[
        torch.where(ending_times != 0)
    ] = 0  # set the duration of already started operations to 0
    durations[x, y] = ending_times[
        x, y
    ]  # set the duration of the last operation of each job to its start time
    temp2 = torch.cumsum(
        durations, dim=1
    )  # calculate the cumulative sum of the durations of the operations on the same machine
    temp2[
        torch.where(ending_times != 0)
    ] = 0  # set the cumulative sum of the durations of already started operations to 0
    ret = ending_times + temp2
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
    duration_possible_pos = torch.take(
        durations, op_for_selected_machine[possible_pos].to(torch.long)
    )
    earliest_start_time = torch.maximum(
        op_ready_time,
        start_times_for_selected_machine[possible_pos[0] - 1]
        + torch.take(
            durations, op_for_selected_machine[possible_pos[0] - 1].to(torch.long)
        ),
    )
    end_time_possible_pos = torch.cat(
        [earliest_start_time, start_times_of_possible_pos + duration_possible_pos]
    )[
        :-1
    ]  # end time for last ops don't care
    possible_gaps = start_times_of_possible_pos - end_time_possible_pos
    index_legal_pos = torch.where(op_duration <= possible_gaps)[0]
    legal_pos = torch.take(possible_pos, index_legal_pos)
    # op_for_selected_machine[
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
    start_times_for_selected_machine[:] = torch.cat(
        [
            start_times_for_selected_machine[:earliest_pos],
            start_time.unsqueeze(0),
            start_times_for_selected_machine[earliest_pos:],
        ]
    )[:-1]
    op_for_selected_machine[:] = torch.cat(
        [
            op_for_selected_machine[:earliest_pos],
            action.unsqueeze(0),
            op_for_selected_machine[earliest_pos:],
        ]
    )[:-1]
    return start_time


def job_machines_ready_time(
    action: torch.Tensor,
    machines: torch.Tensor,
    durations: torch.Tensor,
    machine_start_times: torch.Tensor,
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
        machine_start_times (torch.Tensor): matrices with the starting time of each task.
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
        op_ready_time = (
            machine_start_times[machine_previous_op_in_job][
                torch.where(
                    operations_on_machines[machine_previous_op_in_job]
                    == previous_op_in_job
                )
            ]
            + duration_previous_op_in_job
        )
    else:
        op_ready_time = torch.tensor([0])
    # cal machine_ready_time
    previous_op_in_machine = (
        operations_on_machines[selected_machine][
            torch.where(operations_on_machines[selected_machine] >= 0)
        ][-1]
        if len(torch.where(operations_on_machines[selected_machine] >= 0)[0]) != 0
        else None
    )
    if previous_op_in_machine is not None:
        duration_previous_op_in_machine = torch.take(
            durations, previous_op_in_machine.to(torch.long)
        )
        machine_ready_time = (
            machine_start_times[selected_machine][
                torch.where(machine_start_times[selected_machine] >= 0)
            ][-1]
            + duration_previous_op_in_machine
        ).item()
    else:
        machine_ready_time = 0

    return op_ready_time, machine_ready_time


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
    coordAction = torch.nonzero(op_id_on_mchs == action, as_tuple=True)
    precd = op_id_on_mchs[
        coordAction[0],
        coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1],
    ].item()
    succ_temp = op_id_on_mchs[
        coordAction[0],
        coordAction[1] + 1
        if coordAction[1].item() + 1 < op_id_on_mchs.shape[-1]
        else coordAction[1],
    ].item()
    succd = action.item() if succ_temp < 0 else succ_temp
    return int(precd), int(succd)


def permute_rows(x: torch.Tensor) -> torch.Tensor:
    ix_i = torch.tile(torch.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = torch.rand(x.shape).argsort(dim=1)
    return x[ix_i, ix_j]


def uniform_instance_gen(n_j, n_m, low, high):
    times = torch.randint(low=low, high=high, size=(n_j, n_m), dtype=torch.float32)
    machines = torch.arange(1, n_m + 1).unsqueeze(0).repeat(n_j, 1)
    machines = permute_rows(machines)
    return TensorDict(
        {"durations": times.unsqueeze(0), "machines": machines.unsqueeze(0)}, batch_size=1
    )
