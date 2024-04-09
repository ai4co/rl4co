from typing import Optional

import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase


class JSSPEnv(RL4COEnvBase):
    """Job Shop Scheduling Problem (JSSP) environment.
    As per the definition given in https://arxiv.org/pdf/2010.12367.pdf.
    The goal is to schedule a set of jobs on a set of machines such that the makespan is minimized.
    In this variation, the number of operations per job is equal to the number of machines.

    Args:
        num_jobs (int): Number of jobs.
        num_machines (int): Number of machines.
        low (int, optional): Lower bound for the random generation of the durations. Defaults to 1.
        high (int, optional): Upper bound for the random generation of the durations. Defaults to 99.
        et_normalize_coef (int, optional): Coefficient used to normalize the end time of each job. Defaults to 1000.
        rewardscale (float, optional): Scale factor for the reward. Defaults to 0.0.
        init_quality_flag (bool, optional): Flag to initialize the quality of the initial solution to 0.
            Defaults to False.

    Note:
        The number of operations per job is equal to the number of machines.
    """

    name = "jssp"

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        low: int = 1,
        high: int = 99,
        stepwise_reward: bool = False,
        normalize_reward: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._low = low
        self._high = high

        self.stepwise_reward = stepwise_reward

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_tasks = self.num_jobs * self.num_machines
        self.normalize_reward = normalize_reward

        # create specs for observation and action
        # self._make_spec()

        # the task id for first operation in precedence relation; shape=(num_jobs,)
        self.first_task = torch.arange(
            start=0,
            end=self.num_tasks,
            step=num_machines,
            dtype=torch.int64,
            device=self.device,
        )
        # the task id for last column; shape=(num_jobs,)
        self.last_task = torch.arange(
            start=num_machines - 1,
            end=self.num_tasks,
            step=num_machines,
            dtype=torch.int64,
            device=self.device,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        td = td.clone()
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)
        # (bs,)
        job_idx = td["action"]
        # action is an int 0 - num_jobs
        # the actual action is the operation id of the selected job; shape=(bs,)
        task_idx = td["next_op"][batch_idx, job_idx]
        col = task_idx % self.num_machines
        ma_idx = torch.where(td["machines"][batch_idx, job_idx] == col[:, None])[1]
        assert (task_idx // self.num_machines == job_idx).all()

        # mark scheduled operations
        td["finished_mark"][batch_idx, job_idx, col] = 1
        op_duration = td["durations"][batch_idx, job_idx, col]

        # (bs,) first operation of job has no predecessor
        previous_op_in_job = ~torch.isin(task_idx, self.first_task)
        # gather ending times of previous operation if exists
        # (NOTE clipping only effects first operation which has no predecessor)
        previous_op_end_time = td["end_times"][batch_idx, job_idx, torch.clip(col - 1, 0)]
        # previous_op_end_time1 = td["end_times"].reshape(*batch_size, -1)[batch_idx, torch.clip(task_idx-1, 0)]
        # assert torch.all(previous_op_end_time[previous_op_in_job] == previous_op_end_time1[previous_op_in_job])
        # (bs,)
        op_ready_time = torch.where(
            previous_op_in_job,
            previous_op_end_time,
            # if no preceeding operation exists, op can start directly,
            torch.zeros((*batch_size,), dtype=torch.float32, device=self.device),
        )
        assert not op_ready_time.lt(0).any()
        # sort the schedule in accordance to the machine indices; shape=(bs, jobs, ma)
        ma_starts = td["start_times"].gather(2, td["machines"])
        ma_ends = td["end_times"].gather(2, td["machines"])
        # get start and end times of selected machine; shape=(bs, num_jobs)
        start_chosen_ma = ma_starts.gather(
            2, ma_idx[:, None, None].expand(-1, self.num_jobs, 1)
        ).squeeze(2)
        end_chosen_ma = ma_ends.gather(
            2, ma_idx[:, None, None].expand(-1, self.num_jobs, 1)
        ).squeeze(2)
        start_chosen_ma, ids1 = start_chosen_ma.sort()
        end_chosen_ma, ids2 = end_chosen_ma.sort()
        assert (ids1 == ids2).all(), "The order in the schedule is messed up"
        # get the time the machine is idle again; shape=(bs,)
        ma_ready_time = end_chosen_ma.max(1).values.clip(0)
        # we may put an op before another op, if it is ready before the other one starts
        possible_positions = op_ready_time[:, None] < start_chosen_ma
        eligible4ls = possible_positions.any(1)

        # (bs, jobs)
        lag1_end_times_ma = torch.cat(
            (torch.zeros_like(end_chosen_ma[:, :1]), end_chosen_ma), dim=1
        )[batch_idx, :-1]
        # determine idle gaps on selected machine. We pick the maximum of the (potentially) preceeding
        # operations end time and the op_ready_time as the latter poses the earliest possible start time.
        # Hence, the determined gap must be big enough when starting at this time step
        gaps = start_chosen_ma - torch.max(op_ready_time[:, None], lag1_end_times_ma)
        gaps[~possible_positions] = -torch.inf
        sufficient_gaps = gaps > op_duration[:, None]
        # get the index of the smallest but still sufficiently large gap
        _, min_pos = gaps.masked_fill(~sufficient_gaps, torch.inf).min(1)
        min_pos_start = lag1_end_times_ma.gather(1, min_pos[:, None]).squeeze(1)
        # (bs, )
        left_shift_possible = torch.logical_and(eligible4ls, sufficient_gaps.any(1))
        start_times = torch.where(
            left_shift_possible,
            torch.max(op_ready_time, min_pos_start),
            # if no gap is big enough, we have to put the op at the end of schedule
            torch.max(op_ready_time, ma_ready_time),
        )
        # update schedule
        td["start_times"][batch_idx, job_idx, col] = start_times
        td["end_times"][batch_idx, job_idx, col] = start_times + op_duration

        # (bs, 1)
        job_done = torch.isin(task_idx, self.last_task)

        # mask jobs that are completed; shape=(bs, jobs)
        td["action_mask"][job_done, job_idx[job_done]] = 0
        # increment operation counter on all other jobs; shape=(bs, jobs)
        td["next_op"][~job_done, job_idx[~job_done]] += 1

        adjacency = self.update_adjacency(
            td, job_idx, task_idx, ma_idx, previous_op_in_job, left_shift_possible
        )

        old_lbs = td["lower_bounds"].clone()
        LBs = end_time_lb(td)

        done = torch.all(td["action_mask"] == 0, 1)
        if self.stepwise_reward:
            reward = -(LBs.amax(dim=(1, 2)) - old_lbs.amax(dim=(1, 2)))
        elif done.all():
            # get makespan
            reward = -td["end_times"].amax(dim=(1, 2))
        else:
            reward = td["reward"]

        return td.update(
            {
                "adjacency": adjacency,
                "lower_bounds": LBs,
                "reward": reward,
                "done": done,
            }
        )

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        """Reset the environment."""
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        self.to(td.device)
        # (bs, jobs, ma)
        durations = td["durations"]
        # initialize adj matrix of predecessors. We leave an offset of -1, since the first
        # operation has no predecessor: shape=(num_tasks, num_tasks)
        predecessors_adj = torch.diag_embed(
            torch.ones(self.num_tasks - 1, device=self.device), offset=-1
        )
        # first operation of jobs do not have predecessors
        predecessors_adj[self.first_task] = 0
        # ---------------
        # NOTE in original implementation authors dont use successors.
        # initialize adj matrix of successors. We leave an offset of +1, since the last
        # operation has no successors: shape=(num_tasks, num_tasks)
        successor_adj = torch.diag_embed(
            torch.ones(self.num_tasks - 1, device=self.device), offset=1
        )
        # first operation of jobs do not have successors
        successor_adj[self.last_task] = 0
        # ---------------
        # self loops and adding all neighbors in final adjacency matrix
        self_loop = torch.eye(self.num_tasks, dtype=torch.float32, device=self.device)
        adjacency = torch.unsqueeze(self_loop + predecessors_adj, 0).repeat(
            *batch_size, 1, 1
        )
        # the following adjacency matrix indicates which ops are executed on the same machine
        ops_on_same_ma_adj = (
            td["ops"].reshape(*batch_size, -1, 1) == td["ops"].reshape(*batch_size, 1, -1)
        ).to(torch.float32)

        # (bs, jobs, ma)
        LBs = torch.cumsum(durations, dim=-1)
        # (bs, jobs, ma)
        finished_mark = torch.zeros_like(durations)

        machines = td["machines"]
        # ops = torch.arange(self.num_machines)[None, None, :].repeat(batch_size, self.num_jobs, 1)
        # task_ma_map=(ops[:, :, None] == machines[..., None]).reshape(batch_size, self.num_tasks, -1)

        # initialize next_op; shape=(bs, num_jobs)
        next_op = self.first_task[None].repeat(*batch_size, 1).to(dtype=torch.int64)

        # initialize mask (mask specifies feasible actions)
        mask = torch.ones(
            size=(*batch_size, self.num_jobs),
            dtype=torch.bool,
            device=self.device,
        )

        # start time of operations on machines; (bs, jobs, ma)
        start_times = torch.full_like(durations, fill_value=-self._high)
        # ending times of ops on machines; (bs, jobs, ma)
        ending_times = torch.zeros_like(durations, dtype=torch.float32)

        tensordict = TensorDict(
            {
                "adjacency": adjacency,
                "ops_on_same_ma_adj": ops_on_same_ma_adj,
                "machines": machines,
                "start_times": start_times,
                "end_times": ending_times,
                "lower_bounds": LBs,
                "finished_mark": finished_mark,
                "next_op": next_op,
                "durations": durations,
                "action_mask": mask,
                "done": torch.full((*batch_size,), False),
                "reward": torch.empty(
                    (*batch_size,), dtype=torch.float32, device=self.device
                ),
            },
            batch_size=batch_size,
        )
        return tensordict

    def generate_data(self, batch_size):
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        # randomly generate processing times for each operation (job-machine combination)
        # (bs, jobs, ma)
        proc_times = torch.randint(
            low=self._low,
            high=self._high,
            size=(*batch_size, self.num_jobs, self.num_machines),
            dtype=torch.float32,
            device=self.device,
        )

        # the machines2ops tensor of shape=(bs, jobs, ma) specifies which machine executes which op,
        # e.g. ops[:, 0, 1] = 2 means that machine=1 executes operation=2 of job=0
        machines2ops = torch.rand(
            (*batch_size, self.num_jobs, self.num_machines), device=self.device
        ).argsort(dim=-1)
        # the ops2machines tensor of shape=(bs, jobs, ma) specifies which operation is executed on
        # which machine, e.g. machines[:, 0, 1] = 2 means that operation=1 of job=0 is executed
        # on machine=1
        ops2machines = machines2ops.argsort(2)

        return TensorDict(
            {"durations": proc_times, "machines": machines2ops, "ops": ops2machines},
            batch_size=batch_size,
        )

    def get_reward(self, td, _):
        reward = td["reward"]
        if not td["done"].all() and not self.stepwise_reward:
            raise AttributeError(
                "Use stepwise_reward=True in JSSPEnv if you require the reward after every step"
            )
        if self.normalize_reward:
            reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        return reward

    def update_adjacency(
        self,
        td: TensorDict,
        job_idx,
        task_idx,
        ma_idx,
        previous_op_in_job,
        left_shift_possible,
    ):
        # define batch_idx
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)
        # get machine predecessor and successor nodes of operation, i.e. ops
        # that are processed directly before and after the selected operation
        pred_of_op, succ_of_op = self.get_machine_precedence(
            td, batch_idx, job_idx, task_idx, ma_idx
        )

        adjacency = td["adjacency"]
        adjacency[batch_idx, task_idx] = 0  # reset
        adjacency[batch_idx, task_idx, task_idx] = 1  # self-loop
        # preceeding op in job
        task_masked = task_idx[previous_op_in_job]
        adjacency[previous_op_in_job, task_masked, task_masked - 1] = 1

        adjacency[batch_idx, task_idx, pred_of_op] = 1  # preceeding op on machine
        adjacency[batch_idx, succ_of_op, task_idx] = 1  # succeeding op on machine

        # remove arc between pred and succ of of, if op has been placed between them
        s_masked = succ_of_op[left_shift_possible]
        p_masked = pred_of_op[left_shift_possible]
        # assert torch.all(adjacency[left_shift_possible, s_masked, p_masked] == 1)
        adjacency[left_shift_possible, s_masked, p_masked] = 0

        return adjacency

    def get_machine_precedence(
        self, td: TensorDict, batch_idx, job_idx, task_idx, ma_idx
    ):
        """This function determines the precedence relationships of operations on the same
        machine given the current schedule. Given a partial schedule of start times, we
        first align these start_times per machine type. Then, we determine, given these start_times,
        which job is executed first, second and so on and after that determine the task_id that
        corresponds to the job-machine combo.
        ------------
        NOTE
        In the original implementation the authors track the machine schedule in a numpy array,
        where schedule[1,2]=3 specifies that machine=1 processes task=3 in position=2. This does not
        work in torch, since we cannot simply insert tasks at different positions along the batches
        (without using a for loop). This solution is imo cleaner.
        ------------
        """

        # transform the 'operations schedule' into a 'machine schedule', i.e. sort
        # the last dimension according to the machine the respective op is executed on
        ma_end_times = td["end_times"].gather(2, td["machines"])
        # mask not scheduled ops
        ma_end_times[ma_end_times == 0] = torch.inf

        # (bs, jobs, ma) --> sort jobs in accordence to their position in current schedule
        ma_precedence_order = (
            ma_end_times.gather(
                2, ma_idx[:, None, None].expand(*td.batch_size, self.num_jobs, 1)
            )
            .squeeze(2)
            .argsort(1)
        )
        # get position of the selected operation in current schedule
        op_pos_on_ma = torch.where(ma_precedence_order == job_idx[:, None])[1]
        # get the id of the job, directly preceeding the selected job. Select the job itself
        # if it has no predecessor (i.e. scheduled on first position in machine schedule)
        pred_job = torch.where(
            op_pos_on_ma == 0,
            job_idx,
            ma_precedence_order[batch_idx, torch.clip(op_pos_on_ma - 1, 0)],
        )
        # get the id of the job, directly succeeding the selected job. Select the job itself
        # if it has no successor (i.e. scheduled on last position in machine schedule)
        succ_job = torch.where(
            op_pos_on_ma == self.num_jobs - 1,
            job_idx,
            ma_precedence_order[
                batch_idx, torch.clip(op_pos_on_ma + 1, max=self.num_jobs - 1)
            ],
        )
        # to uniquely identify the tasks, add the first_task increment to the task indices in machines tensor
        tasks_on_ma = td["machines"] + self.first_task[None, :, None]
        # mask operations that are not scheduled yet
        tasks_on_ma[~td["finished_mark"].gather(2, td["machines"]).to(torch.bool)] = -1
        # get the operation that corresponds to the job-machine combo
        pred_of_op = tasks_on_ma[batch_idx, pred_job, ma_idx]
        succ_of_op = tasks_on_ma[batch_idx, succ_job, ma_idx]
        # mask successors that have not been scheduled yet
        succ_of_op = torch.where(succ_of_op < 0, task_idx, succ_of_op)
        return pred_of_op, succ_of_op

    def render(self, td: TensorDict, idx: int = 0):
        """Display a gantt chart of the solution."""
        from collections import OrderedDict

        import matplotlib.pyplot as plt

        # plt.figure(figsize=(50, 25))
        plt.title("Gantt Chart")
        plt.xlabel("Time")
        plt.ylabel("Machine")
        plt.yticks(
            range(self.num_machines), [str(x) for x in range(1, self.num_machines + 1)]
        )
        plt.grid(True)

        # bs, jobs, ma
        durations = td["durations"][idx]
        machines = td["machines"][idx]
        dur_along_machines = durations.gather(1, machines)
        starts = td["start_times"][idx].gather(1, machines)

        for machine in range(self.num_machines):
            for job in range(self.num_jobs):
                task = machines[job, machine]
                # job_num = task // self.num_machines
                # assert job == job_num
                plt.barh(
                    y=machine,
                    left=starts[job, machine],
                    width=dur_along_machines[job, machine],
                    color="C{}".format(job),
                    label="Job {}".format(job),
                    alpha=0.6,
                )
                # add the duration on the bar
                text_to_add = [
                    f"Task {task}:",
                    f"{starts[job, machine].item()}",
                    f"{(starts[job, machine] + dur_along_machines[job, machine]).item()}",
                ]
                for i, line in enumerate(text_to_add):
                    plt.text(
                        starts[job, machine],
                        machine - i * 0.2,
                        line,
                        ha="left",
                        va="bottom",
                    )
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(
            by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.tight_layout()
        plt.show()


def end_time_lb(td: TensorDict) -> torch.Tensor:
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
    # bs, jobs, ma
    durations = td["durations"].clone()
    # bs, jobs, ma
    ending_times = td["end_times"].clone()
    # bs, jobs
    op_scheduled = td["finished_mark"].any(2)
    last_op_idx = ending_times[torch.where(op_scheduled)].argmax(1)
    last_op_complete_idx = (*torch.where(op_scheduled), last_op_idx)

    # set the duration of already started operations to 0
    durations[torch.where(ending_times != 0)] = 0
    durations[last_op_complete_idx] = ending_times[last_op_complete_idx]
    # calculate the cumulative sum of the durations of the operations on the same machine
    temp2 = torch.cumsum(durations, dim=2)
    # set the cumulative sum of the durations of already started operations to 0
    temp2[torch.where(ending_times != 0)] = 0
    ret = ending_times + temp2
    return ret


if __name__ == "__main__":
    torch.manual_seed(123)
    env = JSSPEnv(num_jobs=5, num_machines=6)
    td = env._reset(batch_size=100)
    while not td["done"].all():
        logit = torch.zeros(*td["action_mask"].shape)
        logit[~td["action_mask"]] = -torch.inf
        actions = torch.multinomial(torch.softmax(logit, 1), 1).squeeze(1)
        td.set("action", actions)
        td = env.step(td)["next"]
    env.render(td, 0)
