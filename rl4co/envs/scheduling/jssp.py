from typing import Optional, Tuple

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec

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
        num_jobs,
        num_machines,
        low=1,
        high=99,
        et_normalize_coef=1000,
        rewardscale=0.0,
        init_quality_flag=False,
        stepwise_reward=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._low = low
        self._high = high
        self._et_normalize_coef = et_normalize_coef
        self._rewardscale = rewardscale
        self._init_quality_flag = init_quality_flag
        self.stepwise_reward = stepwise_reward

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_tasks = self.num_jobs * self.num_machines

        # create specs for observation and action
        #self._make_spec()

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
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)
        # (bs,)
        job_idx = td["action"]
        # action is an int 0 - num_jobs
        # the actual action is the operation id of the selected job; shape=(bs,)
        task_idx = td["next_op"][batch_idx, job_idx]
        col = task_idx % self.num_machines
        ma_idx = torch.where(td["machines"][batch_idx, job_idx] == col[:, None])[1]

        # mark scheduled operations
        td["finished_mark"][batch_idx, job_idx, col] = 1
        op_duration = td["durations"][batch_idx, job_idx, col]

        # (bs,) first operation of job has no predecessor
        previous_op_in_job = ~torch.isin(task_idx, self.first_task)
        # gather ending times of previous operation if exists 
        # (NOTE clipping only effects first operation which has no predecessor)
        previous_op_end_time = td["end_times"][batch_idx, job_idx, torch.clip(col-1, 0)]
        previous_op_end_time1 = td["end_times"].reshape(*batch_size, -1)[batch_idx, torch.clip(task_idx-1, 0)]
        assert torch.all(previous_op_end_time[previous_op_in_job] == previous_op_end_time1[previous_op_in_job])
        # (bs,)
        op_ready_time = torch.where(
            previous_op_in_job,
            previous_op_end_time,
            # if no preceeding operation exists, op can start directly,
            torch.zeros((*batch_size,), dtype=torch.float32, device=self.device)
        )
        # sort the schedule in accordance to the machine indices; shape=(bs, jobs, ma) 
        ma_start_times = td["start_times"].gather(2, td["machines"])
        ma_end_times = td["end_times"].gather(2, td["machines"])
        # get start and end times of selected machine; shape=(bs, num_jobs)
        start_chosen_ma = ma_start_times[batch_idx, :, ma_idx]
        end_chosen_ma = ma_end_times[batch_idx, :, ma_idx]
        # get the time the machine is idle again; shape=(bs,)
        ma_ready_time = end_chosen_ma.max(1).values.clip(0)
        # we may put an op before another op, if it is ready before the other one starts
        possible_positions = op_ready_time[:, None] < start_chosen_ma
        eligible4ls = possible_positions.any(1)

        # (bs, jobs)
        lag1_end_times_ma = torch.cat(
            (torch.zeros_like(end_chosen_ma[:, :1]), end_chosen_ma), dim=1
        )[batch_idx, :-1]
        # determine idle gaps on selected machine
        gaps = start_chosen_ma - lag1_end_times_ma
        gaps[~possible_positions] = -torch.inf
        sufficient_gaps = gaps > op_duration[:, None]
        # get the first index of the gap; shape=(bs,)
        min_pos = sufficient_gaps.to(torch.int32).max(1).indices
        min_pos_start = lag1_end_times_ma.gather(1, min_pos[:, None]).squeeze(1)
        # (bs, )
        left_shift_possible = torch.logical_and(eligible4ls, sufficient_gaps.any(1))
        start_times = torch.where(
            left_shift_possible,
            min_pos_start,
            # if no gap is big enough, we have to put the op at the end of schedule
            torch.max(op_ready_time, ma_ready_time) 
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
        LBs = torch.where(td["finished_mark"].to(torch.bool), td["end_times"], td["durations"]).cumsum(2)

        done = torch.all(td["action_mask"]==0, 1)
        if self.stepwise_reward:
            reward = -(LBs.amax(dim=(1,2)) - old_lbs.amax(dim=(1,2)))
        elif done.all():
            # get makespan
            reward = td["end_times"].amax(dim=(1,2))

        return td.update({
            "adjacency": adjacency,
            "lower_bounds": LBs,
            "reward": reward,
            "done": done
        })

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
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
        # NOTE in original implementation authors dont use succeesors
        # initialize adj matrix of successors. We leave an offset of -1, since the first
        # operation has no successors: shape=(num_tasks, num_tasks)
        successor_adj = torch.diag_embed(
            torch.ones(self.num_tasks - 1, device=self.device), offset=1
        )
        # first operation of jobs do not have successors
        successor_adj[self.last_task] = 0
        # ---------------

        # self loops and adding all neighbors in final adjacency matrix
        self_loop = torch.eye(self.num_tasks, dtype=torch.float32, device=self.device)
        adjacency = torch.unsqueeze(self_loop + predecessors_adj, 0).repeat(*batch_size, 1, 1)
        # initialize features
        # (bs, jobs, ma)
        LBs = torch.cumsum(durations, dim=-1)
        # (bs, jobs, ma)
        finished_mark = torch.zeros_like(durations)
        # (bs, num_tasks, 2)

        # TODO move feature logic into init_embed part
        # features = torch.concatenate(
        #     [
        #         # (bs, num_tasks, 1)
        #         LBs.reshape(batch_size, self.num_tasks, 1) / self._et_normalize_coef,
        #         # (bs, num_tasks, 1)
        #         finished_mark.reshape(batch_size, self.num_tasks, 1),
        #     ],
        #     dim=1,
        # )
        
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

        # matrices with the indexes of the operations executed on each machine.
        # For example, if operations_on_machines[1, 2] = 3, it means that
        # the machine of index 1 executes the operation 3 at position 2.
        # shape (num_machines, num_jobs)
        ops_on_ma = (
            torch.full_like(durations.transpose(-2,-1), fill_value=-self.num_jobs, dtype=torch.int32)
        )

        tensordict = TensorDict(
            {
                "adjacency": adjacency,
                "machines": machines,
                "start_times": start_times,
                "end_times": ending_times,
                "ops_on_ma": ops_on_ma,
                "lower_bounds": LBs,
                "finished_mark": finished_mark,
                "next_op": next_op,
                "durations": durations,
                "action_mask": mask,
                "done": torch.full((*batch_size,), False),
                "reward": torch.empty((*batch_size,), dtype=torch.float32, device=self.device)
            },
            batch_size=batch_size,
        )
        return tensordict
    
    def generate_data(self, batch_size):
        # randomly generate processing times for each operation (job-machine combination)
        # (bs, jobs, ma)
        proc_times = torch.randint(
            low=self._low, 
            high=self._high, 
            size=(*batch_size, self.num_jobs, self.num_machines), 
            dtype=torch.float32,
            device=self.device
        )
        # the machines tensor of shape=(bs, jobs, ma) specifies which operation is executed on
        # which machine, e.g. machines[:, 0, 1] = 2 means that operation=2 of job=0 is executed
        # on machine=1
        machines = torch.rand((*batch_size, self.num_jobs, self.num_machines)).argsort(dim=-1)
        # machines += self.first_task[None, :, None]
        return TensorDict(
            {"durations": proc_times, "machines": machines}, batch_size=batch_size
        )


    def get_reward(self, td, actions):
        reward = td["reward"]
        if not td["done"].all() and not self.stepwise_reward:
            raise AttributeError("Use stepwise_reward=True in JSSPEnv if you require the reward after every step")
        return reward

    def update_adjacency(self, td: TensorDict, job_idx, task_idx, ma_idx, previous_op_in_job, left_shift_possible):
        # define batch_idx
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)
        # get machine predecessor and successor nodes of operation, i.e. ops
        # that are processed directly before and after the selected operation
        pred_of_op, succ_of_op = self.get_machine_precedence(td, batch_idx, job_idx, task_idx, ma_idx)

        # adj matrix NOTE i dont get this ########
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
    

    def get_machine_precedence(self, td: TensorDict, batch_idx, job_idx, task_idx, ma_idx):
        """This function determines the precedence relationships of operations on the same
        machine given the current schedule. Given a partial schedule of start times, we 
        first align these start_times per machine type. Then, we determine, given these start_times, 
        which job is executed first, second and so on and after that determine the task_id that
        corresponds to the job-machine combo. 
        ------------
        NOTE
        In the original implementation the authors track the machine schedule in a numpy array,
        where schedule[1,2]=3 specifies that machine=1 processes task 3 in position=2. This does not
        work in torch, since we cannot simply insert tasks at different positions along the batches 
        (without using a for loop). This solution is imo cleaner.
        ------------
        """

        # transform the 'operations schedule' into a 'machine schedule', i.e. sort
        # the last dimension according to the machine the respective op is executed on
        ma_end_times = td["end_times"].gather(2, td["machines"])
        # mask not scheduled ops
        ma_end_times[ma_end_times==0] = torch.inf
        # (bs, jobs, ma) --> sort jobs in accordence to their position in current schedule
        ma_precedence_order = ma_end_times.argsort(1)
        # get position of the selected operation in current schedule
        op_pos_on_ma = torch.where(ma_precedence_order[batch_idx, :, ma_idx] == job_idx[:, None])[1]
        # get the id of the job, directly preceeding the selected job. Select the job itself
        # if it has no predecessor (i.e. scheduled on first position in machine schedule)
        pred_job = torch.where(
            op_pos_on_ma == 0,
            job_idx,
            ma_precedence_order[batch_idx, torch.clip(op_pos_on_ma-1, 0), ma_idx]
        )
        # get the id of the job, directly succeeding the selected job. Select the job itself
        # if it has no successor (i.e. scheduled on last position in machine schedule)
        succ_job = torch.where(
            op_pos_on_ma == self.num_jobs-1,
            job_idx,
            ma_precedence_order[batch_idx, torch.clip(op_pos_on_ma+1, max=self.num_jobs-1), ma_idx]
        )
        # to uniquely identify the tasks, add the first_task increment to the task indices in machines tensor
        tasks_on_ma = td["machines"] + self.first_task[None, :, None]
        # mask operations that are not scheduled yet
        tasks_on_ma[~td["finished_mark"].gather(2, td["machines"]).to(torch.bool)] = -1
        # get the operation that corresponds to the job-machine combo
        pred_of_op = tasks_on_ma[batch_idx, pred_job, ma_idx]
        succ_of_op = tasks_on_ma[batch_idx, succ_job, ma_idx]
        # mask successors that have not been scheduled yet
        succ_of_op = torch.where(
            succ_of_op < 0,
            task_idx, 
            succ_of_op
        )
        return pred_of_op, succ_of_op


if __name__ == "__main__":
    torch.manual_seed(123) 
    env = JSSPEnv(num_jobs=4, num_machines=3)
    td = env._reset(batch_size=20)
    while not td["done"].all():
        logit = torch.zeros(*td["action_mask"].shape)
        logit[~td["action_mask"]]= -torch.inf
        actions = torch.multinomial(torch.softmax(logit, 1), 1).squeeze(1)
        td.set("action", actions)
        td = env.step(td)["next"]

    