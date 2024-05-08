from typing import Tuple

import torch

from einops import rearrange


def get_job_ops_mapping(
    start_op_per_job: torch.Tensor, end_op_per_job: torch.Tensor, n_ops_max: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implements a mapping function from operations to jobs

    :param torch.Tensor start_op_per_job: index of first operation of each job
    :param torch.Tensor end_op_per_job: index of last operation of each job
    :return Tuple[torch.Tensor, torch.Tensor]:
        1st.) index mapping (bs, num_ops): [0,0,1,1,1] means that first two operations belong to job 0
        2st.) binary mapping (bs, num_jobs, num_ops): [[1,1,0], [0,0,1]] means that first two operations belong to job 0
    """
    device = end_op_per_job.device
    end_op_per_job = end_op_per_job.clone()

    bs, num_jobs = end_op_per_job.shape

    # in order to avoid shape conflicts, set the end operation id to the id of max_ops (all batches have same #ops)
    end_op_per_job[:, -1] = n_ops_max - 1

    # here we will generate the operations-job mapping:
    # Therefore we first generate a sequence of operation ids and expand it the the size of the mapping matrix:
    # (bs, jobs, max_ops)
    ops_seq_exp = torch.arange(n_ops_max, device=device)[None, None].expand(
        bs, num_jobs, -1
    )
    # (bs, jobs, max_ops)  # expanding start and end operation ids
    end_op_per_job_exp = end_op_per_job[..., None].expand_as(ops_seq_exp)
    start_op_per_job_exp = start_op_per_job[..., None].expand_as(ops_seq_exp)
    # given ids of start and end operations per job, this generates the mapping of ops to jobs
    # (bs, jobs, max_ops)
    ops_job_map = torch.nonzero(
        (ops_seq_exp <= end_op_per_job_exp) & (ops_seq_exp >= start_op_per_job_exp)
    )
    # (bs, max_ops)
    ops_job_map = torch.stack(ops_job_map[:, 1].split(n_ops_max), dim=0)

    # we might also want a binary mapping / adjacency matrix connecting jobs to operations
    # (bs, num_jobs, num_ops)
    ops_job_bin_map = torch.scatter_add(
        input=ops_job_map.new_zeros((bs, num_jobs, n_ops_max)),
        dim=1,
        index=ops_job_map.unsqueeze(1),
        src=ops_job_map.new_ones((bs, num_jobs, n_ops_max)),
    )

    return ops_job_map, ops_job_bin_map


def get_action_mask(td):
    # helper function to combine no_op and ma-job mask to mask over full action space; (bs, jobs+1, ma)
    return torch.cat((td["no_op_mask"].unsqueeze(1), td["action_mask"]), dim=1)


def get_flat_action_mask(td):
    """This function creates a mask of shape (bs, j*m) specifying which job machine combination
    (including 'no job') cannot be selected using the no_op_mask and the (j x m)
    action_mask
    """
    # (bs, 1)
    no_op_mask = td["no_op_mask"].all(1, keepdims=True)
    mask = rearrange(td["action_mask"], "bs j m -> bs (j m)")
    mask = torch.cat((no_op_mask, mask), dim=1)
    return mask


def get_job_mask(td):
    """This function creates a mask of shape (bs, 1 + jobs) specifying which job
    (including 'no job') cannot be selected using the no_op_mask and the (j x m)
    action_mask
    """
    # (bs, 1); mask no op if all machines are idle (no job is processed atm)
    no_op_mask = td["no_op_mask"].all(1, keepdims=True)
    # if a job cannot be processed on any machine, mask it; shape=(bs, jobs)
    job_mask = td["action_mask"].all(2)
    # (bs, 1 + jobs)
    return torch.cat((no_op_mask, job_mask), dim=1)


def get_machine_mask(td):
    """This function creates a mask of shape (bs, 1 + ma) specifying which machine
    (including 'dummy machine') cannot be selected using the no_op_mask and the (j x m)
    action_mask
    """
    bs = td.size(0)
    num_mas = td["proc_times"].size(1)
    # (bs, 1+jobs, ma)
    full_mask = get_action_mask(td)
    # (bs, ma); first get a mask specifying which machine cannot process selected job
    ma_mask = full_mask.gather(
        1, td["selected_job"][:, None, None].expand(bs, 1, num_mas) + 1
    ).view(bs, num_mas)
    # (bs, 1); if no job is selected (waiting operation), dummy machine is only feasible option
    dummy_ma_mask = torch.where(td["selected_job"] == -1, False, True)[:, None]
    # (bs, 1 + ma); concat
    return torch.cat((dummy_ma_mask, ma_mask), dim=1)


def get_no_op_mask_per_ma(td):
    action_mask = td["action_mask"]
    no_op_to_process = torch.logical_or(
        (td["busy_until"] > td["time"][:, None]), action_mask.all(1)
    )
    no_op_mask = torch.logical_and(~no_op_to_process, ~td["done"])
    return no_op_mask
