from typing import Tuple

import torch


def get_job_ops_mapping(
    start_op_per_job: torch.Tensor, end_op_per_job: torch.Tensor
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
    n_ops_max = int(end_op_per_job.max() + 1)

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
