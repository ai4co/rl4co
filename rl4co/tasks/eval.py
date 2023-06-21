import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rl4co.data.dataset import tensordict_collate_fn
from rl4co.models.zoo.pomo.augmentations import (
    StateAugmentation as Dihedral8StateAugmentation,
)
from rl4co.models.zoo.symnco.augmentations import (
    StateAugmentation as SymmetricStateAugmentation,
)
from rl4co.utils.ops import batchify, gather_by_index, unbatchify


def check_unused_kwargs(class_, kwargs):
    if len(kwargs) > 0 and not (len(kwargs) == 1 and "progress" in kwargs):
        print(f"Warning: {class_.__class__.__name__} does not use kwargs {kwargs}")


class EvalBase:
    """Base class for evaluation

    Args:
        env: Environment
        progress: Whether to show progress bar
        **kwargs: Additional arguments (to be implemented in subclasses)
    """

    name = "base"

    def __init__(self, env, progress=True, **kwargs):
        check_unused_kwargs(self, kwargs)
        self.env = env
        self.progress = progress

    def __call__(self, policy, dataloader, **kwargs):
        """Evaluate the policy on the given dataloader with **kwargs parameter
        self._inner is implemented in subclasses and returns actions and rewards
        """

        # Collect timings for evaluation (more accurate than timeit)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with torch.no_grad():
            rewards_list = []
            actions_list = []

            for batch in tqdm(
                dataloader, disable=not self.progress, desc=f"Running {self.name}"
            ):
                td = batch.to(next(policy.parameters()).device)
                td = self.env.reset(td)
                actions, rewards = self._inner(policy, td, **kwargs)
                rewards_list.append(rewards)
                actions_list.append(actions)

            rewards = torch.cat(rewards_list)

            # Padding: pad actions to the same length with zeros
            max_length = max(action.size(-1) for action in actions)
            actions = torch.cat(
                [
                    torch.nn.functional.pad(action, (0, max_length - action.size(-1)))
                    for action in actions
                ],
                0,
            )

        end_event.record()
        torch.cuda.synchronize()
        inference_time = start_event.elapsed_time(end_event)

        tqdm.write(f"Mean reward for {self.name}: {rewards.mean():.4f}")
        tqdm.write(f"Time: {inference_time/1000:.4f}s")

        # Empty cache
        torch.cuda.empty_cache()

        return {
            "actions": actions.cpu(),
            "rewards": rewards.cpu(),
            "inference_time": inference_time,
            "avg_reward": rewards.cpu().mean(),
        }

    def _inner(self, policy, td):
        """Inner function to be implemented in subclasses.
        This function returns actions and rewards for the given policy
        """
        raise NotImplementedError("Implement in subclass")


class GreedyEval(EvalBase):
    """Evaluates the policy using greedy decoding and single trajectory"""

    name = "greedy"

    def __init__(self, env, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))

    def _inner(self, policy, td):
        out = policy(
            td.clone(),
            decode_type="greedy",
            num_starts=0,
            return_actions=True,
        )
        rewards = self.env.get_reward(td, out["actions"])
        return out["actions"], rewards


class AugmentationEval(EvalBase):
    """Evaluates the policy via N state augmentations
    `force_dihedral_8` forces the use of 8 augmentations (rotations and flips) as in POMO
    https://en.wikipedia.org/wiki/Examples_of_groups#dihedral_group_of_order_8

    Args:
        num_augment (int): Number of state augmentations
        force_dihedral_8 (bool): Whether to force the use of 8 augmentations
    """

    name = "augmentation"

    def __init__(self, env, num_augment=8, force_dihedral_8=False, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))

        assert not (
            num_augment != 8 and force_dihedral_8
        ), "Cannot force dihedral when num_augment != 8"
        if num_augment == 8 and not force_dihedral_8:
            self.augmentation = Dihedral8StateAugmentation(
                env.name, num_augment=num_augment
            )
        else:
            self.augmentation = SymmetricStateAugmentation(
                env.name, num_augment=num_augment
            )

    def _inner(self, policy, td, num_augment=None):
        if num_augment is None:
            num_augment = self.augmentation.num_augment
        td_init = td.clone()
        td = self.augmentation(td, num_augment=num_augment)
        out = policy(td.clone(), decode_type="greedy", num_starts=0, return_actions=True)

        # Move into batches and compute rewards
        rewards = self.env.get_reward(batchify(td_init, num_augment), out["actions"])
        rewards = unbatchify(rewards, num_augment)
        actions = unbatchify(out["actions"], num_augment)

        # Get best reward and corresponding action
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards

    @property
    def num_augment(self):
        return self.augmentation.num_augment


class SamplingEval(EvalBase):
    """Evaluates the policy via N samples from the policy

    Args:
        samples (int): Number of samples to take
        softmax_temp (float): Temperature for softmax sampling. The higher the temperature, the more random the sampling
    """

    name = "sampling"

    def __init__(self, env, samples, softmax_temp=None, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))

        self.samples = samples
        self.softmax_temp = softmax_temp

    def _inner(self, policy, td):
        td = batchify(td, self.samples)
        out = policy(
            td.clone(),
            decode_type="sampling",
            num_starts=0,
            return_actions=True,
            softmax_temp=self.softmax_temp,
        )

        # Move into batches and compute rewards
        rewards = self.env.get_reward(td, out["actions"])
        rewards = unbatchify(rewards, self.samples)
        actions = unbatchify(out["actions"], self.samples)

        # Get the best reward and action for each sample
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards


class GreedyMultiStartEval(EvalBase):
    """Evaluates the policy via `num_starts` greedy multistarts samples from the policy

    Args:
        num_starts (int): Number of greedy multistarts to use
    """

    name = "greedy_multistart"

    def __init__(self, env, num_starts=None, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))

        assert num_starts is not None, "Must specify num_starts"
        self.num_starts = num_starts

    def _inner(self, policy, td):
        td_init = td.clone()
        out = policy(
            td.clone(),
            decode_type="greedy_multistart",
            num_starts=self.num_starts,
            return_actions=True,
        )

        # Move into batches and compute rewards
        td = batchify(td_init, self.num_starts)
        rewards = self.env.get_reward(td, out["actions"])
        rewards = unbatchify(rewards, self.num_starts)
        actions = unbatchify(out["actions"], self.num_starts)

        # Get the best trajectories
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards


class GreedyMultiStartAugmentEval(EvalBase):
    """Evaluates the policy via `num_starts` samples from the policy
    and `num_augment` augmentations of each sample.`
    `force_dihedral_8` forces the use of 8 augmentations (rotations and flips) as in POMO
    https://en.wikipedia.org/wiki/Examples_of_groups#dihedral_group_of_order_8

    Args:
        num_starts: Number of greedy multistart samples
        num_augment: Number of augmentations per sample
        force_dihedral_8: If True, force the use of 8 augmentations (rotations and flips) as in POMO
    """

    name = "greedy_multistart_augment"

    def __init__(
        self, env, num_starts=None, num_augment=8, force_dihedral_8=False, **kwargs
    ):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))

        assert num_starts is not None, "Must specify num_starts"
        self.num_starts = num_starts
        assert not (
            num_augment != 8 and force_dihedral_8
        ), "Cannot force dihedral 8 when num_augment != 8"
        if num_augment == 8 and not force_dihedral_8:
            self.augmentation = Dihedral8StateAugmentation(
                env.name, num_augment=num_augment
            )
        else:
            self.augmentation = SymmetricStateAugmentation(
                env.name, num_augment=num_augment
            )

    def _inner(self, policy, td, num_augment=None):
        if num_augment is None:
            num_augment = self.augmentation.num_augment

        td_init = td.clone()

        td = self.augmentation(td, num_augment=num_augment)
        out = policy(
            td.clone(),
            decode_type="greedy_multistart",
            num_starts=self.num_starts,
            return_actions=True,
        )

        # Move into batches and compute rewards
        td = batchify(td_init, (num_augment, self.num_starts))
        rewards = self.env.get_reward(td, out["actions"])
        rewards = unbatchify(rewards, self.num_starts * num_augment)
        actions = unbatchify(out["actions"], self.num_starts * num_augment)

        # Get the best trajectories
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards

    @property
    def num_augment(self):
        return self.augmentation.num_augment


def get_automatic_batch_size(eval_fn, start_batch_size=8192, max_batch_size=4096):
    """Automatically reduces the batch size based on the eval function

    Args:
        eval_fn: The eval function
        start_batch_size: The starting batch size. This should be the theoretical maximum batch size
        max_batch_size: The maximum batch size. This is the practical maximum batch size
    """
    batch_size = start_batch_size

    effective_ratio = 1

    if hasattr(eval_fn, "num_starts"):
        batch_size = batch_size // (eval_fn.num_starts // 10)
        effective_ratio *= eval_fn.num_starts // 10
    if hasattr(eval_fn, "num_augment"):
        batch_size = batch_size // eval_fn.num_augment
        effective_ratio *= eval_fn.num_augment
    if hasattr(eval_fn, "samples"):
        batch_size = batch_size // eval_fn.samples
        effective_ratio *= eval_fn.samples

    batch_size = min(batch_size, max_batch_size)
    # get closest integer power of 2
    batch_size = 2 ** int(np.log2(batch_size))

    print(f"Effective batch size: {batch_size} (ratio: {effective_ratio})")

    return batch_size


def evaluate_policy(
    env,
    policy,
    dataset,
    method="greedy",
    batch_size=None,
    max_batch_size=4096,
    start_batch_size=8192,
    auto_batch_size=True,
    save_results=False,
    save_fname="results.npz",
    **kwargs,
):
    num_loc = getattr(env, "num_loc", None)

    methods_mapping = {
        "greedy": {"func": GreedyEval, "kwargs": {}},
        "sampling": {
            "func": SamplingEval,
            "kwargs": {"samples": 100, "softmax_temp": 1.0},
        },
        "greedy_multistart": {
            "func": GreedyMultiStartEval,
            "kwargs": {"num_starts": num_loc},
        },
        "augment_dihedral_8": {
            "func": AugmentationEval,
            "kwargs": {"num_augment": 8, "force_dihedral_8": True},
        },
        "augment": {"func": AugmentationEval, "kwargs": {"num_augment": 8}},
        "greedy_multistart_augment_dihedral_8": {
            "func": GreedyMultiStartAugmentEval,
            "kwargs": {
                "num_augment": 8,
                "force_dihedral_8": True,
                "num_starts": num_loc,
            },
        },
        "greedy_multistart_augment": {
            "func": GreedyMultiStartAugmentEval,
            "kwargs": {"num_augment": 8, "num_starts": num_loc},
        },
    }

    assert method in methods_mapping, "Method {} not found".format(method)

    # Set up the evaluation function
    eval_settings = methods_mapping[method]
    func, kwargs_ = eval_settings["func"], eval_settings["kwargs"]
    # subsitute kwargs with the ones passed in
    kwargs_.update(kwargs)
    kwargs = kwargs_
    eval_fn = func(env, **kwargs)

    if auto_batch_size:
        assert (
            batch_size is None
        ), "Cannot specify batch_size when auto_batch_size is True"
        batch_size = get_automatic_batch_size(
            eval_fn, max_batch_size=max_batch_size, start_batch_size=start_batch_size
        )
        print("Using automatic batch size: {}".format(batch_size))

    # Set up the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=tensordict_collate_fn,
    )

    # Run evaluation
    retvals = eval_fn(policy, dataloader)

    # Save results
    if save_results:
        print("Saving results to {}".format(save_fname))
        np.savez(save_fname, **retvals)

    return retvals
