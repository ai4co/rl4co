import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import ttest_rel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rl4co import utils
from rl4co.data.dataset import ExtraKeyDataset, tensordict_collate_fn

log = utils.get_pylogger(__name__)


class REINFORCEBaseline(nn.Module):
    """Base class for REINFORCE baselines"""

    def __init__(self, *args, **kw):
        super().__init__()
        pass

    def wrap_dataset(self, dataset, *args, **kw):
        """Wrap dataset with baseline-specific functionality"""
        return dataset

    def eval(self, td, reward):
        """Evaluate baseline"""
        pass

    def epoch_callback(self, *args, **kw):
        """Callback at the end of each epoch
        For example, update baseline parameters and obtain baseline values
        """
        pass

    def setup(self, *args, **kw):
        """To be called before training during setup phase
        This follow PyTorch Lightning's setup() convention
        """
        pass


class NoBaseline(REINFORCEBaseline):
    def eval(self, td, reward):
        return 0, 0  # No baseline, no neg_los


class SharedBaseline(REINFORCEBaseline):
    def eval(self, td, reward, on_dim=1):  # e.g. [batch, pomo, ...]
        return reward.mean(dim=on_dim, keepdims=True), 0


class ExponentialBaseline(REINFORCEBaseline):
    def __init__(self, beta=0.8):
        super(REINFORCEBaseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, td, reward):
        if self.v is None:
            v = reward.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * reward.mean()
        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss


class WarmupBaseline(REINFORCEBaseline):
    def __init__(
        self,
        baseline,
        n_epochs=1,
        warmup_exp_beta=0.8,
    ):
        super(REINFORCEBaseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset, *args, **kw):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset, *args, **kw)
        return self.warmup_baseline.wrap_dataset(dataset, *args, **kw)

    def setup(self, *args, **kw):
        self.baseline.setup(*args, **kw)

    def eval(self, td, reward):
        if self.alpha == 1:
            return self.baseline.eval(td, reward)
        if self.alpha == 0:
            return self.warmup_baseline.eval(td, reward)
        v_b, l_b = self.baseline.eval(td, reward)
        v_wb, l_wb = self.warmup_baseline.eval(td, reward)
        # Return convex combination of baseline and of loss
        return self.alpha * v_b + (1 - self.alpha) * v_wb, self.alpha * l_b + (
            1 - self.alpha * l_wb
        )

    def epoch_callback(self, *args, **kw):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(*args, **kw)
        self.alpha = (kw["epoch"] + 1) / float(self.n_epochs)
        if kw["epoch"] < self.n_epochs:
            log.info("Set warmup alpha = {}".format(self.alpha))


class CriticBaseline(REINFORCEBaseline):
    def __init__(self, critic, **unused_kw):
        super(CriticBaseline, self).__init__()
        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        # detach v since actor should not backprop through baseline, only for neg_loss
        return v.detach(), -F.mse_loss(v, c.detach())


class RolloutBaseline(REINFORCEBaseline):
    def __init__(self, bl_alpha=0.05, progress_bar=False):
        super(RolloutBaseline, self).__init__()
        self.bl_alpha = bl_alpha
        self.progress_bar = progress_bar

    def setup(self, *args, **kw):
        self._update_model(*args, **kw)

    def _update_model(
        self, model, env, batch_size=64, device="cpu", dataset_size=None, dataset=None
    ):
        self.model = copy.deepcopy(model).to(device)
        if dataset is None:
            log.info("Creating evaluation dataset for rollout baseline")
            self.dataset = env.dataset(batch_size=[dataset_size])

        log.info("Evaluating baseline model on evaluation dataset")
        self.bl_vals = (
            self.rollout(self.model, env, batch_size, device, self.dataset).cpu().numpy()
        )
        self.mean = self.bl_vals.mean()

    def eval(self, td, reward):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            reward = self.model(td)["reward"]
        return reward, 0

    def epoch_callback(
        self, model, env, batch_size=64, device="cpu", epoch=None, dataset_size=None
    ):
        """Challenges the current baseline with the model and replaces the baseline model if it is improved"""
        log.info("Evaluating candidate model on evaluation dataset")
        candidate_vals = self.rollout(model, env, batch_size, device).cpu().numpy()
        candidate_mean = candidate_vals.mean()

        log.info(
            "Candidate mean: {:.3f}, Baseline mean: {:.3f}".format(
                candidate_mean, self.mean
            )
        )
        if candidate_mean - self.mean > 0:
            # Calc p value with inverse logic (costs)
            t, p = ttest_rel(-candidate_vals, -self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            log.info("p-value: {:.3f}".format(p_val))
            if p_val < self.bl_alpha:
                log.info("Updating baseline")
                self._update_model(model, env, batch_size, device, dataset_size)

    def rollout(self, model, env=None, batch_size=64, device="cpu", dataset=None):
        """Rollout the model on the given dataset"""
        # if dataset is None, use the dataset of the baseline
        dataset = self.dataset if dataset is None else dataset

        model.eval()
        model = model.to(device)

        def eval_model(batch):
            with torch.no_grad():
                batch = env.reset(batch.to(device))
                return model(batch, decode_type="greedy")["reward"].data.cpu()

        dl = DataLoader(dataset, batch_size=batch_size, collate_fn=tensordict_collate_fn)

        retval = torch.cat(
            [eval_model(batch) for batch in tqdm(dl, disable=not self.progress_bar)], 0
        )
        return retval

    def wrap_dataset(self, dataset, env, batch_size=64, device="cpu", **kw):
        """Wrap the dataset in a baseline dataset"""
        rewards = (
            self.rollout(self.model, env, batch_size, device, dataset=dataset)
            .detach()
            .cpu()
        )
        return ExtraKeyDataset(dataset, rewards)

    def __getstate__(self):
        """Do not include datasets in state to avoid pickling issues"""
        state = self.__dict__.copy()
        try:
            del state["dataset"]
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        """Restore datasets after unpickling. Will be restored in setup"""
        self.__dict__.update(state)
        self.dataset = None
