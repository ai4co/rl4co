import copy
from scipy.stats import ttest_rel

import torch
import torch.nn as nn


class REINFORCEBaseline(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def eval(self, td, cost):
        pass

    def epoch_callback(self, model, td, epoch):
        pass


class NoBaseline(REINFORCEBaseline):
    def eval(self, td, cost):
        return 0, 0  # No baseline, no loss


class SharedBaseline(REINFORCEBaseline):
    def eval(self, td, cost):
        return 0, cost.mean(dim=1).view(-1, 1)


class ExponentialBaseline(REINFORCEBaseline):
    def __init__(self, beta):
        super(REINFORCEBaseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, td, cost):
        if self.v is None:
            v = cost.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * cost.mean()
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

    def setup(self, model, dl, env=None):
        self.baseline.setup(model, dl, env)

    def eval(self, td, cost):
        if self.alpha == 1:
            return self.baseline.eval(td, cost)
        if self.alpha == 0:
            return self.warmup_baseline.eval(td, cost)
        v, l = self.baseline.eval(td, cost)
        vw, lw = self.warmup_baseline.eval(td, cost)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (
            1 - self.alpha * lw
        )

    def epoch_callback(self, model, dl, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, dl, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))


class RolloutBaseline(REINFORCEBaseline):
    def __init__(self, bl_alpha=0.05):
        super(RolloutBaseline, self).__init__()
        self.bl_alpha = bl_alpha

    def setup(self, model, dl, env=None):
        self._update_model(model, dl, env)

    def _update_model(self, model, dl, env=None):
        device = next(model.parameters()).device
        self.model = copy.deepcopy(model).to(device)
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = -self.rollout(model, dl, env).cpu().numpy()
        self.mean = self.bl_vals.mean()

    def eval(self, td, cost):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            out = -self.model(td)["reward"]
        return out, 0

    def epoch_callback(self, model, dl, epoch=None, env=None):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        """
        print("\nEvaluating candidate model on evaluation dataset")
        candidate_vals = -self.rollout(model, dl, env).cpu().numpy()
        candidate_mean = candidate_vals.mean()

        print("Candidate mean: {}".format(candidate_mean))
        print("Baseline mean: {}".format(self.mean))
        print("----------------------")
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("\np-value: {}".format(p_val))
            if p_val < self.bl_alpha:
                print("\nUpdating baseline")
                self._update_model(model, dl, env)

    def rollout(self, model, dl, env=None):
        """
        Rollout the model on the given dataset.
        """
        env_fn = lambda x: x if env is None else env.reset(init_observation=x)
        with torch.no_grad():
            model.eval()
            device = next(
                model.parameters()
            ).device  # same as transfer_batch_to_device in lightning, may be cleaner but hey
            rewards = [
                model(env_fn(batch.to(device)), decode_type="greedy")["reward"]
                for batch in dl
            ]
        return torch.cat(rewards, dim=0)
