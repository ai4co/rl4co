from rl4co.models.rl.a2c.baseline import CriticBaseline
from rl4co.models.rl.reinforce.baselines import (
    ExponentialBaseline,
    NoBaseline,
    REINFORCEBaseline,
    RolloutBaseline,
    SharedBaseline,
    WarmupBaseline,
)

REINFORCE_BASELINES_REGISTRY = {
    "no": NoBaseline,
    "shared": SharedBaseline,
    "exponential": ExponentialBaseline,
    "critic": CriticBaseline,
    "rollout_only": RolloutBaseline,
    "warmup": WarmupBaseline,
}


def get_reinforce_baseline(name, **kw):
    """Get a REINFORCE baseline by name
    The rollout baseline default to warmup baseline with one epoch of
    exponential baseline and the greedy rollout
    """
    if name == "warmup":
        inner_baseline = kw.get("baseline", "rollout")
        if not isinstance(inner_baseline, REINFORCEBaseline):
            inner_baseline = get_reinforce_baseline(inner_baseline, **kw)
        return WarmupBaseline(inner_baseline, **kw)
    elif name == "rollout":
        warmup_epochs = kw.get("n_epochs", 1)
        warmup_exp_beta = kw.get("exp_beta", 0.8)
        bl_alpha = kw.get("bl_alpha", 0.05)
        return WarmupBaseline(
            RolloutBaseline(bl_alpha=bl_alpha), warmup_epochs, warmup_exp_beta
        )

    baseline_cls = REINFORCE_BASELINES_REGISTRY.get(name, None)
    if baseline_cls is None:
        raise ValueError(
            f"Unknown baseline {baseline_cls}. Available baselines: {REINFORCE_BASELINES_REGISTRY.keys()}"
        )
    return baseline_cls(**kw)
