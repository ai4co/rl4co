import abc

from typing import Any

from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
from torch.utils.data import Dataset

from rl4co.models.rl.common.base import RL4COLitModule


class TransductiveModel(RL4COLitModule, metaclass=abc.ABCMeta):
    """Base class for transductive algorithms (i.e. that optimize policy parameters for
    specific instances, see https://en.wikipedia.org/wiki/Transduction_(machine_learning)).
    Transductive algorithms are used online to find better solutions for a given dataset, i.e.
    given a policy, improve (a part of) its parameters such that
    the policy performs better on the given dataset.

    Note:
        By default, we use manual optimization to handle the search.

    Args:
        env: RL4CO environment
        policy: policy network
        dataset: dataset to use for training
        batch_size: batch size
        max_iters: maximum number of iterations
        max_runtime: maximum runtime in seconds
        save_path: path to save the model
        **kwargs: additional arguments
    """

    def __init__(
        self,
        env,
        policy,
        dataset: Dataset | str,
        batch_size: int = 1,
        max_iters: int = 100,
        max_runtime: int | None = 86_400,
        save_path: str | None = None,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False, ignore=["env", "policy", "dataset"])
        super().__init__(env, policy, **kwargs)
        self.dataset = dataset
        self.automatic_optimization = False  # we optimize manually

    def setup(self, stage="fit"):
        """Setup the dataset and attributes.
        The RL4COLitModulebase class automatically loads the data.
        """
        if isinstance(self.dataset, str):
            # load from file
            self.dataset = self.env.dataset(filename=self.dataset)

        # Set all datasets and batch size as the same
        for split in ["train", "val", "test"]:
            setattr(self, f"{split}_dataset", self.dataset)
            setattr(self, f"{split}_batch_size", self.hparams.batch_size)

        # Setup loggers
        self.setup_loggers()

    def setup_optimizer(self, parameters) -> LightningOptimizer:
        """Instantiate an optimizer over `parameters` and register it with the Lightning
        strategy, so that it is the optimizer returned by `self.optimizers()`.

        Transductive methods only adapt a small subset of the parameters, which is not known
        yet when Lightning calls the `configure_optimizers` hook at setup time. Registering
        the optimizer here is what makes `self.optimizers().step()` update the adapted
        parameters, and routes the step through the precision plugin so that mixed precision
        (the `RL4COTrainer` default) unscales the gradients before stepping.

        Args:
            parameters: parameters to be adapted during the search
        """
        optimizer = self.configure_optimizers(parameters)
        if not isinstance(optimizer, Optimizer):
            raise ValueError(
                "Transductive methods do not support learning rate schedulers; "
                f"`configure_optimizers` must return a single optimizer, got {type(optimizer)}."
            )
        self.trainer.strategy.optimizers = [optimizer]
        return self.optimizers()

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """Called before training (i.e. search) for a new batch begins.
        This can be used to perform changes to the model or optimizer at the start of each batch.
        """
        pass  # Implement in subclass

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        """Main search loop. We use the training step to effectively adapt to a `batch` of instances."""
        raise NotImplementedError("Implement in subclass")

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Called when the train batch ends. This can be used for
        instance for logging or clearing cache.
        """
        pass  # Implement in subclass

    def on_train_epoch_end(self) -> None:
        """Called when the train ends."""
        pass  # Implement in subclass

    def validation_step(self, batch: Any, batch_idx: int):
        """Not used during search"""
        pass

    def test_step(self, batch: Any, batch_idx: int):
        """Not used during search"""
        pass
