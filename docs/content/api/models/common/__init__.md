# NCO Methods Overview


We categorize NCO approaches (which are in fact not necessarily trained with RL!) into the following: 1) constructive, 2) improvement, 3) transductive.



```{eval-rst}
.. tip::
   Note that in RL4CO we distinguish the RL algorithms and the actors via the following naming:

   * **Model:** Refers to the reinforcement learning algorithm encapsulated within a `LightningModule`. This module is responsible for training the policy.
   * **Policy:** Implemented as a `nn.Module`, this neural network (often referred to as the *actor*) takes an instance and outputs a sequence of actions, :math:`\pi = \pi_0, \pi_1, \dots, \pi_N`, which constitutes the solution.

   Here, :math:`\pi_i` represents the action taken at step :math:`i`, forming a sequence that leads to the optimal or near-optimal solution for the given instance.
```


The following table contains the categorization that we follow in RL4CO:


```{eval-rst}
.. list-table:: Overview of RL Models and Policies
   :widths: 5 5 5 5 25
   :header-rows: 1
   :stub-columns: 1

   * - Category
     - Model or Policy?
     - Input
     - Output
     - Description
   * - `Constructive <constructive.md>`_
     - Policy
     - Instance
     - Solution
     - Policies trained to generate solutions from scratch. Can be categorized into AutoRegressive (AR) and Non-Autoregressive (NAR).
   * - `Improvement <improvement.md>`_
     - Policy
     - Instance, Current Solution
     - Improved Solution
     - Policies trained to improve existing solutions iteratively, akin to local search algorithms. They focus on refining *existing* solutions rather than generating them from scratch.
   * - `Transductive <transductive.md>`_
     - Model
     - Instance, (Policy)
     - Solution, (Updated Policy)
     - Updates policy parameters during online testing to improve solutions of a specific instance.
```






