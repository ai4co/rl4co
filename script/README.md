# Notes for the VRPLib eval script

**Step 1.** Prepare trained model checkpoint and put it to your preferred directory. This branch prorvided an example at `./checkpoints/pomo-cvrp50.ckpt`.

**Step 2.** Change the place to save the VRPLib's data files. By default, they will be saved at `./data/vrplib/`. 

**Step 3.** Run the script with the following command (at the root path of `rl4co` repo):

```bash
python script/eval_vrplib.py
```

If VRPLib files are not saved, it will download and save them automatically. By default, it will download all VRPLib sets: `A, B, E, F, M, P, X`. If you don't want some of them, you could comment lines in the `for name in problem_names:` loop. Also, the line:

```python
problem_names = vrplib.list_names(low=500, high=1003, vrp_type='cvrp') 
```

will restrict the size of the VRPLib instances to be evaluated. You can change the `low` and `high` values to restrict the size of the instances.

The script will evaluate the trained model on the VRPLib instances and print the results.

Example output:

```
The capacity capacity for 501 locations is not defined. Using the closest capacity: 100.0                    with 500 locations.
Problem: X-n502-k39      Cost: 146100     Optimal Cost: 69226            Gap: 111.048%
The capacity capacity for 512 locations is not defined. Using the closest capacity: 100.0                    with 500 locations.
Problem: X-n513-k21      Cost: 31790      Optimal Cost: 24201            Gap: 31.358%
The capacity capacity for 523 locations is not defined. Using the closest capacity: 100.0                    with 500 locations.
Problem: X-n524-k153     Cost: 175833     Optimal Cost: 154593           Gap: 13.739%
The capacity capacity for 535 locations is not defined. Using the closest capacity: 100.0                    with 500 locations.
Problem: X-n536-k96      Cost: 107123     Optimal Cost: 94846            Gap: 12.944%
The capacity capacity for 547 locations is not defined. Using the closest capacity: 100.0                    with 500 locations.
```
