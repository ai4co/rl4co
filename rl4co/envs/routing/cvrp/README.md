As local search in CVRP, we adopted the SWAP* algorithm by Vidal et al. [1, 2] ([repo](https://github.com/vidalt/HGS-CVRP)). Specifically, we use the modified version of code provided by [DeepACO](https://github.com/henry-yeh/DeepACO/tree/main/cvrp_nls/HGS-CVRP-main), which we uploaded to [https://github.com/ai4co/HGS-CVRP](https://github.com/ai4co/HGS-CVRP) for convenience.


### Installation

```bash
cd rl4co/envs/routing/cvrp
git clone git@github.com:ai4co/HGS-CVRP.git
cd HGS-CVRP
bash build.sh
```

### References

[1] Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., Rei, W. (2012). A hybrid genetic algorithm for multidepot and periodic vehicle routing problems. Operations Research, 60(3), 611-624.

[2] Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. Computers & Operations Research, 140, 105643.

[3] Ye, H., Wang J., Cao, Z., Liang, H., Li, Y. (2023).
DeepACO: Neural-enhanced ant systems for combinatorial optimization. Advances in neural information processing systems (NeurIPS) 36 (2023): 43706-43728.
