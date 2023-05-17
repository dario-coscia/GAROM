# GAROM
GAROM-Generative Adversarial Reduce Order Modelling

This repository contains a very näive implementation of the GAROM model, as well as the experiments for reproduce the results.

### Baseline results.
Go into the `experiments/POD` folder. The programm can be lauched by:
```
bash perform_pod.sh
```

### GAROM results.
Go into the `experiments/GAROM` folder. The training programm can be lauched by:
```
bash perform_tests.sh
```
Once the train in finished, the results can be visualized by:
```
bash perform_analysis.sh
```

**NOTE**: The program lauches the simulations for regularized and not-regularized GAROM for all hidden dimensions for 5 different training. This reproduces all the results of the paper, but the overall time is considerable (~2h training for one model and one hidden dimension).
