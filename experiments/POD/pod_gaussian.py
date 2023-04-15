import pandas as pd
import numpy as np
from ezyrb import POD, RBF, Database, GPR, ANN, AE
from ezyrb import ReducedOrderModel as ROM
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import get_args

args = get_args()


class ParametricGaussian(object):

    def __init__(self, nx=40, ny=40, domain=[-1, 1], numpy=False) -> None:
        import torch
        import matplotlib

        # params
        xx = torch.linspace(domain[0], domain[1], 20)
        yy = xx
        params = torch.cartesian_prod(xx, yy)
        # define domain
        x = torch.linspace(domain[0], domain[1], nx)
        y = torch.linspace(domain[0], domain[1], ny)
        domain = torch.cartesian_prod(x, y)

        self.triang = matplotlib.tri.Triangulation(domain[:, 0], domain[:, 1])

        sol = []
        for p in params:
            sol.append(self.func(domain, p[0], p[1]))
        snapshots = torch.stack(sol)

        if numpy:
            params = params.numpy()
            snapshots = snapshots.numpy()

        self.params = params
        self.snapshots = snapshots

    def func(self, x, mu1, mu2):
        import torch
        x_m1 = (x[:, 0] - mu1).pow(2)
        x_m2 = (x[:, 1] - mu2).pow(2)
        norm = x[:, 0]**2 + x[:, 1]**2
        return torch.exp(-(x_m1 + x_m2))


data = ParametricGaussian(numpy=True)
snap_training = 320

inputs = data.snapshots
parameters = data.params

N_snap = inputs.shape[0]
print(f"N_snap: {N_snap}")

if snap_training is None:
    snap_training = N_snap

# splitting indeces for train and test
indeces_train = np.linspace(0, N_snap-1, num=snap_training, dtype=int)
mask_train = np.zeros(N_snap, dtype=bool)
mask_train[indeces_train] = True
mask_test = np.ones(N_snap, dtype=bool)
mask_test[indeces_train] = False
print(f"snap_training: {sum(mask_train)}")
print(f"snap_testing: {sum(mask_test)}")

params_training = parameters[mask_train, :]
snapshots_training = inputs[mask_train, :]
params_testing = parameters[mask_test, :]
snapshots_testing = inputs[mask_test, :]


if args.singular_values:
    import matplotlib.pyplot as plt
    db = Database(parameters, inputs)
    pod = POD('svd')
    rbf = RBF()
    rom = ROM(db, pod, rbf)
    rom.fit()
    plt.semilogy(range(len(pod.singular_values)), pod.singular_values)
    plt.title('gaussian')
    plt.savefig('singularvalues/gaussian.png')


data_res = []

# input
input_dim = inputs.shape[1]
db_training = Database(params_training, snapshots_training)
db_testing = Database(params_testing, snapshots_testing)

method = {"RBF": RBF(),
          "ANN": ANN([60, 60], nn.Tanh(), [10000, 1e-12])}

dimensions = [4, 8, 16, 24, 32, 64, 120]

for dim in dimensions:

    for key, interp in method.items():

        pod = POD('svd', rank=dim)
        rbf = interp
        rom = ROM(db_training, pod, rbf)
        rom.fit()

        train_error = rom.test_error(db_training)
        test_error = rom.test_error(db_testing)
        data_res.append(
            [f"POD + {key}", dim, 100*train_error, 100*test_error])

        print(f"POD + {key} == dimension: {dim}")
        print(f"    ROM train error : {train_error:.5%}")
        print(f"    ROM test error : {test_error:.5%} ")
        print()

    for key, interp in method.items():

        pod = AE(layers_encoder=[input_dim // 3, input_dim // 6, dim],
                 layers_decoder=[dim, input_dim // 3, input_dim // 6],
                 function_encoder=nn.ReLU(),
                 function_decoder=nn.ReLU(),
                 stop_training=[1000, 1e-12])
        rbf = interp
        rom = ROM(db_training, pod, rbf)
        rom.fit()

        train_error = rom.test_error(db_training)
        test_error = rom.test_error(db_testing)
        data_res.append([f"AE + {key}", dim, 100*train_error,
                         100*test_error])

        print(f"AE + {key} == dimension: {dim}")
        print(f"    ROM train error : {train_error:.5%}")
        print(f"    ROM test error : {test_error:.5%} ")
        print()

    print("===============")
    data_res.append(["-", "-", "-", "-"])

df_res = pd.DataFrame(
    data_res, columns=['method', 'dim', 'train %', 'test %'])
df_res.to_csv('gaussian.csv')
