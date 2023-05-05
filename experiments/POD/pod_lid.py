import pandas as pd
import numpy as np
from ezyrb import POD, RBF, Database, GPR, ANN, AE
from ezyrb import ReducedOrderModel as ROM
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import get_args

args = get_args()


class LidCavity(object):

    def __init__(self) -> None:
        import numpy as np
        import matplotlib.tri as tri

        params = np.load('data/params.npy')
        snapshots_u = np.load('data/snapshots_u.npy')
        snapshots_p = np.load('data/snapshots_p.npy')
        coordinates = np.load('data/coordinates.npy')
        triang = tri.Triangulation(coordinates[:, 0], coordinates[:, 1])

        self.params = params.reshape((-1, 1))
        self.snapshots = {'u': snapshots_u, 'p': snapshots_p}
        self.triang = triang
        self.coordinates = coordinates


data = LidCavity()
snap_training = 240
key = 'u'

inputs = data.snapshots[key]
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
    plt.title('lid')
    plt.savefig('singularvalues/lid.png')


data_res = []

# input
input_dim = inputs.shape[1]
db_training = Database(params_training, snapshots_training)
db_testing = Database(params_testing, snapshots_testing)

method = {"RBF": RBF,
          "ANN": ANN}

dimensions = [16, 64, 120]

for dim in dimensions:

    for key, interp in method.items():

        pod = POD('svd', rank=dim)
        if key == "ANN":
            rbf = interp([24, 60], nn.ReLU(), [20000, 1e-12])
        else:
            rbf = interp()
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
                 stop_training=[200, 1e-12])
        if key == "ANN":
            rbf = interp([24, 60], nn.ReLU(), [20000, 1e-12])
        else:
            rbf = interp()
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
df_res.to_csv('results/lid.csv')
