from garom import GAROM
import torch
from plot_results import plot_POD_vs_GAROM, plot_densities_generator, save_data
from utils import preprocessing_lidcavity, assess_model_quality, get_args
from smithers.dataset import LidCavity
import time

# get args parser
args = get_args()
path = args.path

if path is not None:
    TRAIN = False
    ANALYSIS = True
else:
    TRAIN = True
    ANALYSIS = False

key='mag(v)'
dataset_training, dataset_testing = preprocessing_lidcavity(snap_training=240, key = key)


training_loader = torch.utils.data.DataLoader(
    dataset_training, batch_size=8, drop_last=False, shuffle=True) #8

testing_loader = torch.utils.data.DataLoader(
    dataset_testing, batch_size=1, drop_last=False, shuffle=True)

# Model setup
nz = args.noise_dim
latent_dim = args.encoding_dim
plotting = args.plot
printing = args.print
every = args.print_every
epochs = args.epochs
gamma = args.gamma
lambda_k = args.lambda_k
input_dim = 5041
param_dim = 1
print(
    f"simulation: seed {args.seed}, encoding dim {latent_dim}, noise dim {nz}, gamma {gamma}, regularizer {args.regular}")

# optimizer and scheduler
optimizers = {'generator': torch.optim.Adam,
              'discriminator': torch.optim.Adam}
optimizers_kwds = {'generator': {"lr": 0.001},
                   'discriminator': {"lr": 0.001}}

schedulers = None
schedulers_kwds = None

# create GAROM and train
if args.seed is not None:
    torch.manual_seed(args.seed)

garom = GAROM(input_dimension=input_dim,
              hidden_dimension=latent_dim,
              parameters_dimension=param_dim,
              noise_dimension=nz,
              regularizer=args.regular,
              optimizers=optimizers,
              optimizer_kwargs=optimizers_kwds,
              schedulers=schedulers,
              scheduler_kwargs=schedulers_kwds)

# if training is on
if TRAIN:
    start = time.time()
    garom.train(training_loader, epochs=epochs,
                gamma=gamma, lambda_k=lambda_k,
                every=args.print_every,save_csv=f'lid_{latent_dim}_{args.seed}')
    print(f"total time {time.time()-start}")

    garom.save(f'generator_lid_{latent_dim}', f'discriminator_lid_{latent_dim}')

# loading model
garom.load(f'{path}/generator_lid_{latent_dim}',
           f'{path}/discriminator_lid_{latent_dim}',
           map_location=torch.device('cpu'))


if ANALYSIS:
    assess_model_quality(garom, training_loader,
                        testing_loader, 'Lid', LidCavity,
                        plotting=plotting,
                        printing=printing)
    save_data(garom, dataset_training, dataset_testing, file_name=f'lid_violin_data_{args.regular}', append=True)