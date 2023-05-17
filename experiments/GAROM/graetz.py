from smithers.dataset import GraetzDataset
from garom import GAROM
import torch
from plot_results import plot_POD_vs_GAROM, plot_densities_generator, save_data
from utils import preprocessing_graetz, assess_model_quality, get_args
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

dataset_training, dataset_testing = preprocessing_graetz(snap_training=160)

training_loader = torch.utils.data.DataLoader(
    dataset_training, batch_size=8, drop_last=False, shuffle=True)

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
input_dim = 5160
param_dim = 2
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
                every=args.print_every, save_csv=f'graetz_{latent_dim}_{args.seed}')
    print(f"total time {time.time()-start}")

    garom.save(f'generator_graetz_{latent_dim}', f'discriminator_graetz_{latent_dim}')

# loading model
garom.load(f'{path}/generator_graetz_{latent_dim}',
           f'{path}/discriminator_graetz_{latent_dim}',
           map_location=torch.device('cpu'))


if ANALYSIS:
    # asses quality
    assess_model_quality(garom, training_loader,
                           testing_loader, 'Graetz', GraetzDataset,
                           plotting=plotting,
                           printing=printing)
    save_data(garom, dataset_training, dataset_testing, file_name=f'graetz_violin_data_{args.regular}', append=True)
