import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



def plot_POD_vs_GAROM(data_train, data_test, dataset, hidden_dim, garom, path_save, key=None):

    from ezyrb import POD, RBF, Database
    from ezyrb import ReducedOrderModel as ROM
    from numpy.linalg import norm
    import numpy
    import torch

    snapshots_train = data_train.tensors[0].numpy()
    params_train = data_train.tensors[1].numpy()
    snapshots_test = data_test.tensors[0].numpy()
    params_test = data_test.tensors[1].numpy()


    N_snap = snapshots_train.shape[0] + snapshots_test.shape[0]
    snap_training = snapshots_train.shape[0]

    indeces_train = numpy.linspace(0, N_snap - 1, num=snap_training, dtype=int)
    indeces_test = [i for i in range(N_snap) if i not in indeces_train]

    params = dataset().params
    snapshots = dataset().snapshots
    if key is not None:
        snapshots = snapshots[key]


    db_training = Database(params_train, snapshots_train)
    db = Database(params_test, snapshots_test)


    # perform POD
    pod = POD('svd', rank=hidden_dim)
    rbf = RBF()
    rom = ROM(db_training, pod, rbf)
    rom.fit()

    # predicting POD
    predicted_test = rom.predict(db.parameters)
    err_pod = norm(predicted_test - db.snapshots, axis=1) / norm(db.snapshots, axis=1)

    # predicting GAROM
    valid = torch.tensor(params_test, dtype=torch.float) #torch.tensor(params, dtype=torch.float)
    gen_imgs, variance = garom.estimate(valid, variance=True, mc_steps=100)
    gen_imgs = gen_imgs.detach().numpy()
    variance = variance.detach().numpy()

    err_garom = norm(gen_imgs - db.snapshots, axis=1) / norm(db.snapshots, axis=1)


    # TODO make better plot
    plt.semilogy(range(len(err_pod)), err_pod, 'r-')
    plt.semilogy(range(len(err_pod)), err_garom, 'b-')
    plt.savefig(f'{path_save}_PODvsGAROM.pdf')


def plot_densities_generator(garom, data_train, data_test, path_save):

    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    garom.to("cpu")
    garom.eval()

    sns.set(style="white", color_codes=True)

    # get the snapshots
    snapshots_train = data_train.tensors[0]
    params_train = data_train.tensors[1]
    snapshots_test = data_test.tensors[0]
    params_test = data_test.tensors[1]

    # reduce
    gen_test = garom.estimate(params_test, mc_steps=100)
    gen_train = garom.estimate(params_train, mc_steps=100)

    #TODO make better plot
    # residual distribution
    gen_test = (gen_test - snapshots_test).mean(dim=-1)
    gen_train = (gen_train - snapshots_train).mean(dim=-1)

    fig, ax = plt.subplots()
    sns.kdeplot(gen_test.detach().numpy(), color='r', ax=ax, label='test')
    sns.kdeplot(gen_train.detach().numpy(), color='b', ax=ax, label='train')
    plt.legend()
    plt.savefig(f'{path_save}_density.pdf')


def save_data(garom, data_train, data_test, file_name=None, append=False):

    import torch.nn.functional as F
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    garom.to("cpu")
    garom.eval()

    sns.set(style="white", color_codes=True)

    # get the snapshots
    snapshots_train = data_train.tensors[0]
    params_train = data_train.tensors[1]
    snapshots_test = data_test.tensors[0]
    params_test = data_test.tensors[1]

    # get latent dim
    hidden_dim = float(garom.discriminator.encoding(snapshots_test).shape[1])

    # reduce
    gen_test = garom.estimate(params_test, mc_steps=100)
    gen_train = garom.estimate(params_train, mc_steps=100)

    # residual distribution
    gen_test = (gen_test - snapshots_test).mean(dim=-1)
    gen_train = (gen_train - snapshots_train).mean(dim=-1)

    # create dataframe
    label_test = ['test' for _ in range(gen_test.shape[0])]
    label_train = ['train' for _ in range(gen_train.shape[0])]

    # train + test
    label = label_train + label_test
    data = gen_train.tolist() + gen_test.tolist()

    latent_dim = [hidden_dim for _ in range(len(label))]
    data_res = list(zip(data, latent_dim, label))
    df_res = pd.DataFrame(data_res, columns=['data', 'latent_dim', 'kind'])

    if not append:
        df_res.to_csv(f'{file_name}.csv')
    else:
        try:
            df = pd.read_csv(f'{file_name}.csv')
            df = pd.concat([df, df_res])
            df = df[['data', 'latent_dim', 'kind']]
            df.to_csv(f'{file_name}.csv')
        except:
            df_res.to_csv(f'{file_name}.csv')


def plot_violins():
    import seaborn as sns
    import pandas as pd
    sns.set(style="white", color_codes=True)
    sns.set_style("ticks", {"font.family": "serif", "text.usetex": True})

    # hyper paramters
    font_size_ticks = 18
    font_size_legend = 18
    font_size_ticks_scientific = (3*font_size_ticks)//4
    line_width = 1.2
    grid_line_width = 0.6
    alpha = 0.6

    types=['gaussian', 'graetz', 'lid']
    reg = ['1.0', '0.0']

    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(16, 9)
    fig.subplots_adjust(hspace=0.0, wspace=0.3)
    ax = ax.flat

    my_pal = {species: "#e41a1c" if species == "train" else "#377eb8" for species in ['train', 'test']}

    for i_r, is_reg in enumerate(reg):
        for i_t, type in enumerate(types):

            if type != 'gaussian':
                hidden_dims=[16, 64, 120]
            else:
                hidden_dims=[4, 16, 64]

            df = pd.read_csv(f'{type}_violin_data_{is_reg}.csv')
            sns.violinplot(data=df, x="latent_dim", y="data", hue="kind", 
                           split=True, ax=ax[i_r*3 + i_t], linewidth=line_width, palette=my_pal, saturation=alpha)
            ax[i_r*3 + i_t].get_legend().remove()
            ax[i_r*3 + i_t].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            ax[i_r*3 + i_t].yaxis.get_offset_text().set_fontsize(font_size_ticks_scientific)
            ax[i_r*3 + i_t].xaxis.get_offset_text().set_fontsize(font_size_ticks_scientific)
            ax[i_r*3 + i_t].tick_params(axis='both', which='major', labelsize=font_size_ticks)

            ax[i_r*3 + i_t].set_xticklabels([f'${x}$' for x in hidden_dims])
            ax[i_r*3 + i_t].tick_params(axis='both', which='minor', labelsize=font_size_ticks)

    # Add legend outside the plot
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', fontsize=font_size_legend)

    # Remove y-labels
    for i in range(len(ax)):
        ax[i].set_ylabel('')

    # adding only needed labels
    for i, name in enumerate(types):
        ax[i].set_title( r"\textbf{" + str(name.capitalize()) + "}", fontsize=font_size_legend)


    ax[0].set_ylabel('r-GAROM', rotation=0,  ha='right', va='center', labelpad=15, fontsize=font_size_ticks)
    ax[3].set_ylabel(ylabel='GAROM', rotation=0,  ha='right', va='center', labelpad=15, fontsize=font_size_ticks)
    for i in range(len(ax)):
        if i >= 3:
            ax[i].set_xlabel('latent dimension', fontsize=font_size_ticks)
        else:
            ax[i].set_xlabel(xlabel='')
    
    plt.tight_layout()
    plt.savefig("violin.pdf")


def plot_metric_residual_convergence(sim='results_1', what_to_plot = ['residual']):
    import pandas as pd
    from matplotlib.pyplot import cm
    import glob
    import numpy as np
    import matplotlib.ticker as ticker

    simulations = glob.glob("sim_*")

    # hyper paramters
    font_size_ticks = 18
    font_size_ticks_scientific = (3*font_size_ticks)//4
    line_width = 1.2
    grid_line_width = 0.6
    alpha = 0.2

    types=['gaussian', 'graetz', 'lid']
    #fig, ax = plt.subplots(3, 3, sharey=True)
    fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
    fig.set_size_inches(16, 9)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    ax = ax.flat

    for id_type, type in enumerate(types):
            
        if type != 'gaussian':
            hidden_dims=[16, 64, 120]
        else:
            hidden_dims=[4, 16, 64]

        # create dataset
        #colors = cm.rainbow(np.linspace(0, 1, len(what_to_plot)))
        if len(what_to_plot) == 1:
            colors = "#377eb8"
        else:
            raise ValueError

        for i, hd in enumerate(hidden_dims):
            data = []
            

            for idx in simulations:
                df = pd.read_csv(f'{idx}/{sim}/{type}/{type}_{hd}_{idx[4:]}.csv')
                data.append(df)

            concat = pd.concat(data).groupby(level=0)
            df = concat.mean()
            df_max = concat.max()
            df_min = concat.min()
            df_std = concat.std()
            df.drop([0])

            #MAKE BEUTIFUL PLOT
            for ic, pp in enumerate(what_to_plot):

                #some confidence interval
                ci = 1.96 * df_std[pp]/np.sqrt(len(df_std['epoch']))

                idx_plot = i + 3 * id_type 
                ax[idx_plot].plot(df['epoch'], df[pp], color=colors, ls='-', linewidth = line_width, marker='o')
                ax[idx_plot].ticklabel_format(style='sci',scilimits=(0,0),axis='both')
                ax[idx_plot].yaxis.get_offset_text().set_fontsize(font_size_ticks_scientific)
                ax[idx_plot].xaxis.get_offset_text().set_fontsize(font_size_ticks_scientific)
                ax[idx_plot].set_title(f'latent dimension {hd}', fontsize=font_size_ticks)
                # ax[i].get_legend().remove()
                #ax[idx_plot].fill_between(df['epoch'], df_min[pp], df_max[pp], alpha=alpha, color = colors)
                ax[idx_plot].fill_between(df['epoch'], df[pp]- ci, df[pp]+ ci, alpha=alpha, color = colors)
                ax[idx_plot].grid(True, which = 'major', linestyle=':', linewidth=grid_line_width)
                ax[idx_plot].grid(True, which = 'minor', linestyle=':', linewidth=grid_line_width, alpha=alpha)
                ax[idx_plot].tick_params(axis='both', which='major', labelsize=font_size_ticks)
                ax[idx_plot].tick_params(axis='both', which='minor', labelsize=font_size_ticks)

                # x-locator
                ax[idx_plot].xaxis.set_minor_locator(plt.MultipleLocator(1000))
                ax[idx_plot].xaxis.set_major_locator(plt.MultipleLocator(5000))
                ax[idx_plot].yaxis.set_major_locator(ticker.MaxNLocator(2))
                ax[idx_plot].yaxis.set_minor_locator(ticker.AutoMinorLocator())

                # add title on the left side of the plot
                ax[3 * id_type ].set_ylabel( r"\textbf{" + str(type.capitalize()) + "}", fontsize=font_size_ticks, rotation=0, ha='right', va='center', labelpad=15)

    plt.tight_layout()
    plt.savefig(f"convergence_{sim}.pdf")

