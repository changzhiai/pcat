import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

def generate_csv(name):
    import pandas as pd
    step, energy_rmse, forces_rmse = [], [], []
    with open(name) as log:
        lines = log.readlines()[3:-1]
        for line in lines:
            if 'exiting' in line:
                break
            cols = line.split(',')
            cols[1] = cols[1].split(' ')[-1]
            dicts = {}
            for col in cols[1:]:
                kv = col.split('=')
                k = kv[0].strip()
                v = kv[1].strip()
                dicts[k] = v
            # print(dicts)
            step.append(dicts['step'])
            energy_rmse.append(float(dicts['energy_rmse']))
            forces_rmse.append(float(dicts['forces_rmse']))
    tuples = {
            'step': step,
            'energy_rmse': energy_rmse,
            'forces_rmse': forces_rmse,
            }
    df = pd.DataFrame(tuples)
    df.to_csv('data.csv')
    return df


def plot_curve(df, i, dir, save=True):
    # fig, ax = plt.subplots()
    # c1 = 'red'; c2 = 'blue'
    plt.xlabel('Number of step')
    ax.plot(df['step'], df['energy_rmse'], color=f'C{i}', label=dir)
    ax.set_ylabel('Energy RMSE')
    ax.tick_params(axis="y")
    plt.ylim([0.05, 0.3])

    # ax2 = ax.twinx()
    # ax2.plot(df['step'], df['forces_rmse'], color=c2)
    # ax2.set_ylabel('Forces RMSE', color=c2)
    # ax2.tick_params(axis="y", labelcolor=c2)

def plot_energy_var(df):
    plt.xlabel('Number of step')
    mean = df.mean(axis=1)
    var = df.var(axis="columns")
    ax.plot(df.index, mean, color='red', label='Average')
    ax.fill_between(df.index, mean-var, mean+var, color='blue', label='Variance')
    ax.set_ylabel('Energy RMSE')
    # ax.tick_params(axis="y")
    plt.xlim([0, 720000])
    plt.ylim([0.07, 0.2])
    plt.legend()
    plt.show()
    fig.savefig('learning_curve_energy_uncertainty.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    dirs = ['116_nodes', '120_nodes', '124_nodes', '128_nodes', '132_nodes', '136_nodes', '140_nodes', '144_nodes']
    fig, ax = plt.subplots()
    df_energy = pd.DataFrame()
    df_forces = pd.DataFrame()
    for i, dir in enumerate(dirs):
        name = f'./{dir}/model_output/printlog.txt'
        df = generate_csv(name)
        df = pd.read_csv('data.csv', skiprows=[1, 2,])
        df = df.set_index('step')
        df_energy[dir] = df['energy_rmse']
        df_forces[dir] = df['forces_rmse']
        print(dir)
        # print(df)
        # plot_curve(df, i, dir, save=True)
    # plt.legend()
    # fig.savefig('learning_curve.png')
    print(df_energy)
    print(df_forces)
    plot_energy_var(df_energy)
