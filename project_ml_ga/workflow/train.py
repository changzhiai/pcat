"""Train set from single point calculations and multiple models with different archetecutres are trained"""
import numpy as np
import math
import json, os, sys, toml
from pathlib import Path
import argparse
import logging
import itertools
import torch
import time
from ase.io import read, write
from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Load model parameters from previous run",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help="Atomic interaction cutoff distance [�~E]",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        help="Number of interaction layers used",
    )
    parser.add_argument(
        "--node_size", type=int, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset", type=str, help="Path to ASE trajectory",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Number of molecules per minibatch",
    )
    parser.add_argument(
        "--initial_lr", type=float, help="Initial learning rate",
    )
    parser.add_argument(
        "--forces_weight",
        type=float,
        help="Tradeoff between training on forces (weight=1) and energy (weight=0)",
    )
    parser.add_argument(
        "--log_inverval",
        type=int,
        help="The interval of model evaluation",
    )
    parser.add_argument(
        "--normalization",
        action="store_true",
        help="Enable normalization of the model",
    )
    parser.add_argument(
        "--atomwise_normalization",
        action="store_true",
        help="Enable atomwise normalization",
    )
    parser.add_argument(
        "--stop_tolerance",
        type=int,
        help="Stop training when validation loss is larger than best loss for 'stop_tolerance' steps",
    )   
    parser.add_argument(
        "--cfg",
        type=str,
        help="Path to config file. e.g. 'arguments.toml'"
    )
    
    return parser.parse_args(arg_list)


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

def plot_learning_curve(name='printlog1.txt', save=True):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd
    # mpl.use('TkAgg')
    df = generate_csv(name)
    df = pd.read_csv('data.csv', skiprows=[1, 2, 3])
    fig, ax = plt.subplots()
    c1 = 'red'; c2 = 'blue'
    plt.xlabel('Number of step')
    ax.plot(df['step'], df['energy_rmse'], color=c1)
    ax.set_ylabel('Energy RMSE', color=c1)
    ax.tick_params(axis="y", labelcolor=c1)
    ax2 = ax.twinx()
    ax2.plot(df['step'], df['forces_rmse'], color=c2)
    ax2.set_ylabel('Forces RMSE', color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    if save == True:
        fig.savefig('learning_curve.png')

def split_data(dataset, args):
    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * 0.10))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

    # Save split file
    with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
        json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits

def forces_criterion(predicted, target, reduction="mean"):
    # predicted, target are (bs, max_nodes, 3) tensors
    # node_count is (bs) tensor
    diff = predicted - target
    total_squared_norm = torch.linalg.norm(diff, dim=1)  # bs
    if reduction == "mean":
        scalar = torch.mean(total_squared_norm)
    elif reduction == "sum":
        scalar = torch.sum(total_squared_norm)
    else:
        raise ValueError("Reduction must be 'mean' or 'sum'")
    return scalar

def get_normalization(dataset, per_atom=True):
    # Use double precision to avoid overflows
    x_sum = torch.zeros(1, dtype=torch.double)
    x_2 = torch.zeros(1, dtype=torch.double)
    num_objects = 0
    for i, sample in enumerate(dataset):
        if i == 0:
            # Estimate "bias" from 1 sample
            # to avoid overflows for large valued datasets
            if per_atom:
                bias = sample["energy"] / sample["num_atoms"]
            else:
                bias = sample["energy"]
        x = sample["energy"]
        if per_atom:
            x = x / sample["num_atoms"]
        x -= bias
        x_sum += x
        x_2 += x ** 2.0
        num_objects += 1
    # Var(X) = E[X^2] - E[X]^2
    x_mean = x_sum / num_objects
    x_var = x_2 / num_objects - x_mean ** 2.0
    x_mean = x_mean + bias

    default_type = torch.get_default_dtype()

    return x_mean.type(default_type), torch.sqrt(x_var).type(default_type)

def eval_model(model, dataloader, device, forces_weight):
    energy_running_ae = 0
    energy_running_se = 0

    forces_running_l2_ae = 0
    forces_running_l2_se = 0
    forces_running_c_ae = 0
    forces_running_c_se = 0
    forces_running_loss = 0

    running_loss = 0
    count = 0
    forces_count = 0
    criterion = torch.nn.MSELoss()

    for batch in dataloader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        out = model(device_batch)

        # counts
        count += batch["energy"].shape[0]
        forces_count += batch['forces'].shape[0]
        
        # use mean square loss here
        forces_loss = forces_criterion(out["forces"], device_batch["forces"]).item()
        energy_loss = criterion(out["energy"], device_batch["energy"]).item()  #problem here
        total_loss = forces_weight * forces_loss + (1 - forces_weight) * energy_loss
        running_loss += total_loss * batch["energy"].shape[0]
        
        # energy errors
        outputs = {key: val.detach().cpu().numpy() for key, val in out.items()}
        energy_targets = batch["energy"].detach().cpu().numpy()
        energy_running_ae += np.sum(np.abs(energy_targets - outputs["energy"]), axis=0)
        energy_running_se += np.sum(
            np.square(energy_targets - outputs["energy"]), axis=0
        )

        # force errors
        forces_targets = batch["forces"].detach().cpu().numpy()
        forces_diff = forces_targets - outputs["forces"]
        forces_l2_norm = np.sqrt(np.sum(np.square(forces_diff), axis=1))

        forces_running_c_ae += np.sum(np.abs(forces_diff))
        forces_running_c_se += np.sum(np.square(forces_diff))

        forces_running_l2_ae += np.sum(np.abs(forces_l2_norm))
        forces_running_l2_se += np.sum(np.square(forces_l2_norm))

    energy_mae = energy_running_ae / count
    energy_rmse = np.sqrt(energy_running_se / count)

    forces_l2_mae = forces_running_l2_ae / forces_count
    forces_l2_rmse = np.sqrt(forces_running_l2_se / forces_count)

    forces_c_mae = forces_running_c_ae / (forces_count * 3)
    forces_c_rmse = np.sqrt(forces_running_c_se / (forces_count * 3))

    total_loss = running_loss / count

    evaluation = {
        "energy_mae": energy_mae,
        "energy_rmse": energy_rmse,
        "forces_l2_mae": forces_l2_mae,
        "forces_l2_rmse": forces_l2_rmse,
        "forces_mae": forces_c_mae,
        "forces_rmse": forces_c_rmse,
        "sqrt(total_loss)": np.sqrt(total_loss),
    }

    return evaluation

def update_namespace(ns, d):
    for k, v in d.items():
        if not ns.__dict__.get(k):
            ns.__dict__[k] = v

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        if best_loss < 1.0 and (val_loss - best_loss) > self.min_delta:
            self.counter +=1
            print(self.counter, self.tolerance)
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
        return self.early_stop

def main(args):
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, args.printlog), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Create device
    device = torch.device(args.device)
    # Put a tensor on the device before loading data. This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)

    # Setup dataset and loader
    logging.info("loading data %s", args.dataset)
    if isinstance(args.dataset, list):
        images_real = []
        n = len(args.dataset)
        for data in args.dataset:
            images = read(data, ':')
            images_real += images
        real_traj = f'vasp_PdTiH_adss_r{n}_real.traj'
        write(real_traj, images_real)
        args.dataset = real_traj

    dataset = AseDataset(
        args.dataset,
        cutoff = args.cutoff,
    )
    
    datasplits = split_data(dataset, args)

    train_loader = torch.utils.data.DataLoader(
        datasplits["train"],
        args.batch_size,
        sampler=torch.utils.data.RandomSampler(datasplits["train"]),
        collate_fn=collate_atomsdata,
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"], 
        args.batch_size, 
        collate_fn=collate_atomsdata,
    )
    
    logging.info("Computing mean and variance")
    target_mean, target_stddev = get_normalization(
        datasplits["train"], 
        per_atom=args.atomwise_normalization,
    )
    logging.debug("target_mean=%f, target_stddev=%f" % (target_mean, target_stddev))

    net = PainnModel(
        num_interactions=args.num_interactions, 
        hidden_state_size=args.node_size, 
        cutoff=args.cutoff,
        normalization=args.normalization,
        target_mean=target_mean.tolist(),
        target_stddev=target_stddev.tolist(),
        atomwise_normalization=args.atomwise_normalization,
    )
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr)
    criterion = torch.nn.MSELoss()
    scheduler_fn = lambda step: 0.96 ** (step / 100000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
    early_stop = EarlyStopping(tolerance=args.stop_tolerance)    

    running_loss = 0
    running_loss_count = 0
    best_val_loss = np.inf
    step = 0
    training_time = 0    

    if args.load_model:
        state_dict = torch.load(args.load_model)
        net.load_state_dict(state_dict["model"])
        best_val_loss = state_dict["best_val_loss"]
    
    for epoch in itertools.count():
        for batch_host in train_loader:
            start = time.time()
            # Transfer to 'device'
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            # Reset gradient
            optimizer.zero_grad()

            # Forward, backward and optimize
            outputs = net(
                batch, compute_forces=bool(args.forces_weight)
            )
            energy_loss = criterion(outputs["energy"], batch["energy"])
            if args.forces_weight:
                forces_loss = forces_criterion(outputs['forces'], batch['forces'])
            else:
                forces_loss = 0.0
            total_loss = (
                args.forces_weight * forces_loss
                + (1 - args.forces_weight) * energy_loss
            )
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * batch["energy"].shape[0]
            running_loss_count += batch["energy"].shape[0]
            training_time += time.time() -  start

            # print(step, loss_value)
            # Validate and save model
            if (step % args.log_interval == 0) or ((step + 1) == args.max_steps):
                eval_start = time.time()
                train_loss = running_loss / running_loss_count
                running_loss = running_loss_count = 0

                eval_dict = eval_model(net, val_loader, device, args.forces_weight)
                eval_formatted = ", ".join(
                    ["%s=%g" % (k, v) for (k, v) in eval_dict.items()]
                )

                logging.info(
                    "step=%d, %s, sqrt(train_loss)=%g, max memory used=%g, training time=%g min, eval time=%g min",
                    step,
                    eval_formatted,
                    math.sqrt(train_loss),
                    torch.cuda.max_memory_allocated() / 2**20,
                    training_time / 60,
                    (time.time() - eval_start) / 60
                )
                training_time = 0
                # Save checkpoint
                print(eval_dict["sqrt(total_loss)"], best_val_loss)
                if not early_stop(eval_dict["sqrt(total_loss)"], best_val_loss):
                    best_val_loss = eval_dict["sqrt(total_loss)"]
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_loss": best_val_loss,
                            "node_size": args.node_size,
                            "num_layer": args.num_interactions,
                            "cutoff": args.cutoff,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )
                else:
                    logging.info("early stop, exiting")
                    # sys.exit(0)
                    return args

            step += 1

            scheduler.step()

            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                torch.save(
                    {
                        "model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "node_size": args.node_size,
                        "num_layer": args.num_interactions,
                        "cutoff": args.cutoff,
                    },
                    os.path.join(args.output_dir, "exit_model.pth"),
                )
                # sys.exit(0)
                return args

def generate_run_config(params):
    with open(os.path.join(params['run_dir'], "run.toml"), 'w') as f:
        toml.dump(params, f)
    return params

if __name__ == "__main__":
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
    _ = generate_run_config(params)
    update_namespace(args, params) 
    args = main(args)
    plot_learning_curve(name=args.printlog_dir, save=True)
    print('Train done!')
