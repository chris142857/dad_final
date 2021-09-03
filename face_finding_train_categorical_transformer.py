"""
# @Description:
    Script: Train DAD network for RAG categorical face finding.
"""
import os
import pickle
import argparse
import math

import numpy as np
import pandas as pd

import torch
from torch import nn
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand
from torch.distributions.utils import broadcast_all
from pyro.poutine.util import prune_subsample_sites
import pyro
import pyro.distributions as dist
from tqdm import trange
import mlflow
import mlflow.pytorch
import torch.nn.functional as F

from neural.modules import (
    SetEquivariantDesignNetwork, BatchDesignBaseline, SetEquivariantDesignRNN
)
from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from contrastive.mi import (
    PriorContrastiveEstimationDiscreteObsTotalEnum,
    PriorContrastiveEstimationScoreGradient,
    PriorContrastiveEstimation
)
import matplotlib.pyplot as plt
from plotters import plot_trace_2d, plot_trace_3d, plot_trace
from rag_gru_net import GRUEmitterNetwork, GRUEncoderNetwork
from extra_distributions.gumbel_softmax import GumbelSoftmax

# <editor-fold desc="[FB] Load libraries ...">
from face.face_model import  AppearanceModel
import pickle
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # <multi-runs>
# </editor-fold>

class EncoderNetwork(nn.Module):
    def __init__(
        self,
        design_dim,
        osbervation_dim,
        hidden_dim,
        encoding_dim,
        include_t,
        T,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.include_t = include_t
        self.T = T
        self.activation_layer = activation()
        self.design_dim = design_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]
        # if include_t:
        #     input_dim = design_dim + 1
        # else:
        #     input_dim = design_dim
        input_dim = self.design_dim_flat
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                  nn.LayerNorm(hidden_dim),
                                  activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer_0 = nn.Linear(hidden_dim, encoding_dim)
        self.output_layer_1 = nn.Linear(hidden_dim, encoding_dim)
        self.output_layer_2 = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y, t):
        # if self.include_t:
        #     t = xi.new_tensor(t) / self.T
        #     x = torch.cat([lexpand(t, *xi.shape[:-1]), xi], axis=-1)
        # else:
        #     x = xi
        # x = xi.squeeze(-2)
        x = xi.flatten(-2)        
        x = self.input_layer(x)
        x = self.activation_layer(x)
        x = self.middle(x)
        x_0 = self.output_layer_0(x)
        x_1 = self.output_layer_1(x)
        x_2 = self.output_layer_2(x)
        
        # y = F.one_hot(y, num_classes=3)
        # x = y.unsqueeze(-1) * x_1 + (1.0 - y).unsqueeze(-1) * x_0
        if len(y.shape) >= 2:
            x = y[:, 0].unsqueeze(-1) * x_0 + y[:, 1].unsqueeze(-1) * x_1 + y[:, 2].unsqueeze(-1) * x_2
        else:
            x = y[0] * x_0 + y[1] * x_1 + y[2] * x_2
        return x


class EmitterNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        design_dim,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.activation_layer = activation()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                  nn.LayerNorm(hidden_dim),
                                  activation()
                                  )
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.design_dim = design_dim
        output_dim = design_dim[0] * design_dim[1]
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, r):
        # x = self.activation_layer(r)
        x = self.input_layer(r)
        x = self.activation_layer(x)
        x = self.middle(x)
        xi_flat = self.output_layer(x)
        return xi_flat.reshape(xi_flat.shape[:-1] + self.design_dim)        



class HyperbolicTemporalDiscounting(nn.Module):
    """Hyperbolic Temporal Discounting example."""

    def __init__(
        self,
        design_net,
        T=2,
        design_dim=(1, 1),
        theta_loc=None,  # prior on theta mean hyperparam
        theta_covmat=None,  # prior on theta covariance hyperparam
        noise_scale=None,  # this is the scale of the noise term
        use_rnn=False
    ):
        super().__init__()
        # theta prior hyperparams
        self.design_net = design_net
        self.design_dim = design_dim
        self.T = T  # number of experiments
        self.sigmoid = nn.Sigmoid()
        
        # Set prior:
        self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((1, 1))
        self.theta_covmat = theta_covmat if theta_covmat is not None else torch.ones((1, 1))
        self.theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        ).to_event(1)
        self.use_rnn = use_rnn

        # self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((1,))
        # self.theta_covmat = theta_covmat if theta_covmat is not None else torch.ones((1,))
        # self.theta_prior = dist.Normal(
        #     self.theta_loc, self.theta_covmat
        # )

    # def transform_xi(self, xi, shift=0.0):

    #     d_b, r_a = xi[..., 0], xi[..., 1]
    #     # Put this logic inside the design net?
    #     # Return transformed or untransformed inputs?
    #     d_b = (d_b - shift).exp()
    #     # print(d_b.min(), d_b.max())
    #     r_a = self.r_b * self.sigmoid(r_a)
    #     return r_a, d_b

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        ########################################################################
        # Sample latent variables
        ########################################################################
        # k = latent_sample("log_k", dist.Normal(self.log_k_loc, self.log_k_scale)).exp()
        # # Use this as an offset to help with initialization
        # log_k_mean = self.log_k_loc + 0.5 * self.log_k_scale * self.log_k_scale

        # alpha = latent_sample("alpha", self.alpha_prior_distribution)
        # alpha = 1e-3 + alpha.abs()

        # epsilon = latent_sample("epsilon", self.epsilon_prior_distribution)
        
        theta = latent_sample("theta", self.theta_prior)
        
        y_outcomes = []
        xi_designs = []
        xi, y = None, None
        
        for t in range(self.T):
            ####################################################################
            # Get a design xi
            ####################################################################
            if not self.use_rnn:
                xi = compute_design(
                    f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes)),
                )
            else:
                xi = compute_design(
                    f"xi{t + 1}", self.design_net.lazy(xi, y)
                )            

            ####################################################################
            # Sample y
            ####################################################################
            # v_a = r_a / (1.0 + k * self.d_a)
            # v_b = self.r_b / (1.0 + k * d_b)
            # erf_arg = (v_a - v_b) / alpha
            # psi = epsilon + (1.0 - 2.0 * epsilon) * (0.5 + 0.5 * torch.erf(erf_arg))

            # y = observation_sample(f"y{t + 1}", dist.Bernoulli(probs=psi))          
            # two_norm = ((xi.squeeze(-1) - theta).pow(2) + 1e-10).sqrt()
            two_norm = ((xi - theta).pow(2).sum(-1) + 1e-10).sqrt()  
            scale = 10.0
            mu = math.sqrt(self.design_dim[1]) # Expansion factor due to the number of dimensions
            p_red = self.sigmoid(scale*(two_norm - 0.6 * mu))
            p_green = self.sigmoid(2*scale*(0.2 * mu - two_norm))
            f1_amber = self.sigmoid(scale*(0.6 * mu - two_norm))
            f2_amber = self.sigmoid(2*scale*(two_norm - 0.2 * mu))
            p_amber = torch.cat([f1_amber.unsqueeze(-1), f2_amber.unsqueeze(-1)], dim=-1).min(-1)[0]
            
            # p_amber = torch.exp(-(rating - 0.5).pow(2) * 0.13)
            
            # Triangle shaped membership function
            # f1_amber = 5 * (two_norm - 0.3)
            # f2_amber = 1 - 5 * (two_norm - 0.5)
            # f3_amber = torch.zeros_like(two_norm).cuda()
            # f_amber = torch.cat([f1_amber.unsqueeze(-1), f2_amber.unsqueeze(-1)], dim=-1).min(-1, keepdim=True)[0]
            # p_amber = torch.cat([f_amber, f3_amber.unsqueeze(-1)], dim=-1).max(-1)[0]
            
            probs = torch.cat([
                p_green.unsqueeze(-1),
                p_red.unsqueeze(-1),
                p_amber.unsqueeze(-1)
            ], dim=-1).squeeze(-2)
            probs = F.normalize(probs, p=1, dim=-1)
            
            # y = observation_sample(f"y{t + 1}", dist.Categorical(probs=probs)) # Sample observation from likelihood
            # y = observation_sample(f"y{t + 1}", dist.OneHotCategorical(probs=probs))

            if self.design_net.training:
                y = observation_sample(f"y{t + 1}", GumbelSoftmax(probs=probs))
            else:
                y = observation_sample(f"y{t + 1}", dist.OneHotCategorical(probs=probs))

            y_outcomes.append(y)
            xi_designs.append(xi)

        return y_outcomes

    def eval(self, n_trace=10, theta=None, verbose=False):
        """run the policy, print output and return in a pandas df"""
        self.design_net.eval()
        if theta is not None:
            model = pyro.condition(self.model, data={"theta": theta})
        else:
            model = self.model

        # <editor-fold desc="[FB] Load appearance model ...">
        output_path = "./face/output"  # path to "app_model.pkl" file  TODO
        with open(os.path.join(output_path, "app_model.pkl"), "rb") as f:
            app_model = pickle.load(f)
        # </editor-fold>

        output = []
        true_thetas = []
        with torch.no_grad():
            for i in range(n_trace):
                print("\nExample run {}".format(i + 1))
                trace = pyro.poutine.trace(model).get_trace()
                true_theta = trace.nodes["theta"]["value"].cpu()

                # <editor-fold desc="[FB] Save target face params ...">
                if i % 1 == 0:
                    recon = app_model.decode(true_theta)
                    _img = Image.fromarray((recon * 255).astype(np.uint8))
                    _img.save(os.path.join(output_path, f"target_{i}.jpg"))
                # </editor-fold>

                if verbose:
                    print(f"*True Theta: {true_theta}*")
                run_xis = []
                run_ys = []
                # Print optimal designs, observations for given theta
                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].cpu().reshape(-1)
                    run_xis.append(xi)
                    y = trace.nodes[f"y{t + 1}"]["value"].cpu().numpy()
                    run_ys.append(y)
                    if verbose:
                        print(f"xi{t + 1}: {xi}")
                        print(f" y{t + 1}: {y}")

                    # <editor-fold desc="[FB] Save design face params ..">
                    if i % 1 == 0 and t % 5 == 0:
                        recon = app_model.decode(xi)
                        _img = Image.fromarray((recon * 255).astype(np.uint8))
                        _img.save(os.path.join(output_path, f"target_{i}_recon_{t}.jpg"))
                    # </editor-fold>

                run_df = pd.DataFrame(torch.stack(run_xis).numpy())
                run_df.columns = [f"xi_{i}" for i in range(self.design_dim[1])]
                run_df["observations"] = run_ys
                run_df["order"] = list(range(1, self.T + 1))
                run_df["run_id"] = i + 1
                output.append(run_df)
                true_thetas.append(true_theta.numpy())

                # <editor-fold desc="[FB] Plot ...">
                if i % 1 == 0:
                    plot_trace(i, 3, self.T, run_df, true_theta.numpy(), categorical_face_finding=True, face_folder='./face/output')
                # </editor-fold>

        print(pd.concat(output))
        return pd.concat(output), true_thetas


def single_run(
    seed,
    num_steps,
    num_inner_samples,  # L in denom
    num_outer_samples,  # N to estimate outer E
    lr,  # learning rate of sgd optim
    gamma,  # scheduler for sgd optim
    T,  # number of experiments
    p, # Number of dimensions of the design
    device,
    hidden_dim,
    encoding_dim,
    num_layers,
    arch,
    mlflow_experiment_name,
    complete_enum=False,
    include_t=False,
    use_rnn=False
):

    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)
    mlflow.set_experiment(mlflow_experiment_name)
    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")

    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("arch", arch)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("complete_enum", complete_enum)
    mlflow.log_param("num_outer_samples", num_outer_samples)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("include_t", include_t)
    mlflow.log_param("p", p)

    ## set up model
    if arch == "static":
        design_net = BatchDesignBaseline(T, 2).to(device)
    else:
        # design_dim = (1, 1) # 1D space
        design_dim = (1, p) # p-D space
        if not use_rnn:
            encoder = EncoderNetwork(
                design_dim=design_dim,
                osbervation_dim=1,
                hidden_dim=hidden_dim,
                encoding_dim=encoding_dim,
                include_t=include_t,
                T=T,
                n_hidden_layers=num_layers,
                activation=nn.LeakyReLU
            )
            emitter = EmitterNetwork(
                input_dim=encoding_dim,
                hidden_dim=hidden_dim,
                design_dim=design_dim,
                n_hidden_layers=num_layers,
                activation=nn.LeakyReLU
            )
        else:
            encoder = GRUEncoderNetwork(design_dim=design_dim, observation_dim=3, 
                                        hidden_dim=hidden_dim, encoding_dim=encoding_dim)
            emitter = GRUEmitterNetwork(encoding_dim=encoding_dim, hidden_dim=hidden_dim,
                                        design_dim=design_dim)

        if arch == "sum":
            if not use_rnn:
                design_net = SetEquivariantDesignNetwork(
                    encoder, emitter, empty_value=torch.ones(design_dim)
                ).to(device)
            else:
                design_net = SetEquivariantDesignRNN(
                    encoder, emitter, empty_value=torch.ones(1, design_dim[1]) * 0.01
                ).to(device)
        else:
            raise ValueError(f"Unexpected architecture specification: '{arch}'.")
    
    ### Prior hyperparams ###    
    theta_prior_loc = torch.zeros(design_dim, device=device)  # mean of the prior
    theta_prior_covmat = torch.eye(design_dim[1], device=device)  # covariance of the prior
    
    # noise of the model: the sigma in N(G(theta, xi), sigma)
    noise_scale_tensor = 0.1 * torch.tensor(
        1.0, dtype=torch.float32, device=device
    )
    
    temporal_model = HyperbolicTemporalDiscounting(
        design_net=design_net,
        T=T,
        design_dim=design_dim,
        theta_loc=theta_prior_loc,
        theta_covmat=theta_prior_covmat,
        noise_scale=noise_scale_tensor,
        use_rnn=use_rnn
    )

    # Annealed LR optimiser --------
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": lr, "betas": [0.9, 0.999], "weight_decay": 0,},
            "gamma": gamma,
        }
    )
    if complete_enum:
        pce_loss = PriorContrastiveEstimationDiscreteObsTotalEnum(
            num_outer_samples=num_outer_samples, num_inner_samples=num_inner_samples
        )
    else:
        pce_loss = PriorContrastiveEstimationScoreGradient(
            num_outer_samples=num_outer_samples, num_inner_samples=num_inner_samples
        )

    oed = OED(temporal_model.model, scheduler, pce_loss)

    # ----------
    # optimise
    loss_history = []
    t = trange(0, num_steps, desc="Loss: 0.000 ")
    for i in t:
        loss = oed.step()
        loss = torch_item(loss)
        t.set_description("Loss: {:.3f} ".format(loss))
        loss_history.append(loss)
        if i % 100 == 0:
            mlflow.log_metric("loss", oed.evaluate_loss())  # oed.evaluate_loss()
        if i % 1000 == 0:
            scheduler.step()

    mlflow.log_metric(
        "loss_diff50", np.mean(loss_history[-51:-1]) / np.mean(loss_history[0:50]) - 1
    )
    # evaluate and store results
    runs_output = temporal_model.eval()  ###
    results = {
        "design_network": design_net.cpu(),
        "seed": seed,
        "loss_history": loss_history,
        "runs_output": runs_output,
    }
    # log model ----------------------------
    print("Storing model to MlFlow... ", end="")
    # store the model:
    mlflow.pytorch.log_model(temporal_model.cpu(), "model")
    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model"
    print(f"Model sotred in {model_loc}.")

    print(f"Run completed {mlflow.active_run().info.artifact_uri}.")
    print(f"The experiment-id of this run is {ml_info.experiment_id}")
    # --------------------------------------------------------------------------
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design example: Hyperbolic Temporal Discounting."
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=5, type=int)
    parser.add_argument("--num-inner-samples", default=1024, type=int)
    parser.add_argument("--num-outer-samples", default=256, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--num-experiments", default=30, type=int)  # == T
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--encoding-dim", default=32, type=int)
    parser.add_argument("--complete-enum", default=False, action="store_true")
    parser.add_argument("--include-t", default=False, action="store_true")
    parser.add_argument("--p", default=3, type=int, help="Number of dimensions of designs")
    parser.add_argument(
        "--num-layers", default=2, type=int, help="Number of hidden layers."
    )
    parser.add_argument(
        "--arch",
        default="sum",
        type=str,
        help="Architecture",
        choices=["static", "sum"],
    )
    parser.add_argument(
        "--use-rnn", default=False, type=bool, help="Whether to use rnn")
    parser.add_argument("--mlflow-experiment-name", default="face_finding-transformer-categorical", type=str)
    args = parser.parse_args()

    single_run(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
        T=args.num_experiments,
        p=args.p,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        num_layers=args.num_layers,
        arch=args.arch,
        mlflow_experiment_name=args.mlflow_experiment_name,
        complete_enum=args.complete_enum,
        include_t=args.include_t,
        use_rnn=args.use_rnn
    )
