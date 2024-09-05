import torch
import numpy as np
import torch.nn as nn
import random
import scipy as sp
from pprint import pprint
from torch.autograd.functional import jacobian
from torch.func import jacfwd, jacrev, vmap
from tqdm import tqdm
from scipy.stats import ortho_group
import argparse

from torch.nn import Linear
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.nn import LeakyReLU
from torch.nn import Module
import torch
from pathlib import Path
import pickle
from torch.utils.data import Dataset, DataLoader

np.set_printoptions(precision=4, suppress=True)


class SyntheticZPairDataset(Dataset):
    def __init__(self, z_t_pair_data, u_t_data):
        self.z_t_pair_data = z_t_pair_data
        self.u_t_data = u_t_data

    def __len__(self):
        return len(self.u_t_data)

    def __getitem__(self, idx):
        return self.z_t_pair_data[idx], self.u_t_data[idx]


class LinearProbing(Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbing, self).__init__()
        self.linear = Linear(input_dim, output_dim)
        self.relu = LeakyReLU(0.1)

    def forward(self, x):
        return self.linear(self.relu(x))


# create a MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth=1):
        super(MLP, self).__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            h = self.relu(hidden_layer(h))
        out = self.fc2(h)
        return out


class JacobianMLP(nn.Module):
    def __init__(self, jacobian_support, hid_dim):
        super(JacobianMLP, self).__init__()
        jacobian_support = torch.tensor(jacobian_support)
        out_dim, in_dim = jacobian_support.shape
        self.out_dim = out_dim
        self.input_layers = nn.ModuleList()
        self.output_layer = nn.ModuleList()
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.jacobian_support = jacobian_support
        for i in range(out_dim):
            linear_layer = nn.Linear(in_dim, hid_dim)
            # zero out the weights
            linear_layer.weight.data.zero_()
            # Use boolean indexing for efficient weight initialization
            mask = jacobian_support[i] == 1
            assert mask.sum() > 0, "Each output must depend on at least one input"
            normal_weights = torch.randn(hid_dim, in_dim)
            small_values_mask = normal_weights.abs() < 0.01
            adjusted_values = 0.02 * (normal_weights >= 0).float() - 0.01
            normal_weights[small_values_mask] = adjusted_values[small_values_mask]
            normal_weights = normal_weights * mask.float()
            linear_layer.weight.data = normal_weights  # Transpose to match shape

            self.input_layers.append(linear_layer)
            out_put_layer = nn.Linear(hid_dim, 1)
            self.output_layer.append(out_put_layer)

    def forward(self, x):
        outs = []
        for i in range(self.out_dim):
            hidden = self.relu(self.input_layers[i](x))
            out = self.output_layer[i](hidden)
            outs.append(out)
        outs = torch.cat(outs, dim=-1)
        return outs


def create_transition_matrix_and_stationary_distribution(
    num_states, state_stickiness=0.9, shift=False, clean=False
):
    # Create a transition matrix with random probabilities
    if clean:
        transition_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            for j in range(num_states):
                if j == i:
                    transition_matrix[i, j] = state_stickiness
                if j == i + 1:
                    transition_matrix[i, j] = 1.0 - state_stickiness
                if i == num_states - 1 and j == 0:
                    transition_matrix[i, j] = 1.0 - state_stickiness
        transition_matrix /= transition_matrix.sum(1, keepdims=True)
    else:
        # Create a transition matrix with random probabilities
        transition_matrix = np.random.uniform(
            low=0.2, high=0.8, size=(num_states, num_states)
        )

        identity_matrix = np.eye(num_states) * state_stickiness
        if shift:
            identity_matrix = np.rot90(identity_matrix)
        transition_matrix += identity_matrix  # Add the identity matrix to make sure that the matrix is irreducible
        # Normalize each row to sum to 1
        transition_matrix = transition_matrix / transition_matrix.sum(
            axis=1, keepdims=True
        )

    # Find the eigenvalues and eigenvectors of the transpose of the transition matrix
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    # Find the index of the eigenvalue 1
    stationary_index = np.argmin(np.abs(eigenvalues - 1))
    # Corresponding eigenvector is the stationary distribution
    stationary_vector = np.real(eigenvectors[:, stationary_index])
    # Normalize the vector to make the sum of probabilities equal to 1
    stationary_distribution = stationary_vector / stationary_vector.sum()

    # check if the stationary distribution is correct
    assert np.allclose(
        np.dot(stationary_distribution, transition_matrix), stationary_distribution
    )

    return transition_matrix, stationary_distribution


def simulate_markov_chain(transition_matrix, stationary_distribution, num_steps):
    num_states = transition_matrix.shape[0]
    current_state = np.random.choice(num_states, p=stationary_distribution)
    states_visited = [current_state]

    for _ in range(num_steps - 1):
        current_state = np.random.choice(num_states, p=transition_matrix[current_state])
        states_visited.append(current_state)

    return states_visited


def sample_batched_visited_states(
    num_batches, num_steps, transition_matrix, stationary_distribution
):
    batch_states_visited = []
    for _ in range(num_batches):
        states_visited = simulate_markov_chain(
            transition_matrix, stationary_distribution, num_steps
        )
        batch_states_visited.append(states_visited)
    batch_states_visited = np.array(batch_states_visited)
    return batch_states_visited


def estimate_markov_properties(batch_states_visited):
    """
    Estimates the Markov properties of a given batch of states visited.

    Parameters:
    - batch_states_visited (numpy.ndarray): A 2D array containing the states visited in each batch.

    Returns:
    - estimated_transition_matrix (numpy.ndarray): The estimated transition matrix.
    - estimated_stationary_distribution (numpy.ndarray): The estimated stationary distribution.
    """

    num_states = batch_states_visited.max() + 1
    # Initialize the distributions
    estimated_stationary_distribution = np.zeros(num_states)
    estimated_transition_matrix = np.zeros((num_states, num_states))

    # Process each batch of states visited
    for states_visited in batch_states_visited:
        # Update stationary distribution counts
        unique, counts = np.unique(states_visited, return_counts=True)
        estimated_stationary_distribution[unique] += counts

        # Transition matrix count updates
        for i in range(len(states_visited) - 1):
            estimated_transition_matrix[states_visited[i], states_visited[i + 1]] += 1

    # Normalize the estimated stationary distribution
    estimated_stationary_distribution /= estimated_stationary_distribution.sum()

    # Normalize the estimated transition matrix
    row_sums = estimated_transition_matrix.sum(axis=1, keepdims=True)
    valid_rows = (row_sums != 0).flatten()  # Find rows where the sum is not zero
    estimated_transition_matrix[valid_rows] /= row_sums[
        valid_rows
    ]  # Only normalize valid rows

    return estimated_transition_matrix, estimated_stationary_distribution


def generate_distinct_sparse_masks(num_masks, in_dim, out_dim, sparsity, max_iter=1000):

    assert in_dim <= out_dim, "Input dimension must be less than output dimension"

    masks = []
    iter_count = 0
    while len(masks) < num_masks:
        if iter_count > max_iter:
            raise ValueError(
                "Cannot generate distinct masks within the maximum allowed iterations"
            )

        # Generate a candidate mask
        mask = np.random.rand(out_dim, in_dim) < sparsity
        mask = mask.astype(np.float32)

        # get a permutation for the direct edges
        permutation = np.random.permutation(in_dim)
        for i in range(in_dim):
            mask[permutation[i], i] = 1.0

        # Check for distinctiveness against all existing masks
        is_distinct = True
        thd = 0.1 * np.prod(mask.shape)
        for m in masks:
            diff = mask - m
            num_of_ones = np.sum(diff == 1)
            num_of_neg_ones = np.sum(diff == -1)
            if num_of_ones <= thd or num_of_neg_ones <= thd:
                is_distinct = False
                break
        # If mask is distinct, add it to the list
        if is_distinct:
            masks.append(mask)
        iter_count += 1

    return masks


def generate_distinct_sparse_jacobian_mlps(
    num_mlp=5, in_dim=7, hid_dim=7, out_dim=7, sparsity=0.5
):
    # first generate distinct jacobian support for each MLP, note that this is different from the sparse pattern for each layer
    distinct_jacobian_supports = generate_distinct_sparse_masks(
        num_mlp, in_dim, out_dim, sparsity
    )
    # second construct the MLPs based on the distinct jacobian supports
    distinct_jacobian_mlps = []
    for i in range(num_mlp):
        distinct_jacobian_mlps.append(
            JacobianMLP(jacobian_support=distinct_jacobian_supports[i], hid_dim=hid_dim)
        )

    return distinct_jacobian_mlps


def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope


leaky1d = np.vectorize(leaky_ReLU_1d)


def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)


# Next lets construct a synthetic dataset with states as input, we will generate transition functions for each state and sample observations


def main(args):
    use_gpu = args.use_gpu
    B = args.inner_batch_size
    num_states = args.num_states
    num_batches = args.num_batches
    base_steps = args.base_steps
    mc_num_steps = base_steps
    mc2_num_steps = base_steps
    rand_num_steps = base_steps
    z_dim = args.z_dim
    hid_dim = args.hid_dim
    noise_level = args.noise_level
    num_mixing_layer = args.num_mixing_layer
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # First sample the MC part
    transition_matrix, stationary_distribution = (
        create_transition_matrix_and_stationary_distribution(
            num_states, state_stickiness=0.8, clean=True
        )
    )
    batch_states_visited = sample_batched_visited_states(
        num_batches, mc_num_steps, transition_matrix, stationary_distribution
    )

    transition_matrix_2, stationary_distribution_2 = (
        create_transition_matrix_and_stationary_distribution(
            num_states, state_stickiness=0.8, clean=False, shift=True
        )
    )
    batch_states_visited_2 = sample_batched_visited_states(
        num_batches, mc2_num_steps, transition_matrix_2, stationary_distribution_2
    )

    # Second sample the uniformly sampled states
    batch_states_uniformly_sampled = np.random.choice(
        num_states, size=(num_batches, rand_num_steps)
    )

    batch_states = np.concatenate(
        [batch_states_uniformly_sampled, batch_states_visited, batch_states_visited_2],
        axis=1,
    )  # (num_batches, num_steps*2)

    estimated_transition_matrix, estimated_stationary_distribution = (
        estimate_markov_properties(batch_states)
    )
    # print("True stationary distribution:")
    # pprint(stationary_distribution)

    print("True transition matrix:")
    pprint(transition_matrix)
    print("True transition matrix_2:")
    pprint(transition_matrix_2)
    # print("Estimated stationary distribution:")
    # pprint(estimated_stationary_distribution)

    print("Estimated transition matrix:")
    pprint(estimated_transition_matrix)

    distinct_sparse_jacobian_mlps = generate_distinct_sparse_jacobian_mlps(
        num_mlp=num_states, in_dim=z_dim, hid_dim=hid_dim, out_dim=z_dim, sparsity=0.7
    )
    if use_gpu:
        distinct_sparse_jacobian_mlps = [
            mlp.to("cuda") for mlp in distinct_sparse_jacobian_mlps
        ]

    def generateUniformMat(Ncomp, condT):
        """
        generate a random matrix by sampling each element uniformly at random
        check condition number versus a condition threshold
        """
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

        while np.linalg.cond(A) > condT:
            # generate a new A matrix!
            A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
            for i in range(Ncomp):
                A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

        return A

    # condList = []
    # for i in range(int(10000)):
    #     # A = np.random.uniform(0,1, (Ncomp, Ncomp))
    #     A = np.random.uniform(1, 2, (z_dim, z_dim))  # - 1
    #     for i in range(z_dim):
    #         A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
    #     condList.append(np.linalg.cond(A))

    # condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile

    mixingList = []
    for l in range(num_mixing_layer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(z_dim)  #
        # A = generateUniformMat(z_dim, condThresh)
        mixingList.append(A)

    batched_z = []
    for batch_id in tqdm(range(num_batches)):
        states = batch_states[batch_id]
        z_0 = torch.randn(B, z_dim)
        if use_gpu:
            z_0 = z_0.to("cuda")
        z_list = [z_0]
        for state in states:
            state_jacobian_mlp = distinct_sparse_jacobian_mlps[state]
            z_current = z_list[-1]
            z_next = state_jacobian_mlp(
                z_current
            ) + noise_level**0.5 * torch.randn_like(z_current)
            z_list.append(z_next)
        z_list = torch.stack(z_list, dim=1)
        batched_z.append(z_list.detach().cpu())
    batched_z = torch.stack(batched_z, dim=0)

    mixedDat = batched_z.numpy()
    for l in range(num_mixing_layer - 1):
        mixedDat = leaky_ReLU(mixedDat, 0.2)
        mixedDat = np.dot(mixedDat, mixingList[l])
    batched_x = torch.tensor(mixedDat, dtype=torch.float32)
    print(mixedDat.shape)
    print(batched_z.shape)

    # then let's test if we can learn the states from input-output pairs when we can assess z_t
    z_t_pair_data = []
    u_t_data = []
    # generate a dataset with z_{t-1}, z_t as input and u_t as output
    for batch_id in tqdm(range(num_batches)):
        states = batch_states[batch_id]
        for t in range(len(states)):
            for b in range(B):
                u_t = states[t]
                z_curr = batched_z[batch_id, b, t + 1]
                z_prev = batched_z[batch_id, b, t]
                z_t_pair = torch.cat([z_prev, z_curr], dim=-1)
                z_t_pair_data.append(z_t_pair)
                u_t_data.append(u_t)
    z_t_pair_data = torch.stack(z_t_pair_data, dim=0)
    u_t_data = torch.tensor(u_t_data).long()
    print(z_t_pair_data.shape, u_t_data.shape)

    simulation_data_path = Path(
        f"../data/simulation/C_{num_states}_z_dim_{z_dim}_hid_dim_{hid_dim}_num_batches_{num_batches}"
    )
    simulation_data_path.mkdir(parents=True, exist_ok=True)
    with open(simulation_data_path / "data.pkl", "wb") as f:
        pickle.dump(
            {
                # Markov Chain
                "mc_batch_size": num_batches,
                "mc_num_states": num_states,
                "mc_transition_matrix": transition_matrix,
                # "mc_stationary_distribution": stationary_distribution,
                # "mc_steps": mc_num_steps,
                "batch_states": batch_states,
                # "random_steps": rand_num_steps,
                # Z
                "inner_sample_batch_size": B,
                "batch_z": batched_z,
                "batch_x": batched_x,
                # Auxillary data
                "z_t_pair_data": z_t_pair_data,
                "u_t_data": u_t_data,
            },
            f,
        )

    # with open("data/simulation/z_t_pair_u_t_data.pt","wb") as f:
    #     torch.save((z_t_pair_data, u_t_data), f)
    with open(simulation_data_path / "data.pkl", "rb") as f:
        data = pickle.load(f)
        z_t_pair_data = data["z_t_pair_data"]
        u_t_data = data["u_t_data"]

    synthetic_z_pair_dataset = SyntheticZPairDataset(z_t_pair_data, u_t_data)
    # subset_size = 10000
    # synthetic_z_pair_dataset = torch.utils.data.Subset(synthetic_z_pair_dataset, range(subset_size))
    synthetic_z_pair_dataloader = DataLoader(
        synthetic_z_pair_dataset, batch_size=32, shuffle=True
    )
    # optimal_C_model = LinearProbing(input_dim=14, output_dim=5).to('cuda')
    optimal_C_model = MLP(
        input_size=z_t_pair_data.shape[-1], hidden_size=256, output_size=5, depth=1
    )
    if use_gpu:
        optimal_C_model = optimal_C_model.to("cuda")
    criterion = CrossEntropyLoss()
    optimizer = Adam(optimal_C_model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1000):
        for z_t_pair, u_t in synthetic_z_pair_dataloader:
            if use_gpu:
                z_t_pair = z_t_pair.to("cuda")
                u_t = u_t.to("cuda")
            optimizer.zero_grad()
            u_t_hat = optimal_C_model(z_t_pair)
            loss = criterion(u_t_hat, u_t)
            loss.backward()
            optimizer.step()
        accuracy = (u_t_hat.argmax(dim=-1) == u_t).float().mean()
        print(f"Epoch {epoch}, Loss {loss.item():.4f}, Accuracy {accuracy.item():.4f}")
        if accuracy.item() == 1.0:
            print("Perfect accuracy reached")
            break
    assert accuracy.item() == 1.0, "Optimal C model did not reach perfect accuracy"

    # Save optimal C model and gt transition MLPs
    torch.save(
        optimal_C_model.state_dict(), simulation_data_path / "optimal_C_model.pt"
    )
    optimal_transition_mlp_state_dicts = [
        mlp.state_dict() for mlp in distinct_sparse_jacobian_mlps
    ]
    torch.save(
        optimal_transition_mlp_state_dicts, simulation_data_path / "gt_transition_ms.pt"
    )
    gt_transition_jacobian_supports = [
        mlp.jacobian_support for mlp in distinct_sparse_jacobian_mlps
    ]
    torch.save(
        gt_transition_jacobian_supports,
        simulation_data_path / "gt_transition_jacobian_supports.pt",
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--use_gpu", default=True, type=bool)
    argparser.add_argument("--inner_batch_size", default=32, type=int)
    argparser.add_argument("--num_states", default=5, type=int)
    argparser.add_argument("--num_batches", default=1000, type=int)
    argparser.add_argument("--base_steps", default=5, type=int)
    argparser.add_argument("--z_dim", default=8, type=int)
    argparser.add_argument("--hid_dim", default=32, type=int)
    argparser.add_argument("--noise_level", default=0.1, type=float)
    argparser.add_argument("--num_mixing_layer", default=2, type=int)
    argparser.add_argument("--seed", default=0, type=int)
    args = argparser.parse_args()

    main(args)
