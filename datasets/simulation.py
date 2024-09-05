from pathlib import Path
import pickle
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


class SimulationDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = Path(root_dir)

        raw_data = pickle.load(open(self.root_dir / "data.pkl", "rb"))
        self.transition_matrix = raw_data["mc_transition_matrix"] if "mc_transition_matrix" in raw_data else np.eye(raw_data["batch_states"].max().item() + 1)
        batch_states = torch.tensor(
            raw_data["batch_states"], dtype=torch.long
        )  # (mc_batch_size, num_steps)
        mc_batch_size, num_steps = batch_states.shape

        batch_z = torch.tensor(
            raw_data["batch_z"], dtype=torch.float32
        )  # (mc_batch_size, inner_sample_batch_size, num_steps + 1, z_dim)
        _, inner_sample_batch_size, z_steps, z_dim = batch_z.shape

        self.batch_states = (
            batch_states.unsqueeze(1)
            .expand(mc_batch_size, inner_sample_batch_size, num_steps)
            .reshape(-1, num_steps)
        )
        self.batch_z = batch_z.reshape(-1, z_steps, z_dim)

        batch_x = torch.tensor(raw_data["batch_x"], dtype=torch.float32)
        self.batch_x = batch_x.reshape(-1, z_steps, batch_x.shape[-1])

    def __len__(self):
        return len(self.batch_z)

    def __getitem__(self, idx):
        x_t = self.batch_x[idx]
        z_t = self.batch_z[idx]
        c_t = self.batch_states[idx]
        return x_t, z_t, c_t


if __name__ == "__main__":
    dataset = SimulationDataset(
        "data/simulation/C_5_z_dim_7_hid_dim_32_num_batches_1000_mc_steps_2_rand_steps_2"
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for x_t, z_t, u_t in dataloader:
        print(z_t.shape, u_t.shape)
        break
