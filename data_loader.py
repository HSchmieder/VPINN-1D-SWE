import numpy as np
import torch

# load probe data
def load_probe_data(n_probe, T_sample, noise, seed=42):
    gen = torch.Generator()
    gen.manual_seed(42)
    file_path = 'data/data_' + str(n_probe) + 'probes_'
    t_probe = torch.from_numpy(np.loadtxt(file_path+'t.txt', delimiter=',', dtype=np.float32)[::T_sample,:]).reshape(-1).unsqueeze(1)
    x_probe = torch.from_numpy(np.loadtxt(file_path+'x.txt', delimiter=',', dtype=np.float32)[::T_sample,:]).reshape(-1).unsqueeze(1)
    h_probe_clean = torch.from_numpy(np.loadtxt(file_path+'h.txt', delimiter=',', dtype=np.float32)[::T_sample,:]).reshape(-1)
    pts_probe = torch.cat((t_probe,x_probe), dim=1)
    # add noise from Gaussian distribution
    h_probe = h_probe_clean * torch.normal(mean=1,std=noise,size=h_probe_clean.size(), generator=gen)
    return pts_probe, h_probe

# load reference data
def load_reference_data():
    file_path = 'data/data_reference_'
    t_ref = torch.from_numpy(np.loadtxt(file_path+'t.txt', delimiter=',', dtype=np.float32))
    x_ref = torch.from_numpy(np.loadtxt(file_path+'x.txt', delimiter=',', dtype=np.float32))
    h_ref = torch.from_numpy(np.loadtxt(file_path+'h.txt', delimiter=',', dtype=np.float32))
    u_ref = torch.from_numpy(np.loadtxt(file_path+'u.txt', delimiter=',', dtype=np.float32))
    return t_ref, x_ref, h_ref, u_ref
