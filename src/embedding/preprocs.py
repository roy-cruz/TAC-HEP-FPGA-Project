import torch
import torch.nn as nn
from typing import Union
from embedding.utils.data_utils import EPS

class PFPreProcessor(nn.Module):
    def __init__(self, norm_constants: dict = {}):
        super().__init__()
        self.norm_constants = norm_constants
        PDGIDs = [
            211, # h, charged hadrons
            11,  # e
            13,  # mu
            22,  # gamma
            130, # h0
            1,   # h_HF, HF tower identified as a hadron
            2    # egamma_HF, HF tower identified as an EM particle
        ]
        self.register_buffer("avail_pdgIds", torch.tensor(PDGIDs, dtype=torch.long))
        self.num_features_cont = 5 # pt, eta, phi, dxy, dxysig
        self.num_features_disc = 2 + len(self.avail_pdgIds) # is_pf, charge(from pdgId sign), pdgId(one-hot)
        self.num_features = self.num_features_cont + self.num_features_disc
        self.batch_norm = nn.BatchNorm1d(self.num_features_cont)

    def pdgId_to_onehot(self, pdgId_tensor: torch.Tensor) -> torch.Tensor:
        pdgId_tensor = pdgId_tensor.long()  # [B, N]
        one_hot = (pdgId_tensor.unsqueeze(-1).abs() == self.avail_pdgIds).float()  # [B, N, num_pdgIds]
        return one_hot
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, 8] = [pt, eta, phi, dxy, dxysig, is_pf, pdgId]
        Returns: [B, N, 8 + num_pdgIds] with normalized/scaled values and one-hot pdgId.
        - pt:        log(pt / sum_pt) per event
        - eta:       kept as is
        - phi:       kept as is
        - dxy:       tanh(dxy)
        - dxysig:    kept as is
        - charge:    derived from pdgId: +1/-1 for e, mu, pi; 0 else
        - is_pf:     kept as is (0/1)
        - pdgId:     one-hot encoded absolute pdgId from available list
        Padding (pt==0) rows are zeroed.
        Continuous features are batch-normalized.
        """

        pt_raw = x[..., 0]
        eta_raw = x[..., 1]
        phi_raw = x[..., 2]
        dxy_raw = x[..., 3]
        dxysig_raw = x[..., 4]
        is_pf_raw = x[..., 5]
        pdgId_raw = x[..., 6]

        valid = pt_raw > 0 # [B, N]
        
        # if valid.any():
        pt = torch.where(valid, pt_raw, torch.zeros_like(pt_raw))
        pt = pt / (pt.sum(dim=-1, keepdim=True) + EPS)
        pt[valid] = torch.log(pt[valid]) 
        
        dxy = torch.where(valid, dxy_raw, torch.zeros_like(dxy_raw))
        dxy[valid] = torch.tanh(dxy_raw[valid])

        pos = (pdgId_raw == 11) | (pdgId_raw == 13) | (pdgId_raw == 211)
        neg = (pdgId_raw == -11) | (pdgId_raw == -13) | (pdgId_raw == -211)
        charge = torch.zeros_like(valid, dtype=torch.float)
        charge[valid & pos] = 1.0
        charge[valid & neg] = -1.0
        
        pdgId_onehot = self.pdgId_to_onehot(pdgId_raw)
        
        x_proc = torch.cat([
            pt.unsqueeze(-1),
            eta_raw.unsqueeze(-1),
            phi_raw.unsqueeze(-1),
            dxy.unsqueeze(-1),
            dxysig_raw.unsqueeze(-1),
            # Next ones NOT are not continuous, so NOT fed to batch norm layer.
            charge.unsqueeze(-1), # Computed
            is_pf_raw.unsqueeze(-1),
            pdgId_onehot
        ], dim=-1)

        # Masked batch norm on continuous ftrs
        x_cont = x_proc[..., :self.num_features_cont]  # [B, N, num_cont]
        B, N, C = x_cont.shape
        x_cont_flat = x_cont.reshape(B * N, C)
        valid_flat = valid.reshape(B * N)
        x_cont_flat[valid_flat] = self.batch_norm(x_cont_flat[valid_flat])
        x_proc[..., :self.num_features_cont] = x_cont_flat.reshape(B, N, C)
        # else:
            # x_proc = torch.zeros(*x.shape[:-1], self.num_features, device=x.device)
        
        return x_proc