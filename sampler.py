import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        # print(cfg)
        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        
        directions = ray_bundle.directions
        image_size = ray_bundle.image_size
        origins = ray_bundle.origins
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray)
        # z_vals = z_vals.unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
        z_vals = z_vals.view(-1, 1, 1).cuda()
        # z_vals = torch.cat([z_vals]*3, 3).cuda()
        # z_vals = torch.cat([z_vals]*directions.shape[0], 1).view(-1)

        # TODO (1.4): Sample points from z values
        # directions = torch.cat([directions]*self.n_pts_per_ray,0)
        # sample_points = directions * z_vals
        # directions = directions.view(image_size[1], image_size[0], 3).unsqueeze(0)
        directions = directions.unsqueeze(0)
        directions = torch.cat([directions]*self.n_pts_per_ray, 0)

        origins = origins.unsqueeze(0)
        origins = torch.cat([origins]*self.n_pts_per_ray, 0)

        # print(z_vals.shape)
        # print(directions.shape)

        sample_points = origins + directions * z_vals

        sample_points = sample_points.view(-1,self.n_pts_per_ray,3)

        # print(f"sample points shape = {sample_points.shape}")

        sample_lengths = z_vals.view(1,-1,1) * torch.ones_like(sample_points[..., :1])
        
        # print(f"sample lengths shape = {sample_lengths.shape}")

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=sample_lengths,
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}