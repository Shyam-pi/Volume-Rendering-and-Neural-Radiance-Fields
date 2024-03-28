import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        ones_rd = torch.ones_like(rays_density)
        temp = ones_rd * (torch.exp(-rays_density * deltas))
        T = torch.cat((torch.ones_like(rays_density[:1,...]), temp), dim=0)
        T = torch.cumprod(T, dim=0)

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        weights = T[:-1,...]*(1 - torch.exp(-rays_density * deltas))
        # print(f"Weights max = {weights.max()}, weights min = {weights.min()}")
        # exit()
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        # print(f"weights shape = {weights.shape}")
        # print(f"features shape = {rays_feature.shape}")
        feature = torch.sum((weights*rays_feature), dim = 0)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # print(f"Shape of B = {B}")
        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[0]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature'].view(-1,n_pts,3)

            # print(f"Feature shape = {feature.shape}")
            # print(f"Density values shape = {density.shape}")

            # print(f"Density max = {density.max()} Density min = {density.min()}")

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]

            # print(f"Depth values shape = {depth_values.shape}")
            # deltas = torch.cat(
            #     (
            #         depth_values[1:, ...] - depth_values[:-1, ...],
            #         1e10 * torch.ones_like(depth_values[:1,...]),
            #     ),
            #     dim=0,
            # )[..., None]

            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # print(f"Deltas = {deltas.shape}")

            # print(f"Deltas shape = {deltas.view(-1, n_pts, 1).shape}")

            # print(f"Density shape = {density.view(-1, n_pts, 1).shape}")

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights=weights, rays_feature=feature)

            # TODO (1.5): Render depth map
            depth = self._aggregate(weights=weights, rays_feature=depth_values.view(-1,n_pts).unsqueeze(-1))


            # print(f"Color map max = {feature.max()}, Color map min = {feature.min()}")

            # print(f"Depth map shape = {depth.shape}")
            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}
