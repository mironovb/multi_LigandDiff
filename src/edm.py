import torch
import torch.nn.functional as F
import numpy as np
import math
from torch_scatter import scatter_add
from src import utils
from src.egnn import Dynamics
from src.projection import project_exclusion_shell, d_min_schedule
from src.bonding import bond_cutoff
from src.noise import GammaNetwork, PredefinedNoiseSchedule
from src import const
from typing import Union


class EDM(torch.nn.Module):
    def __init__(
            self,
            dynamics: Union[Dynamics],
            in_node_nf: int,
            n_dims: int,
            timesteps: int = 1000,
            noise_schedule='learned',
            noise_precision=1e-4,
            loss_type='vlb',
            norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.),
    ):
        super().__init__()
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned with a vlb objective'
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)

        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases

    def noised_representation(self,xh,ligand_diff,context,batch_seg,gamma_t):
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma(gamma_t)
        eps_t = self.sample_combined_position_feature_noise(xh,ligand_diff)
        z_t = alpha_t[batch_seg] * xh + sigma_t[batch_seg] * eps_t
        z_t = xh * context + z_t * ligand_diff
        return z_t,eps_t
    
    def forward(self, x, h,  context, ligand_diff, batch_seg,batch_size, ligand_site=None):
        # Normalization and concatenation
        x, h = self.normalize(x, h)
        ligand_site = (ligand_site.float() - self.norm_biases[2]) / self.norm_values[2]
        xh = torch.cat([x, h], dim=1)
        delta_log_px=(self.n_dims*self.inflate_batch_array(ligand_diff,batch_seg)*np.log(self.norm_values[0]))
        # Sample t
        lowest_t=0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(batch_size, 1), device=x.device).float()
        s_int = t_int - 1
        t = t_int / self.T
        s = s_int / self.T

        # Masks for t=0 and t>0
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero
        

        # Compute gamma_t and gamma_s according to the noise schedule
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        z_t,eps_t=self.noised_representation(xh,ligand_diff,context,batch_seg,gamma_t)
        # Neural net prediction
        eps_t_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            batch_seg=batch_seg,
            ligand_site=ligand_site,   
        )

        eps_t_hat = eps_t_hat * ligand_diff
        # Computing basic error (further used for computing NLL and L2-loss)
        squared_error=(eps_t-eps_t_hat)**2
        error_t=self.inflate_batch_array(squared_error,batch_seg)
        SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1)
        assert error_t.size() == SNR_weight.size()
        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)]
        neg_log_constants = -self.log_constant_of_p_x_given_z0(ligand_diff,batch_seg,batch_size)
        # The KL between q(z_T | x) and p(z_T) = Normal(0, 1) (should be close to zero)
        kl_prior = self.kl_prior(xh, ligand_diff,batch_seg)
        if self.training:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and selected only relevant via masking
            log_p_x_given_z0_without_constants,log_ph_given_z0 = self.log_p_xh_given_z0_without_constants(h, z_t, gamma_t, eps_t, eps_t_hat, ligand_diff,batch_seg)  
            loss_0_x = -log_p_x_given_z0_without_constants * t_is_zero.squeeze()               
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()
            #apply t_is_zero mask
            error_t=error_t*t_is_not_zero.squeeze()
            
        else:
            t_zeros=torch.zeros_like(s)
            gamma_0=self.gamma(t_zeros)
            z_0,eps_0=self.noised_representation(xh,ligand_diff,context,batch_seg,gamma_0)
            eps_0_hat = self.dynamics.forward(t_zeros,z_0, batch_seg, ligand_site)
            eps_0_hat = eps_0_hat * ligand_diff
            log_p_x_given_z0_without_constants, log_ph_given_z0 = \
                self.log_p_xh_given_z0_without_constants(h, z_0, gamma_0, eps_0, eps_0_hat, ligand_diff,batch_seg)
            loss_0_x = -log_p_x_given_z0_without_constants
            loss_0_h = -log_ph_given_z0
        loss_terms = (
            delta_log_px, error_t, SNR_weight,
            loss_0_x, loss_0_h, neg_log_constants,
            kl_prior
        )
        return loss_terms
    
    
    def sample_normal(self,mu_xh,ligand_diff,sigma,batch_seg):
        eps=self.sample_combined_position_feature_noise(mu_xh,ligand_diff)
        out_xh=mu_xh+sigma[batch_seg]*eps
        return out_xh




    @torch.no_grad()
    def sample_chain(self, x, h, context, ligand_diff, batch_seg,batch_size, ligand_site, keep_frames=None,timesteps=None, resample_r=1,
                     project_enabled=False, ligand_group=None, d_min_start=2.2, d_min_end=1.9,
                     valence_guard=False):
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < keep_frames <= timesteps
        assert timesteps % keep_frames == 0

        x, h, = self.normalize(x, h)
        xh = torch.cat([x, h], dim=1)
        #ligand_site = (ligand_site.float() - self.norm_biases[2]) / self.norm_values[2]
        # Initial linker sampling from N(0, I)
        # should be torch.tensor([0,0,0])
        mu_x=scatter_add(x*context, batch_seg, dim=0)/scatter_add(context, batch_seg, dim=0)
        mu_h=torch.zeros((batch_size,self.in_node_nf),device=x.device)
        mu_xh=torch.cat([mu_x,mu_h],dim=1)[batch_seg]
        sigma=torch.ones((batch_size,1),device=x.device)
        z=self.sample_normal(mu_xh,ligand_diff,sigma,batch_seg)
        z=xh*context+z*ligand_diff

        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Sample p(z_s | z_t) with RePaint resampling (Lugmayr et al. CVPR 2022)

        for s in reversed(range(0, timesteps)):
            s_array = torch.full((batch_size, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            for u in range(resample_r):
                # Denoise: z_t -> z_s
                z = self.sample_p_zs_given_zt_only_ligand_diff(
                    s=s_array,
                    t=t_array,
                    z_t=z,
                    context=context,
                    ligand_diff=ligand_diff,
                    batch_seg=batch_seg,
                    ligand_site=ligand_site,
                )
                # Hard exclusion-shell projection (Christopher et al. 2024).
                #
                # SCALE: the reverse loop runs in NORMALIZED coordinates
                # (positions ÷ norm_values[0]; see self.normalize), but
                # project_exclusion_shell -- and its d_min / BOND_PERCEPTION_CUTOFFS
                # -- are defined in Ångströms, because the shell must enforce the
                # same real-space separation get_bond_order later perceives on the
                # un-normalized output. So round-trip: un-normalize z_pos to Å,
                # project in Å, re-normalize the result. WITHOUT this the shell
                # would push atoms norm_values[0]× too far (~19 Å for d_min≈1.9 at
                # factor 10), scattering every complex into garbage -- an active-
                # but-broken shell that reads as model incapacity, not a no-op.
                if project_enabled and ligand_group is not None:
                    cur_d_min = d_min_schedule(s, timesteps, d_min_start, d_min_end)
                    z_pos = z[:, :self.n_dims]
                    pos_A = z_pos * self.norm_values[0]              # normalized -> Å
                    pos_proj_A = project_exclusion_shell(
                        pos_A, ligand_group, context, cur_d_min)
                    z_pos_proj = pos_proj_A / self.norm_values[0]    # Å -> normalized
                    # Write the projected positions BACK into the working latent so
                    # every subsequent reverse step sees the corrected geometry
                    # (generated atoms only; context is fixed). This mutation is the
                    # whole point -- the shell is a no-op if z is not reassigned.
                    z = torch.cat([
                        z_pos * context + z_pos_proj * ligand_diff,
                        z[:, self.n_dims:]
                    ], dim=1)
                    # One-time proof the shell did real work (vs a silent no-op),
                    # emitted on the first step it actually displaces a generated
                    # atom. If a project_enabled run never prints this, the shell
                    # moved nothing -- the generate-time eligibility guard exists to
                    # prevent exactly that.
                    if not getattr(self, '_projection_logged', False):
                        disp = (pos_proj_A - pos_A).norm(dim=-1)
                        moved = disp > 1e-4
                        if moved.any():
                            self._projection_logged = True
                            print(f"[edm] exclusion-shell projection active: moved "
                                  f"{int(moved.sum().item())}/{int(ligand_diff.sum().item())} "
                                  f"gen atoms (max {float(disp[moved].max().item()):.2f} Å) "
                                  f"at d_min {d_min_start:.2f}->{d_min_end:.2f} Å")
                # Valence-aware type steer (optional, --valence_guard; code-review
                # path item 7). The de-novo failures are dominated by nitrogen in
                # impossible 4-bond environments, so rather than reject the result
                # post-hoc we softly mask the TYPE channel DURING denoising -- making
                # the geometry the model commits to one a valence-legal element can
                # occupy. For each generated atom we count heavy-atom neighbours under
                # the shared bond rule (src.bonding, vectorised) and demote every
                # element whose const.ALLOWED_BONDS valence cannot support that count
                # (a 4-neighbour site -> not N (max 3), pushed toward C (max 4)).
                # The demotion is to just below the atom's OWN logit floor: enough to
                # flip the argmax to a legal element, yet every value stays within the
                # network's own output range (on-manifold), so feeding the steered
                # latent into the next reverse step stays numerically stable -- a soft
                # steer, never a +/-inf hard reject. Generated atoms only; context
                # (incl. the metal) is never touched.
                if valence_guard:
                    h_type = z[:, self.n_dims:]
                    pos_A = z[:, :self.n_dims] * self.norm_values[0]   # normalized -> Å
                    allowed = self._valence_allowed_mask(
                        pos_A, h_type, context, ligand_diff, batch_seg)
                    cur = torch.argmax(h_type, dim=1)
                    corrected = (~allowed.gather(1, cur.view(-1, 1)).squeeze(1)) \
                        & (ligand_diff.view(-1) == 1)
                    floor = h_type.min(dim=1, keepdim=True).values - 1e-2
                    h_type_steer = torch.where(allowed, h_type, floor)
                    z = torch.cat([
                        z[:, :self.n_dims],
                        h_type * context + h_type_steer * ligand_diff,
                    ], dim=1)
                    if corrected.any() and not getattr(self, '_valence_logged', False):
                        self._valence_logged = True
                        print(f"[edm] valence guard active: steered "
                              f"{int(corrected.sum().item())} generated atom(s) off an "
                              f"over-coordinated element (e.g. N>3) at step {s}")
                # Re-noise back to level t (unless last iteration)
                if u < resample_r - 1:
                    gamma_s = self.gamma(s_array)
                    gamma_t = self.gamma(t_array)
                    _, sigma_t_given_s, alpha_t_given_s = \
                        self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
                    eps = self.sample_combined_position_feature_noise(
                        z, ligand_diff)
                    z_renoise = (alpha_t_given_s[batch_seg] * z
                                 + sigma_t_given_s[batch_seg] * eps)
                    z = z * context + z_renoise * ligand_diff

            if (s*keep_frames) % self.T==0:
                write_index = (s * keep_frames) // self.T
                chain[write_index] = self.unnormalize_z(z)
            
        # Finally sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0_only_ligand_diff(
            z_0=z,
            context=context,
            ligand_diff=ligand_diff,
            batch_size=batch_size,
            batch_seg=batch_seg,
            ligand_site=ligand_site,
            valence_guard=valence_guard,
        )
        
        # Correct CoM drift for examples without intermediate states
        if keep_frames==1:
            max_cog = scatter_add(x, batch_seg, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting '
                      f'the positions down.')
                x = utils.remove_partial_mean_with_mask(x, ligand_diff,batch_seg)
                    
        chain[0] = torch.cat([x, h], dim=1)
        
        return chain


    def sample_p_zs_given_zt_only_ligand_diff(self, s, t, z_t, context, ligand_diff, batch_seg, ligand_site):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only linker features and coords"""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
        sigma_s = self.sigma(gamma_s)
        sigma_t = self.sigma(gamma_t)
        # Neural net prediction.
        
        eps_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            batch_seg=batch_seg,
            ligand_site=ligand_site,
            
        )
        eps_hat = eps_hat * ligand_diff

        # Compute mu for p(z_s | z_t)
        mu = z_t / alpha_t_given_s[batch_seg] - (sigma2_t_given_s / alpha_t_given_s / sigma_t)[batch_seg] * eps_hat
        # Compute sigma for p(z_s | z_t)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample z_s given the parameters derived from zt
        z_s = self.sample_normal(mu,ligand_diff,sigma,batch_seg)
        z_s=z_t*context+z_s*ligand_diff
        return z_s

    def sample_p_xh_given_z0_only_ligand_diff(self, z_0, context, ligand_diff, batch_size, batch_seg,ligand_site, valence_guard=False):
        """Samples x ~ p(x|z0). Samples only linker features and coords"""
        zeros = torch.zeros(size=(batch_size, 1), device=z_0.device)
        gamma_0 = self.gamma(zeros)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0)
        eps_hat = self.dynamics.forward(
            xh=z_0,
            t=zeros,
            batch_seg=batch_seg,
            ligand_site=ligand_site,
            
        )
        eps_hat = eps_hat * ligand_diff

        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0,batch_seg=batch_seg)
        xh = self.sample_normal(mu_x,ligand_diff,sigma_x,batch_seg)
        xh=z_0*context+xh*ligand_diff
        x, h = xh[:, :self.n_dims], xh[:, self.n_dims:]
        x, h = self.unnormalize(x, h)
        # Valence-aware read-off (optional, --valence_guard): mask any element whose
        # max heavy-atom valence (const.ALLOWED_BONDS) cannot support a generated
        # atom's neighbour count BEFORE the argmax, so the COMMITTED type is
        # valence-legal (a 4-neighbour site is never read off as N). This is the final
        # discrete step -- nothing is fed back into the network -- so a hard -1e9 mask
        # is safe here (unlike the in-loop steer, which stays on-manifold). x is
        # already in Å (unnormalized above); pairwise distances are translation invariant.
        if valence_guard:
            allowed = self._valence_allowed_mask(x, h, context, ligand_diff, batch_seg)
            h = h.masked_fill(~allowed, -1e9)
        h = F.one_hot(torch.argmax(h, dim=1), self.in_node_nf)

        return x, h

    def _valence_tables(self, device):
        """Vocabulary-indexed valence tables for ``--valence_guard``, built once.

        Returns ``(bond_cutoff_matrix, max_valence)``:
          * ``bond_cutoff_matrix[a, b]`` -- the molSimplify covalent-radii bond cutoff
            (Å) for element pair (a, b), i.e. the vectorised form of
            :func:`src.bonding.are_bonded` (``are_bonded <=> dist < cutoff``) over the
            model's heavy-atom vocabulary ``const.IDX2ATOM``;
          * ``max_valence[a]`` -- the largest heavy-atom valence ``const.ALLOWED_BONDS``
            permits for element ``a`` (N->3, O->2, C->4, S->4, P->5, halogens->1).
        """
        if getattr(self, '_val_cut_mat', None) is None:
            n = self.in_node_nf
            cut_mat = torch.zeros(n, n)
            max_val = torch.zeros(n)
            for a in range(n):
                el_a = const.IDX2ATOM[a]
                permitted = const.ALLOWED_BONDS[el_a]
                max_val[a] = max(permitted) if isinstance(permitted, (list, tuple)) \
                    else permitted
                for b in range(n):
                    cut_mat[a, b] = bond_cutoff(el_a, const.IDX2ATOM[b])
            self._val_cut_mat = cut_mat
            self._val_max = max_val
        return self._val_cut_mat.to(device), self._val_max.to(device)

    def _valence_allowed_mask(self, pos_A, h_type, context, ligand_diff, batch_seg):
        """``(N, in_node_nf)`` bool mask -- element ``t`` is permitted for atom ``i``
        iff ``t``'s max heavy-atom valence can support ``i``'s current heavy-atom
        neighbour count. Drives ``--valence_guard``.

        Neighbours are counted with the shared molSimplify bond rule (vectorised
        :mod:`src.bonding`) using each atom's current argmax element. The metal -- a
        context atom whose (fixed) type row is all-equal (the all-zero one-hot the
        prep prepends) -- is excluded as a neighbour: a metal coordinate bond does not
        consume organic valence, so a 3-bond amine N donating to the metal is NOT
        flagged. Context atoms are never restricted, and any atom with no legal element
        (count exceeds every valence) is left to the model's own choice.
        """
        cut_mat, max_val = self._valence_tables(h_type.device)
        n_atoms = h_type.shape[0]
        elem = torch.argmax(h_type, dim=1)                              # (N,) current element

        # Metal = context atom with an all-equal (all-zero one-hot) type row.
        ctx = context.view(-1) == 1
        flat = (h_type.max(dim=1).values - h_type.min(dim=1).values) < 1e-6
        is_metal = ctx & flat                                           # (N,)

        dist = torch.cdist(pos_A, pos_A)                                # (N, N) Å
        cut = cut_mat[elem][:, elem]                                    # (N, N) pair cutoff
        same = batch_seg.view(-1, 1) == batch_seg.view(1, -1)          # same complex
        eye = torch.eye(n_atoms, dtype=torch.bool, device=h_type.device)
        bonded = (dist < cut) & same & (~eye) & (~is_metal.view(1, -1))
        n_heavy = bonded.sum(dim=1)                                     # (N,) heavy-atom neighbours

        allowed = max_val.view(1, -1) >= n_heavy.view(-1, 1)            # (N, in_node_nf)
        allowed = allowed | (ligand_diff.view(-1, 1) == 0)             # never restrict context
        allowed = allowed | (~allowed.any(dim=1, keepdim=True))        # keep model's choice if none legal
        return allowed

    def compute_x_pred(self, eps_t, z_t, gamma_t,batch_seg):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.sigma(gamma_t)
        alpha_t = self.alpha(gamma_t)
        x_pred = 1. / alpha_t[batch_seg] * (z_t - sigma_t[batch_seg] * eps_t)
        return x_pred

    def kl_prior(self, xh,mask,batch_seg):
        """
        Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss.
        However, you compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T
        batch_size=torch.max(batch_seg)+1
        ones = torch.ones((batch_size, 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T)
        # Compute means
        mu_T = alpha_T[batch_seg].view(-1,1)*xh
        mu_T_x, mu_T_h = mu_T[ :, :self.n_dims], mu_T[:, self.n_dims:]
        # Compute standard deviations (only batch axis for x-part, inflated for h-part)
        sigma_T_x = self.sigma(gamma_T).squeeze(1)
        sigma_T_h = self.sigma(gamma_T).squeeze(1)

        # Compute KL for h-part
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        mu_norm2 = self.inflate_batch_array((mu_T_h - zeros) ** 2*mask, batch_seg)
        kl_distance_h = self.gaussian_kl(mu_norm2, sigma_T_h, ones, d=1)
       
        # Compute KL for x-part
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        mu_norm2 = self.inflate_batch_array((mu_T_x - zeros) ** 2*mask, batch_seg)
        d = self.n_dims*(self.inflate_batch_array(mask,batch_seg)-1)
        kl_distance_x = self.gaussian_kl(mu_norm2, sigma_T_x, ones, d)
        return kl_distance_x + kl_distance_h

    def log_constant_of_p_x_given_z0(self, mask,batch_seg,batch_size):
        degrees_of_freedom_x = self.n_dims*(self.inflate_batch_array(mask,batch_seg)-1)
        zeros = torch.zeros((batch_size, 1), device=mask.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0)
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_p_xh_given_z0_without_constants(self, h, z_0, gamma_0, eps, eps_hat, mask, batch_seg,epsilon=1e-10):
        # Discrete properties are predicted directly from z_0
        z_h = z_0[ :, self.n_dims:]

        # Take only part over x
        eps_x = eps[:, :self.n_dims]
        eps_hat_x = eps_hat[:, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data
        sigma_0 = self.sigma(gamma_0) * self.norm_values[1]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'
        squared_error=(eps_x - eps_hat_x)**2
        log_p_x_given_z_without_constants = -0.5 * self.inflate_batch_array(squared_error, batch_seg)
        

        # Categorical features
        # Compute delta indicator masks
        h = h * self.norm_values[1] + self.norm_biases[1]
        estimated_h = z_h * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded
        centered_h = estimated_h - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=centered_h_cat, stdev=sigma_0_cat)
        log_p_h_proportional = torch.log(
            self.cdf_standard_gaussian((centered_h + 0.5) / sigma_0[batch_seg]) -
            self.cdf_standard_gaussian((centered_h - 0.5) / sigma_0[batch_seg]) +
            epsilon
        )

        # Normalize the distribution over the categories
        log_Z = torch.logsumexp(log_p_h_proportional, dim=1, keepdim=True)
        log_probabilities = log_p_h_proportional - log_Z

        # Select the log_prob of the current category using the onehot representation
        log_p_h_given_z=self.inflate_batch_array(log_probabilities * h * mask,batch_seg)
        # Combine log probabilities for x and h
        #log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_x_given_z_without_constants,log_p_h_given_z

    def sample_combined_position_feature_noise(self, x,ligand_diff):
        z_x = torch.randn(x.shape[0],self.n_dims,device=x.device)*ligand_diff
        z_h = torch.randn(x.shape[0],self.in_node_nf,device=x.device)*ligand_diff
        z = torch.cat([z_x, z_h], dim=1)
        
        return z

    def normalize(self, x, h):
        new_x = x / self.norm_values[0]
        new_h = (h.float() - self.norm_biases[1]) / self.norm_values[1]
        return new_x, new_h

    def unnormalize(self, x, h):
        new_x = x * self.norm_values[0]
        new_h = h * self.norm_values[1] + self.norm_biases[1]
        return new_x, new_h

    def unnormalize_z(self, z):
        assert z.size(1) == self.n_dims + self.in_node_nf
        x, h = z[:, :self.n_dims], z[:, self.n_dims:]
        x, h = self.unnormalize(x, h)
        return torch.cat([x, h], dim=1)

    def delta_log_px(self, mask):
        return -self.dimensionality(mask) * np.log(self.norm_values[0])
        

    def sigma(self, gamma):
        """Computes sigma given gamma."""
        return torch.sqrt(torch.sigmoid(gamma))

    def alpha(self, gamma):
        """Computes alpha given gamma."""
        return torch.sqrt(torch.sigmoid(-gamma))

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


    @staticmethod
    def inflate_batch_array(x,batch_seg):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,),
        or possibly more empty axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        
        return scatter_add(x.sum(-1), batch_seg, dim=0)

    @staticmethod
    def expm1(x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def gaussian_kl(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """Computes the KL distance between two normal distributions.
            Args:
                q_mu_minus_p_mu_squared: Squared difference between mean of
                    distribution q and distribution p: ||mu_q - mu_p||^2
                q_sigma: Standard deviation of distribution q.
                p_sigma: Standard deviation of distribution p.
                d: dimension
            Returns:
                The KL distance
            """
        return d * torch.log(p_sigma / q_sigma) + \
               0.5 * (d * q_sigma ** 2 + q_mu_minus_p_mu_squared) / \
               (p_sigma ** 2) - 0.5 * d
        

    
    

