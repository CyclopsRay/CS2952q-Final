import torch
from tqdm import tqdm


def extract_to_tensor(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionSampler(torch.nn.Module):

    def __init__(self,
                 model,
                 timesteps=1000,
                 betas=None,
                 beta_schedule='linear',
                 loss_type='l2',
                 linear_start=0.00085,
                 linear_end=0.0120,
                 cosine_s=8e-3,
                 sampling_steps=1000,
                 ddim_eta=1.,
                 w=0.,
                 v_posterior=0.,
                 parameterization='eps',
                 force_background=False
                 ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.sampling_steps = sampling_steps
        self.ddim_eta = ddim_eta
        self.w = w
        self.v_posterior = v_posterior
        self.parameterization = parameterization
        self.force_background = force_background

        self.register_schedule(betas=betas,
                               beta_schedule=beta_schedule,
                               timesteps=timesteps,
                               linear_start=linear_start,
                               linear_end=linear_end,
                               cosine_s=cosine_s)

        # ----------End of __init__--------------

    def register_schedule(self, betas, beta_schedule, timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):

        if betas is None:
            # betas increasing
            betas = torch.linspace(linear_start, linear_end, timesteps)

        # alphas decreasing
        alphas = 1. - betas

        # cumulative product of alphas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones_like(alphas[:1]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculating q(x_t | x_{t-1}), add noise to x_{t-1}
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.rsqrt(alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1.))

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        posterior_variance =(1. - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + \
                            self.v_posterior * betas
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * alphas * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * torch.sqrt(alphas_cumprod) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()


    @torch.no_grad()
    def denoise_sample_from_pure_noise(self, batch_size=16, cond=None, return_intermediates=False):
        # use ddim sampling
        pass

    @torch.no_grad()
    def ddim_sample(self, shape, context=None, output=None, clip_scheme='static', desc=None, use_tqdm=True):
        '''

        :param shape: input shape
        :param cond: conditioning, here our textual description embedding
        :param clip_scheme: 'static' or 'dynamic'
        :return: a less noisy sample
        '''
        if desc is None:
            desc = f'DDIM sampling loop time step'
        batch_size = shape[0]
        device = self.betas.device
        total_timesteps = self.timesteps
        ddim_steps = self.sampling_steps
        eta = self.ddim_eta

        times = torch.linspace(-1, total_timesteps - 1, ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if output is None:
            output = torch.randn(shape, device=device)

        if use_tqdm:
            sampling_progress_bar = tqdm(time_pairs, desc=desc)
        else:
            sampling_progress_bar = time_pairs

        for i, (t, t_next) in enumerate(sampling_progress_bar):
            time_cond = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(output, time_cond, context=context, clip_scheme=clip_scheme)

            if t_next < 0:
                output = x_start
                continue

            alpha_bar = self.alphas_cumprod[t]
            alpha_next_bar = self.alphas_cumprod[t_next]

            sigma = torch.sqrt(eta * ((1 - alpha_bar / alpha_next_bar) * (1 - alpha_next_bar) / (1 - alpha_bar)))
            c = torch.sqrt(1 - alpha_next_bar - sigma ** 2)

            noise = torch.randn_like(output)

            output = x_start * torch.sqrt(alpha_next_bar) + \
                     c * pred_noise + \
                     sigma * noise

            if not use_tqdm:
                print(f'\r{desc}, {i + 1}/{ddim_steps}', end='')

        return output


    @torch.no_grad()
    def ddim_sample_condition(self, shape, cond, context=None, output=None, clip_scheme='static', desc=None):
        '''

        :param shape: input shape
        :param cond: conditioning, here our mask for the image
        :param clip_scheme: 'static' or 'dynamic'
        :return: a less noisy sample
        '''
        if desc is None:
            desc = 'DDIM sampling loop time step'
        batch_size = shape[0]
        device = self.betas.device
        total_timesteps = self.timesteps
        ddim_steps = self.sampling_steps
        eta = self.ddim_eta

        times = torch.linspace(-1, total_timesteps - 1, ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if output is None:
            output = torch.randn(shape, device=device)

        for t, t_next in tqdm(time_pairs, desc=desc):
            time_cond = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(output, time_cond, cond=cond, context=context, clip_scheme=clip_scheme)

            if t_next < 0:
                output = x_start
                continue

            alpha_bar = self.alphas_cumprod[t]
            alpha_next_bar = self.alphas_cumprod[t_next]

            sigma = torch.sqrt(eta * ((1 - alpha_bar / alpha_next_bar) * (1 - alpha_next_bar) / (1 - alpha_bar)))
            c = torch.sqrt(1 - alpha_next_bar - sigma ** 2)

            noise = torch.randn_like(output)

            output = x_start * torch.sqrt(alpha_next_bar) + \
                     c * pred_noise + \
                     sigma * noise

        return output

    @torch.no_grad()
    def repaint_ddim_sample(self, shape, gt, mask, context=None, n_resample=10, clip_scheme='static'):
        '''
        Notice: be careful about img normalization
        :param shape: input shape
        :param gt: normalized ground truth image
        :param mask: 1 for unknown, 0 for unknown
        :param clip_scheme: 'static' or 'dynamic'
        :return: a less noisy sample
        '''

        batch_size = shape[0]
        device = self.betas.device
        total_timesteps = self.timesteps
        ddim_steps = self.sampling_steps
        eta = self.ddim_eta

        times = torch.linspace(-1, total_timesteps - 1, ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        output = torch.randn(shape, device=device)
        x_start = None

        for t, t_next in tqdm(time_pairs, desc='DDIM repaint sampling loop time step'):
            time_cond = torch.full((batch_size,), t, device=device, dtype=torch.long)

            alpha_bar = self.alphas_cumprod[t]
            alpha_next_bar = self.alphas_cumprod[t_next]

            gt_weight = torch.sqrt(alpha_bar)
            gt_part = gt_weight * gt

            noise_weight = torch.sqrt(1 - alpha_bar)
            noise_part = noise_weight * torch.randn_like(output)

            weighted_gt = gt_part + noise_part

            output = mask * weighted_gt + (1 - mask) * output

            pred_noise, x_start = self.model_predictions(output, time_cond, context=context, clip_scheme=clip_scheme)

            if t_next < 0:
                output = x_start
                continue



            sigma = torch.sqrt(eta * ((1 - alpha_bar / alpha_next_bar) * (1 - alpha_next_bar) / (1 - alpha_bar)))
            c = torch.sqrt(1 - alpha_next_bar - sigma ** 2)

            noise = torch.randn_like(output)

            output = x_start * torch.sqrt(alpha_next_bar)
            # + \
                     # c * pred_noise + \
                     # sigma * noise

        return output

    @torch.no_grad()
    def repaint_ddim_sample_condition(self, shape, gt, mask, context=None, n_resample=10, clip_scheme='static'):
        '''
        Notice: be careful about img normalization
        :param shape: input shape
        :param gt: normalized ground truth image
        :param mask: 1 for unknown, 0 for unknown
        :param clip_scheme: 'static' or 'dynamic'
        :return: a less noisy sample
        '''

        batch_size = shape[0]
        device = self.betas.device
        total_timesteps = self.timesteps
        ddim_steps = self.sampling_steps
        eta = self.ddim_eta

        times = torch.linspace(-1, total_timesteps - 1, ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        output = torch.randn(shape, device=device)
        x_start = None

        for t, t_next in tqdm(time_pairs, desc='DDIM repaint sampling loop time step'):
            time_cond = torch.full((batch_size,), t, device=device, dtype=torch.long)

            alpha_bar = self.alphas_cumprod[t]
            alpha_next_bar = self.alphas_cumprod[t_next]

            gt_weight = torch.sqrt(alpha_bar)
            gt_part = gt_weight * gt

            noise_weight = torch.sqrt(1 - alpha_bar)
            noise_part = noise_weight * torch.randn_like(output)

            weighted_gt = gt_part + noise_part

            output = mask * weighted_gt + (1 - mask) * output

            mask = mask.unsqueeze(0)[:, 0, ...].to(device)

            pred_noise, x_start = self.model_predictions(output, time_cond, cond=mask, context=context, clip_scheme=clip_scheme)

            if t_next < 0:
                output = x_start
                continue



            sigma = torch.sqrt(eta * ((1 - alpha_bar / alpha_next_bar) * (1 - alpha_next_bar) / (1 - alpha_bar)))
            c = torch.sqrt(1 - alpha_next_bar - sigma ** 2)

            noise = torch.randn_like(output)

            output = x_start * torch.sqrt(alpha_next_bar)
            # + \
            #          c * pred_noise + \
            #          sigma * noise

        return output

    @torch.no_grad()
    def ddim_sample_t_denoise(self, clear_img, t, context=None, output=None, clip_scheme='static', desc=None, use_tqdm=True):
        '''

        :param shape: input shape
        :param cond: conditioning, here our textual description embedding
        :param clip_scheme: 'static' or 'dynamic'
        :param t
        :return: a less noisy sample
        '''

        shape = clear_img.shape

        batch_size = shape[0]
        device = self.betas.device
        total_timesteps = self.timesteps
        ddim_steps = self.sampling_steps
        eta = self.ddim_eta

        t = ddim_steps - t

        times = torch.linspace(-1, total_timesteps - 1, ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        t_level = times[t]
        t_level_tensor = torch.full((batch_size,), t_level, device=device, dtype=torch.long)

        noisy_img = self.q_sample(clear_img, t_level_tensor).to(device)

        time_pairs = time_pairs[t:]

        output = noisy_img

        if desc is None:
            desc = f'DDIM sampling loop time step'


        if output is None:
            output = torch.randn(shape, device=device)

        if use_tqdm:
            sampling_progress_bar = tqdm(time_pairs, desc=desc)
        else:
            sampling_progress_bar = time_pairs

        for i, (t, t_next) in enumerate(sampling_progress_bar):
            time_cond = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(output, time_cond, context=context, clip_scheme=clip_scheme)

            if t_next < 0:
                output = x_start
                continue

            alpha_bar = self.alphas_cumprod[t]
            alpha_next_bar = self.alphas_cumprod[t_next]

            sigma = torch.sqrt(eta * ((1 - alpha_bar / alpha_next_bar) * (1 - alpha_next_bar) / (1 - alpha_bar)))
            c = torch.sqrt(1 - alpha_next_bar - sigma ** 2)

            noise = torch.randn_like(output)

            output = x_start * torch.sqrt(alpha_next_bar) + \
                     c * pred_noise + \
                     sigma * noise

            if not use_tqdm:
                print(f'\r{desc}, {i + 1}/{ddim_steps}', end='')

        return output


    def repaint_p_sample_loop(self, gt_img, mask):
        pass
    def repaint_p_sample(self, xt, t, mask):
        '''

        :param xt: current t_level noisy img
        :param t: timestep t
        :param clip_scheme: 'static' or 'dynamic'
        :return: a less noisy sample
        '''
        x_known = None
        noise = torch.randn_like(xt)
        pass

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        B = shape[0]
        output = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(self.timesteps)), desc='DDPM sampling loop time step'):
            # print(f'sampling time step [{t+1}/{self.timesteps}]')
            time_cond = torch.full((B,), t, device=device, dtype=torch.long)
            try:
                output = self.p_sample(output, time_cond)
            except:
                print('sampling failed, OOM')
                break

        return output

    @torch.no_grad()
    def p_sample(self, x, t, context=None):
        B = x.shape[0]
        device = x.device
        model_mean, _, log_variance = self.p_mean_variance(x, t, context=context)
        noise = torch.randn_like(model_mean).to(device)
        non_zero_mask = (1 - (t == 0).float()).reshape(B, *((1,) * (len(x.shape) - 1)))
        return model_mean + non_zero_mask * torch.exp(0.5 * log_variance) * noise
    def p_mean_variance(self, x, t, context=None, clip_scheme='static'):
        '''

        :param x: input image
        :param t: timestep t
        :param cond: conditioning, here our textual description embedding
        :param clip_scheme: 'static' or 'dynamic'
        :return: mean and variance of the distribution
        '''

        B, C, _, _ = x.shape

        pred_noise, x_recon = self.model_predictions(x, t, context=context, clip_scheme=clip_scheme)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    def model_predictions(self, x, time_cond, cond=None, context=None, clip_scheme='static'):
        '''
        :param x: input image
        :param cond: conditioning, here our textual description embedding
        :param time_cond: time step
        :param clip_scheme: 'static' or 'dynamic'
        :return: prediction noise, x_start
        '''

        # if cond is None:
        model_output = self.model(x, time_cond, cond=cond, context=context)
        # else:
        #     model_output = self.w * self.model(x, time_cond, context) + (1 - self.w) * self.model(x, cond * 0, time_cond)

        if self.parameterization == 'eps':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, time_cond, pred_noise)
        elif self.parameterization == 'x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, time_cond, x_start)
        if clip_scheme == 'static':
            x_start = torch.clamp(x_start, -1., 1.)
        elif clip_scheme == 'dynamic':
            # TODO: see Imagen implementation
            pass

        return pred_noise, x_start

    def predict_start_from_noise(self, x_t, t, pred_noise):
        return extract_to_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
               extract_to_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract_to_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                 extract_to_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract_to_tensor(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start + \
               extract_to_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_loss(self, x_start, t, context=None, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = self.model(x_noisy, t, context=context)

        if self.loss_type == 'l1':
            loss = torch.abs(pred_noise - noise).mean()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
        else:
            raise NotImplementedError(f'Loss type "{self.loss_type}" not implemented')
        return loss

    def p_loss_masked(self, x_start, t, context=None, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # Optional: Forcefully set the background part of x_noisy to be equal to x_start
        if self.force_background:
            x_noisy[mask == 1] = x_start[mask == 1]

        # mask are the same for all channels, so we can just take the first one
        mask = mask[:, 0, ...]

        model_output = self.model(x_noisy, t, cond=mask, context=context)

        mask = torch.stack([mask, mask, mask], dim=1)

        if self.parameterization == 'eps':
            target = noise

            pred_xt_minus1 = self.predict_start_from_noise(x_start, t, model_output)


        elif self.parameterization == 'x0':
            target = x_start
            # pred_x0 = model_output

        '''
        Loss terms:
            1. mse between pred noise and real noise
            2. mse between background part of pred x0 and real x0
        '''
        if self.loss_type == 'l1':
            loss = torch.abs(model_output - target).mean()
            # loss_bg = torch.abs(pred_x0[mask == 0] - x_start[mask == 0]).mean()
            # loss = loss + loss_bg
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(model_output, target)
            t_minus1 = torch.clamp_min(t - 1, 0)
            gt_xt_minus1 = self.q_sample(x_start, t_minus1, noise)

            # shape [B, C, H, W]
            loss_bg = torch.nn.functional.mse_loss(pred_xt_minus1 * (1 - mask), gt_xt_minus1 * (1 - mask), reduction='none')
            # shape [B]
            loss_bg = torch.mean(loss_bg, dim=[1, 2, 3])
            loss_bg = torch.mean(self.sqrt_alpha_cumprod[t_minus1] * loss_bg)

            # when loss_bg weight is too high, the model can hardly converge. set it to a small value
            loss = 0.9 * loss + 0.1 * loss_bg
        else:
            raise NotImplementedError(f'Loss type "{self.loss_type}" not implemented')
        return loss

    def forward(self, x, cond=None, context=None):
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        return self.p_loss(x, t, cond=cond, context=context)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract_to_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start - \
                         extract_to_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract_to_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_to_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
