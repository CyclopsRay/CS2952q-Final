import torch.nn as nn
import functools
import torch
from models.unetmodules.attention import SpatialTransformer
from models.unetmodules import layers, normalization, utils
from abc import abstractmethod
from einops import rearrange

ResnetBlockDDPM = layers.ResnetBlockDDPMpp
ResnetBlockBigGAN = layers.ResnetBlockBigGANpp
# Combine = layers.Combine
conv3x3 = layers.conv3x3
conv1x1 = layers.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

class TimestepBlock(nn.Module):
  """
  Any module where forward() takes timestep embeddings as a second argument.
  """

  @abstractmethod
  def forward(self, x, emb):
    """
    Apply the module to `x` given `emb` timestep embeddings.
    """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, time_embedding, context=None):
        for layer in self:
          try:
            if type(layer) in (layers.ResnetBlockDDPMpp, layers.ResnetBlockBigGANpp):
                x = layer(x, time_embedding)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
          except Exception as e:
            # print('In timestep embed sequential, layer:', layer)
            # print('h shape:', x.shape)
            # print('emb shape:', time_embedding.shape)
            # if context is not None:
            #     print('context shape:', context.shape)
            raise e
        return x

class UNetModel(nn.Module):
  """UNet model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
    self.use_timestep = config.model.use_timestep

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.max_res_num // (2 ** i) for i in range(num_resolutions)]

    self.skip_rescale = skip_rescale = config.model.skip_rescale
    self.resblock_type = resblock_type = config.model.resblock_type.lower()
    init_scale = config.model.init_scale

    self.embedding_type = embedding_type = config.model.embedding_type.lower()
    self.n_heads = n_heads = config.model.n_heads
    self.context_dim = context_dim = config.model.context_dim
    self.mask_embed = config.model.mask_embed

    assert embedding_type in ['fourier', 'positional']

    # These are for the timestep embeddings
    modules = []
    embed_dim = nf
    modules.append(nn.Linear(embed_dim, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    modules.append(nn.Linear(nf * 4, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    self.pre_blocks = nn.ModuleList(modules)

    AttnBlock = functools.partial(layers.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layers.Upsample,
                                 with_conv=resamp_with_conv)

    Downsample = functools.partial(layers.Downsample,
                                   with_conv=resamp_with_conv)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')
    channels = config.data.num_channels
    modules.append(conv3x3(channels, nf))
    self.pre_conv = conv3x3(channels, nf)

    # Downsampling block
    first_block_multiplier = 1
    if self.mask_embed:
        first_block_multiplier = 2
        self.mask_embedding = nn.Embedding(2, nf * ch_mult[0])
        self.combine_mask_conv = conv3x3(nf * 2, nf)

    self.input_blocks = nn.ModuleList([])
    hs_c = [nf]
    in_ch = nf
    self.input_channels = [nf]

    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        cur_layers = [ResnetBlock(in_ch=in_ch, out_ch=out_ch)]

        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          cur_layers.append(AttnBlock(channels=in_ch))
          if config.model.use_spatial_transformer:
            cur_layers.append(SpatialTransformer(in_channels=in_ch,
                                                 n_heads=n_heads,
                                                 d_head=in_ch // n_heads,
                                                 context_dim=context_dim))
        self.input_blocks.append(TimestepEmbedSequential(*cur_layers))
        hs_c.append(in_ch)
        self.input_channels.append(in_ch)

      if i_level != num_resolutions - 1:
        self.input_blocks.append(
          TimestepEmbedSequential(
            Downsample(in_ch=in_ch) if resblock_type == 'ddpm'
            else ResnetBlock(down=True, in_ch=in_ch)
          )
        )
        hs_c.append(in_ch)
        self.input_channels.append(in_ch)

    in_ch = hs_c[-1]
    self.mid_channel = self.input_channels[-1]

    if config.model.use_spatial_transformer:
      self.mid_blocks = TimestepEmbedSequential(
        ResnetBlock(in_ch=self.mid_channel),
        AttnBlock(channels=self.mid_channel),
        SpatialTransformer(in_channels=in_ch,
                           n_heads=n_heads,
                           d_head=in_ch // n_heads,
                           context_dim=context_dim),
        ResnetBlock(in_ch=self.mid_channel)
      )
    else:
      self.mid_blocks = TimestepEmbedSequential(
        ResnetBlock(in_ch=self.mid_channel),
        AttnBlock(channels=self.mid_channel),
        ResnetBlock(in_ch=self.mid_channel)
      )

    # Upsampling block
    self.out_blocks = nn.ModuleList([])
    self.out_channels = []
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        cur_layers = [ResnetBlock(in_ch=in_ch + self.input_channels.pop(), out_ch=out_ch)]
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          cur_layers.append(AttnBlock(channels=in_ch))
          if config.model.use_spatial_transformer:
            cur_layers.append(SpatialTransformer(in_channels=in_ch,
                                                 n_heads=n_heads,
                                                 d_head=in_ch // n_heads,
                                                 context_dim=context_dim))

        if i_level != 0 and i_block == num_res_blocks:
          if resblock_type == 'ddpm':
            cur_layers.append(Upsample(in_ch=in_ch))
          else:
            cur_layers.append(ResnetBlock(in_ch=in_ch, up=True))
        self.out_blocks.append(TimestepEmbedSequential(*cur_layers))

    self.out = nn.ModuleList([])


    self.out.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch, eps=1e-6))
    self.out.append(self.act)
    self.out.append(conv3x3(in_ch, channels, init_scale=init_scale))


  def forward(self, x, timesteps=None, cond=None, context=None):

    if self.use_timestep:

      # Sinusoidal positional embeddings.
      used_sigmas = self.sigmas[timesteps.long()].float()
      temb = layers.get_timestep_embedding(timesteps, self.nf)

      # pre blocks
      for module in self.pre_blocks:
        temb = module(temb)
    else:
        temb = None

    x = x.float()
    h = self.pre_conv(x)

    if self.mask_embed and cond is not None:
      cond_emb = self.mask_embedding(cond.long())
      cond_emb = rearrange(cond_emb, 'b h w c -> b c h w')
      h = torch.concat([h, cond_emb], dim=1)
      h = self.combine_mask_conv(h)

    hs = [h]

    # Down Sampling blocks
    for i, module in enumerate(self.input_blocks):
      try:
        h = module(h, temb, context=context)
        # print('in down block', i, 'h shape:', h.shape)
        hs.append(h)
      except Exception as e:
        print(f'in down block {i}, module is:', module)
        print('h shape:', h.shape)
        print('temb shape:', temb.shape)
        # print('text_emb shape:', context.shape)
        raise e

    # Mid Block
    h = self.mid_blocks(h, temb, context=context)
    # Up Sampling blocks
    for module in self.out_blocks:
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, temb, context=context)


    assert not hs

    for out in self.out:
      h = out(h)
    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas
    return h
