from typing import Optional

import torch
from torch import FloatTensor, LongTensor, Size, Tensor, nn as nn

from prior import generate_beta_tensor


# class IPAttnProcessor(nn.Module):
#     r"""
#     Attention processor for IP-Adapater.
#     Args:
#         hidden_size (`int`):
#             The hidden size of the attention layer.
#         cross_attention_dim (`int`):
#             The number of channels in the `encoder_hidden_states`.
#         scale (`float`, defaults to 1.0):
#             the weight scale of image prompt.
#         num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
#             The context length of the image features.
#     """

#     def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.cross_attention_dim = cross_attention_dim
#         self.scale = scale
#         self.num_tokens = num_tokens

#         self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
#         self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

#     def __call__(
#         self,
#         attn,
#         hidden_states,
#         encoder_hidden_states=None,
#         attention_mask=None,
#         temb=None,
#         *args,
#         **kwargs,
#     ):
#         residual = hidden_states

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#         batch_size, sequence_length, _ = (
#             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#         )
#         attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         else:
#             # get encoder_hidden_states, ip_hidden_states
#             end_pos = encoder_hidden_states.shape[1] - self.num_tokens
#             encoder_hidden_states, ip_hidden_states = (
#                 encoder_hidden_states[:, :end_pos, :],
#                 encoder_hidden_states[:, end_pos:, :],
#             )
#             if attn.norm_cross:
#                 encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         key = attn.to_k(encoder_hidden_states)
#         value = attn.to_v(encoder_hidden_states)

#         query = attn.head_to_batch_dim(query)
#         key = attn.head_to_batch_dim(key)
#         value = attn.head_to_batch_dim(value)

#         attention_probs = attn.get_attention_scores(query, key, attention_mask)
#         hidden_states = torch.bmm(attention_probs, value)
#         hidden_states = attn.batch_to_head_dim(hidden_states)

#         # for ip-adapter
#         ip_key = self.to_k_ip(ip_hidden_states)
#         ip_value = self.to_v_ip(ip_hidden_states)

#         ip_key = attn.head_to_batch_dim(ip_key)
#         ip_value = attn.head_to_batch_dim(ip_value)

#         ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
#         self.attn_map = ip_attention_probs
#         ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
#         ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

#         hidden_states = hidden_states + self.scale * ip_hidden_states

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor

#         return hidden_states


class OuterInterpolatedIPAttnProcessor(nn.Module):
    r"""
    Personalized processor for performing outer attention interpolation.

    The attention output of interpolated image is obtained by:
    (1 - t) * Q_t * K_1 * V_1 + t * Q_t * K_m * V_m;
    If fused with self-attention:
    (1 - t) * Q_t * [K_1, K_t] * [V_1, V_t] + t * Q_t * [K_m, K_t] * [V_m, V_t];
    """

    def __init__(
        self,
        t: Optional[float] = None,
        size: int = 7,
        is_fused: bool = False,
        alpha: float = 1,
        beta: float = 1,
        ip_attn: Optional[nn.Module] = None
    ):
        """
        t: float, interpolation point between 0 and 1, if specified, size is set to 3
        """
        super().__init__()
        if t is None:
            ts = generate_beta_tensor(size, alpha=alpha, beta=beta)
            ts[0], ts[-1] = 0, 1
        else:
            assert t > 0 and t < 1, "t must be between 0 and 1"
            ts = [0, t, 1]
            ts = torch.tensor(ts)
            size = 3

        self.size = size
        self.coef = ts
        self.is_fused = is_fused

        self.num_tokens = ip_attn.num_tokens if hasattr(ip_attn, "num_tokens") else (16,)
        self.scale = ip_attn.scale if hasattr(ip_attn, "scale") else None
        self.ip_attn = ip_attn
        self.use_origin = False

    def set_origin(self):
        self.use_origin = True

    def set_interpolation(self):
        self.use_origin = False

    def set_t(self, t):
        assert t > 0 and t < 1, "t must be between 0 and 1"
        ts = [0, t, 1]
        ts = torch.tensor(ts)
        self.coef = ts

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if self.use_origin:
            return self.ip_attn(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        
        residual = hidden_states

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            ip_hidden_states = None
        else:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
                # ip_hidden_states = ip_hidden_states[0]
            else:
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Specify the first and last key and value
        key_begin = key[0:1]
        key_end = key[-1:]
        value_begin = value[0:1]
        value_end = value[-1:]

        key_begin = torch.cat([key_begin] * (self.size))
        key_end = torch.cat([key_end] * (self.size))
        value_begin = torch.cat([value_begin] * (self.size))
        value_end = torch.cat([value_end] * (self.size))

        key_begin = attn.head_to_batch_dim(key_begin)
        value_begin = attn.head_to_batch_dim(value_begin)
        key_end = attn.head_to_batch_dim(key_end)
        value_end = attn.head_to_batch_dim(value_end)

        # Fused with self-attention
        if self.is_fused:
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key_end = torch.cat([key, key_end], dim=-2)
            value_end = torch.cat([value, value_end], dim=-2)
            key_begin = torch.cat([key, key_begin], dim=-2)
            value_begin = torch.cat([value, value_begin], dim=-2)

        attention_probs_end = attn.get_attention_scores(query, key_end, attention_mask)
        hidden_states_end = torch.bmm(attention_probs_end, value_end)
        hidden_states_end = attn.batch_to_head_dim(hidden_states_end)

        attention_probs_begin = attn.get_attention_scores(
            query, key_begin, attention_mask
        )
        hidden_states_begin = torch.bmm(attention_probs_begin, value_begin)
        hidden_states_begin = attn.batch_to_head_dim(hidden_states_begin)

        # for ip-adapter
        if ip_hidden_states is not None:
            key = self.ip_attn.to_k_ip[0](ip_hidden_states[0])
            value = self.ip_attn.to_v_ip[0](ip_hidden_states[0])

            # Specify the first and last key and value
            key_begin = key[0:1]
            key_end = key[-1:]
            value_begin = value[0:1]
            value_end = value[-1:]

            key_begin = torch.cat([key_begin] * (self.size))
            key_end = torch.cat([key_end] * (self.size))
            value_begin = torch.cat([value_begin] * (self.size))
            value_end = torch.cat([value_end] * (self.size))

            key_begin = attn.head_to_batch_dim(key_begin)
            value_begin = attn.head_to_batch_dim(value_begin)
            key_end = attn.head_to_batch_dim(key_end)
            value_end = attn.head_to_batch_dim(value_end)

            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            # Fused with self-attention
            if self.is_fused:
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)
                key_end = torch.cat([key, key_end], dim=-2)
                value_end = torch.cat([value, value_end], dim=-2)
                key_begin = torch.cat([key, key_begin], dim=-2)
                value_begin = torch.cat([value, value_begin], dim=-2)

            ip_attention_probs_end = attn.get_attention_scores(query, key_end, attention_mask)
            ip_hidden_states_end = torch.bmm(ip_attention_probs_end, value_end)
            ip_hidden_states_end = attn.batch_to_head_dim(ip_hidden_states_end)

            ip_attention_probs_begin = attn.get_attention_scores(
                query, key_begin, attention_mask
            )
            ip_hidden_states_begin = torch.bmm(ip_attention_probs_begin, value_begin)
            ip_hidden_states_begin = attn.batch_to_head_dim(ip_hidden_states_begin)

            hidden_states_begin = hidden_states_begin + self.scale[0] * ip_hidden_states_begin
            hidden_states_end = hidden_states_end + self.scale[0] * ip_hidden_states_end

        # Apply outer interpolation on attention
        coef = self.coef.reshape(-1, 1, 1)
        coef = coef.to(key.device, key.dtype)
        hidden_states = (1 - coef) * hidden_states_begin + coef * hidden_states_end

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class OuterInterpolatedAttnProcessor:
    r"""
    Personalized processor for performing outer attention interpolation.

    The attention output of interpolated image is obtained by:
    (1 - t) * Q_t * K_1 * V_1 + t * Q_t * K_m * V_m;
    If fused with self-attention:
    (1 - t) * Q_t * [K_1, K_t] * [V_1, V_t] + t * Q_t * [K_m, K_t] * [V_m, V_t];
    """

    def __init__(
        self,
        t: Optional[float] = None,
        size: int = 7,
        is_fused: bool = False,
        alpha: float = 1,
        beta: float = 1,
    ):
        """
        t: float, interpolation point between 0 and 1, if specified, size is set to 3
        """
        if t is None:
            ts = generate_beta_tensor(size, alpha=alpha, beta=beta)
            ts[0], ts[-1] = 0, 1
        else:
            assert t > 0 and t < 1, "t must be between 0 and 1"
            ts = [0, t, 1]
            ts = torch.tensor(ts)
            size = 3

        self.size = size
        self.coef = ts
        self.is_fused = is_fused

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Specify the first and last key and value
        key_begin = key[0:1]
        key_end = key[-1:]
        value_begin = value[0:1]
        value_end = value[-1:]

        key_begin = torch.cat([key_begin] * (self.size))
        key_end = torch.cat([key_end] * (self.size))
        value_begin = torch.cat([value_begin] * (self.size))
        value_end = torch.cat([value_end] * (self.size))

        key_begin = attn.head_to_batch_dim(key_begin)
        value_begin = attn.head_to_batch_dim(value_begin)
        key_end = attn.head_to_batch_dim(key_end)
        value_end = attn.head_to_batch_dim(value_end)

        # Fused with self-attention
        if self.is_fused:
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key_end = torch.cat([key, key_end], dim=-2)
            value_end = torch.cat([value, value_end], dim=-2)
            key_begin = torch.cat([key, key_begin], dim=-2)
            value_begin = torch.cat([value, value_begin], dim=-2)

        attention_probs_end = attn.get_attention_scores(query, key_end, attention_mask)
        hidden_states_end = torch.bmm(attention_probs_end, value_end)
        hidden_states_end = attn.batch_to_head_dim(hidden_states_end)

        attention_probs_begin = attn.get_attention_scores(
            query, key_begin, attention_mask
        )
        hidden_states_begin = torch.bmm(attention_probs_begin, value_begin)
        hidden_states_begin = attn.batch_to_head_dim(hidden_states_begin)

        # Apply outer interpolation on attention
        coef = self.coef.reshape(-1, 1, 1)
        coef = coef.to(key.device, key.dtype)
        hidden_states = (1 - coef) * hidden_states_begin + coef * hidden_states_end

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class InnerInterpolatedAttnProcessor:
    r"""
    Personalized processor for performing inner attention interpolation.

    The attention output of interpolated image is obtained by:
    (1 - t) * Q_t * K_1 * V_1 + t * Q_t * K_m * V_m;
    If fused with self-attention:
    (1 - t) * Q_t * [K_1, K_t] * [V_1, V_t] + t * Q_t * [K_m, K_t] * [V_m, V_t];
    """

    def __init__(
        self,
        t: Optional[float] = None,
        size: int = 7,
        is_fused: bool = False,
        alpha: float = 1,
        beta: float = 1,
    ):
        """
        t: float, interpolation point between 0 and 1, if specified, size is set to 3
        """
        if t is None:
            ts = generate_beta_tensor(size, alpha=alpha, beta=beta)
            ts[0], ts[-1] = 0, 1
        else:
            assert t > 0 and t < 1, "t must be between 0 and 1"
            ts = [0, t, 1]
            ts = torch.tensor(ts)
            size = 3

        self.size = size
        self.coef = ts
        self.is_fused = is_fused

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Specify the first and last key and value
        key_start = key[0:1]
        key_end = key[-1:]
        value_start = value[0:1]
        value_end = value[-1:]

        key_start = torch.cat([key_start] * (self.size))
        key_end = torch.cat([key_end] * (self.size))
        value_start = torch.cat([value_start] * (self.size))
        value_end = torch.cat([value_end] * (self.size))

        # Apply inner interpolation on attention
        coef = self.coef.reshape(-1, 1, 1)
        coef = coef.to(key.device, key.dtype)
        key_cross = (1 - coef) * key_start + coef * key_end
        value_cross = (1 - coef) * value_start + coef * value_end

        key_cross = attn.head_to_batch_dim(key_cross)
        value_cross = attn.head_to_batch_dim(value_cross)

        # Fused with self-attention
        if self.is_fused:
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key_cross = torch.cat([key, key_cross], dim=-2)
            value_cross = torch.cat([value, value_cross], dim=-2)

        attention_probs = attn.get_attention_scores(query, key_cross, attention_mask)

        hidden_states = torch.bmm(attention_probs, value_cross)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def linear_interpolation(
    l1: FloatTensor, l2: FloatTensor, ts: Optional[FloatTensor] = None, size: int = 5
) -> FloatTensor:
    """
    Linear interpolation

    Args:
        l1: Starting vector: (1, *)
        l2: Final vector: (1, *)
        ts: FloatTensor, interpolation points between 0 and 1
        size: int, number of interpolation points including l1 and l2

    Returns:
    Interpolated vectors: (size, *)
    """
    assert l1.shape == l2.shape, "shapes of l1 and l2 must match"

    res = []
    if ts is not None:
        for t in ts:
            li = torch.lerp(l1, l2, t)
            res.append(li)
    else:
        for i in range(size):
            t = i / (size - 1)
            li = torch.lerp(l1, l2, t)
            res.append(li)
    res = torch.cat(res, dim=0)
    return res


def spherical_interpolation(l1: FloatTensor, l2: FloatTensor, size=5) -> FloatTensor:
    """
    Spherical interpolation

    Args:
        l1: Starting vector: (1, *)
        l2: Final vector: (1, *)
        size: int, number of interpolation points including l1 and l2

    Returns:
        Interpolated vectors: (size, *)
    """
    assert l1.shape == l2.shape, "shapes of l1 and l2 must match"

    res = []
    for i in range(size):
        t = i / (size - 1)
        li = slerp(l1, l2, t)
        res.append(li)
    res = torch.cat(res, dim=0)
    return res


def slerp(v0: FloatTensor, v1: FloatTensor, t, threshold=0.9995):
    """
    Spherical linear interpolation
    Args:
        v0: Starting vector
        v1: Final vector
        t: Float value between 0.0 and 1.0
        threshold: Threshold for considering the two vectors as
                                colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1
    """
    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    # Normalize the vectors to get the directions and angles
    v0_norm: FloatTensor = torch.norm(v0, dim=-1)
    v1_norm: FloatTensor = torch.norm(v1, dim=-1)

    v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
    v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
    dot_mag: FloatTensor = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use torch.lerp
    gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > threshold)
    can_slerp: LongTensor = ~gotta_lerp

    t_batch_dim_count: int = max(0, t.dim() - v0.dim()) if isinstance(t, Tensor) else 0
    t_batch_dims: Size = (
        t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
    )
    out: FloatTensor = torch.zeros_like(v0.expand(*t_batch_dims, *[-1] * v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        lerped: FloatTensor = torch.lerp(v0, v1, t)

        out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():

        # Calculate initial angle between v0 and v1
        theta_0: FloatTensor = dot.arccos().unsqueeze(-1)
        sin_theta_0: FloatTensor = theta_0.sin()
        # Angle at timestep t
        theta_t: FloatTensor = theta_0 * t
        sin_theta_t: FloatTensor = theta_t.sin()
        # Finish the slerp algorithm
        s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
        s1: FloatTensor = sin_theta_t / sin_theta_0
        slerped: FloatTensor = s0 * v0 + s1 * v1

        out: FloatTensor = slerped.where(can_slerp.unsqueeze(-1), out)

    return out
