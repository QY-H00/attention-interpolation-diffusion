from typing import Optional

import torch
from torch import FloatTensor, LongTensor, Size, Tensor

from prior import generate_beta_tensor


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
