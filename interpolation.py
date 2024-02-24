from typing import Optional
import torch
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from scipy.stats import beta as beta_distribution

USE_PEFT_BACKEND = False


class InterpolationAttnProcessorWithUncond:
    """
    Personalized processor for performing attention-related interpolation.
    """

    def __init__(self, size=10, is_fused=False, alpha=1, beta=1, torch_device="cuda", is_interpolate_uncond=False):
        '''
        Args:
        size: int, number of interpolation points including l1 and l2
        is_fused: bool, whether to use the key and value of the token itself
        alpha: float, alpha parameter of the Beta distribution
        beta: float, beta parameter of the Beta distribution
        torch_device: str, device to use for the tensor
        '''
        ts = generate_beta_tensor(size, alpha=alpha, beta=beta)
        ts[0] = 0
        ts[-1] = 1
        self.size = size
        self.coef = ts.to(torch_device)
        self.is_fused = is_fused
        self.is_interpolate_uncond = is_interpolate_uncond

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        '''
        Notice that this attention processor is specifically design for generation with guidance.
        The processor view the hidden states as [l1, ..., l_size, l1_uncond, ..., l_size_uncond] to interpolate
        The processor view l1 as starting vector and l_size as final vector
        For each token, it utilizes query generated itself, but key and value of l1 and l_size
        For unconditional tokens, it utilizes query generated itself, but key and value of l1_uncond and l_size_uncond
        It then interpolates the full attention matrix.
        
        Args:
        attn: Attention module
        hidden_states: Hidden states of the input tokens
        encoder_hidden_states: Hidden states of the encoder tokens
        attention_mask: Attention mask
        temb: Token embeddings
        scale: Scaling factor for the attention scores
        
        Returns:
        Interpolated hidden states
        '''
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        key_a = key[0:1]
        key_b = key[self.size-1:self.size]
        value_a = value[0:1]
        value_b = value[self.size-1:self.size]
        
        uncond_key_a = key[self.size:self.size+1]
        uncond_key_b = key[-1:]
        uncond_value_a = value[self.size:self.size+1]
        uncond_value_b = value[-1:]
        
        if self.is_interpolate_uncond:
            key_a = torch.cat([key_a]*(self.size) + [uncond_key_a]*(self.size))
            key_b = torch.cat([key_b]*(self.size) + [uncond_key_b]*(self.size))
            value_a = torch.cat([value_a]*(self.size) + [uncond_value_a]*(self.size))
            value_b = torch.cat([value_b]*(self.size) + [uncond_value_b]*(self.size))
        else:
            key_a = torch.cat([key_a]*self.size + [key[self.size:]])
            key_b = torch.cat([key_b]*self.size + [key[self.size:]])
            value_a = torch.cat([value_a]*self.size + [value[self.size:]])
            value_b = torch.cat([value_b]*self.size + [value[self.size:]])

        key_a = attn.head_to_batch_dim(key_a)
        value_a = attn.head_to_batch_dim(value_a)
        key_b = attn.head_to_batch_dim(key_b)
        value_b = attn.head_to_batch_dim(value_b)

        if self.is_fused:
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key_b = torch.cat([key, key_b], dim=-2)
            value_b = torch.cat([value, value_b], dim=-2)
            key_a = torch.cat([key, key_a], dim=-2)
            value_a = torch.cat([value, value_a], dim=-2)
        
        attention_probs_b = attn.get_attention_scores(query, key_b, attention_mask)
        hidden_states_b = torch.bmm(attention_probs_b, value_b)
        hidden_states_b = attn.batch_to_head_dim(hidden_states_b)

        attention_probs_a = attn.get_attention_scores(query, key_a, attention_mask)
        hidden_states_a = torch.bmm(attention_probs_a, value_a)
        hidden_states_a = attn.batch_to_head_dim(hidden_states_a)

        coef = self.coef.reshape(-1, 1, 1)
        coef = torch.cat([coef]*2, dim=0)

        hidden_states = (1 - coef) * hidden_states_a + coef * hidden_states_b
        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class InterpolationAttnProcessor:
    r"""
    Personalized processor for performing attention-related interpolation.
    """

    def __init__(self, t=None, size=10, is_fused=False, alpha=1, beta=1, torch_device="cuda"):
        if t is None:
            ts = generate_beta_tensor(size, alpha=alpha, beta=beta)
        else:
            ts = [0, t, 1]
            size = 3
        ts[0] = 0
        ts[-1] = 1
        self.size = size
        self.coef = ts.to(torch_device)
        self.is_fused = is_fused

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        key0 = key[0:1]
        key2 = key[-1:]
        value0 = value[0:1]
        value2 = value[-1:]

        key0 = torch.cat([key0]*(self.size))
        key2 = torch.cat([key2]*(self.size))
        value0 = torch.cat([value0]*(self.size))
        value2 = torch.cat([value2]*(self.size))

        key0 = attn.head_to_batch_dim(key0)
        value0 = attn.head_to_batch_dim(value0)
        key2 = attn.head_to_batch_dim(key2)
        value2 = attn.head_to_batch_dim(value2)

        if self.is_fused:
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key2 = torch.cat([key, key2], dim=-2)
            value2 = torch.cat([value, value2], dim=-2)

        attention_probs12 = attn.get_attention_scores(query, key2, attention_mask)
        hidden_states12 = torch.bmm(attention_probs12, value2)
        hidden_states12 = attn.batch_to_head_dim(hidden_states12)
        
        if self.is_fused:
            key0 = torch.cat([key, key0], dim=-2)
            value0 = torch.cat([value, value0], dim=-2)

        attention_probs01 = attn.get_attention_scores(query, key0, attention_mask)
        hidden_states01 = torch.bmm(attention_probs01, value0)
        hidden_states01 = attn.batch_to_head_dim(hidden_states01)

        coef = self.coef.reshape(-1, 1, 1)

        hidden_states = (1 - coef) * hidden_states01 + coef * hidden_states12
        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)
        
        print("Out")

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class InterpolationAttnProcessorKeyValue:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, size=10, is_fused=False, alpha=1, beta=1, torch_device="cuda"):
        ts = generate_beta_tensor(size, alpha=alpha, beta=beta)
        ts[0] = 0
        ts[-1] = 1
        self.size = size
        self.coef = ts.to(torch_device)
        self.is_fused = is_fused

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        key0 = key[0:1]
        key2 = key[-1:]
        value0 = value[0:1]
        value2 = value[-1:]

        key0 = torch.cat([key0]*(self.size))
        key2 = torch.cat([key2]*(self.size))
        value0 = torch.cat([value0]*(self.size))
        value2 = torch.cat([value2]*(self.size))

        coef = self.coef.reshape(-1, 1, 1)
        key_cross = (1 - coef) * key0 + coef * key2
        value_cross = (1 - coef) * value0 + coef * value2

        key_cross = attn.head_to_batch_dim(key_cross)
        value_cross = attn.head_to_batch_dim(value_cross)

        if self.is_fused:
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key_cross = torch.cat([key, key_cross], dim=-2)
            value_cross = torch.cat([value, value_cross], dim=-2)

        attention_probs = attn.get_attention_scores(query, key_cross, attention_mask)

        hidden_states = torch.bmm(attention_probs, value_cross)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
   

def linear_interpolation(l1: FloatTensor, l2: FloatTensor, size=5):
    '''
    Linear interpolation
    
    Args:
    l1: Starting vector: (1, *)
    l2: Final vector: (1, *)
    size: int, number of interpolation points including l1 and l2
    
    Returns:
    Interpolated vectors: (size, *)
    '''
    assert l1.shape == l2.shape, "shapes of l1 and l2 must match"
    
    res = []
    for i in range(size):
        t = i / (size - 1)
        li = lerp(l1, l2, t)
        res.append(li)
    res = torch.cat(res, dim=0)
    return res


def sphere_interpolation(l1: FloatTensor, l2: FloatTensor, size=5):
    '''
    Linear interpolation
    
    Args:
    l1: Starting vector: (1, *)
    l2: Final vector: (1, *)
    size: int, number of interpolation points including l1 and l2
    
    Returns:
    Interpolated vectors: (size, *)
    '''
    assert l1.shape == l2.shape, "shapes of l1 and l2 must match"
    
    res = []
    for i in range(size):
        t = i / (size - 1)
        li = slerp(l1, l2, t)
        res.append(li)
    res = torch.cat(res, dim=0)
    return res


def slerp(v0: FloatTensor, v1: FloatTensor, t, threshold=0.9995):
    '''
    Spherical linear interpolation
    Args:
        v0: Starting vector
        v1: Final vector
        t: Float value between 0.0 and 1.0
        threshold: Threshold for considering the two vectors as
                                colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1
    '''
    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    # Normalize the vectors to get the directions and angles
    v0_norm: FloatTensor = torch.linalg.norm(v0, dim=-1)
    v1_norm: FloatTensor = torch.linalg.norm(v1, dim=-1)

    v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
    v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
    dot_mag: FloatTensor = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > threshold)
    can_slerp: LongTensor = ~gotta_lerp

    t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, Tensor) else 0
    t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
    out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        # print("no slerp")
        lerped: FloatTensor = lerp(v0, v1, t)

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


def generate_beta_tensor(size, alpha=3, beta=3):
    """
    Assume size as n
    Generates a PyTorch tensor of values [x0, x1, ..., xn-1] for the Beta distribution
    where each xi satisfies F(xi) = i/(n-1) for the CDF F of the Beta distribution.

    Args:
    size (int): The number of values to generate.
    alpha (float): The alpha parameter of the Beta distribution.
    beta (float): The beta parameter of the Beta distribution.

    Returns:
    torch.Tensor: A tensor of the inverse CDF values of the Beta distribution.
    """
    # Generating the inverse CDF values
    prob_values = [i / (size-1) for i in range(size)]
    inverse_cdf_values = beta_distribution.ppf(prob_values, alpha, beta)

    # Converting to a PyTorch tensor
    return torch.tensor(inverse_cdf_values, dtype=torch.float32)
