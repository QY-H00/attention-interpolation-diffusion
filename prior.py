import torch
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from lpips import LPIPS
from scipy.stats import beta as beta_distribution

from utils import compute_lpips, compute_smoothness_and_consistency


def baysian_prior_selection(
    interpolation_pipe,
    latent1: torch.FloatTensor,
    latent2: torch.FloatTensor,
    prompt1: str,
    prompt2: str,
    lpips_model: LPIPS,
    guide_prompt: str | None = None,
    negative_prompt: str = "",
    size: int = 3,
    num_inference_steps: int = 25,
    warmup_ratio: float = 1,
    early: str = "vfused",
    late: str = "self",
    target_score: float = 0.9,
    n_iter: int = 15,
    p_min: float | None = None,
    p_max: float | None = None,
) -> tuple:
    """
    Select the alpha and beta parameters for the interpolation using Bayesian optimization.

    Args:
        interpolation_pipe (any): The interpolation pipeline.
        latent1 (torch.FloatTensor): The first source latent vector.
        latent2 (torch.FloatTensor): The second source latent vector.
        prompt1 (str): The first source prompt.
        prompt2 (str): The second source prompt.
        lpips_model (any): The LPIPS model used to compute perceptual distances.
        guide_prompt (str | None, optional): The guide prompt for the interpolation, if any. Defaults to None.
        negative_prompt (str, optional): The negative prompt for the interpolation, default to empty string. Defaults to "".
        size (int, optional): The size of the interpolation sequence. Defaults to 3.
        num_inference_steps (int, optional): The number of inference steps. Defaults to 25.
        warmup_ratio (float, optional): The warmup ratio. Defaults to 1.
        early (str, optional): The early fusion method. Defaults to "vfused".
        late (str, optional): The late fusion method. Defaults to "self".
        target_score (float, optional): The target score. Defaults to 0.9.
        n_iter (int, optional): The maximum number of iterations. Defaults to 15.
        p_min (float, optional): The minimum value of alpha and beta. Defaults to None.
        p_max (float, optional): The maximum value of alpha and beta. Defaults to None.
    Returns:
        tuple: A tuple containing the selected alpha and beta parameters.
    """

    def get_smoothness(alpha, beta):
        """
        Black-box objective function of Baysian Optimization.
        Get the smoothness of the interpolated sequence with the given alpha and beta.
        """
        if alpha < beta and large_alpha_prior:
            return 0
        if alpha > beta and not large_alpha_prior:
            return 0
        if alpha == beta:
            return init_smoothness
        interpolation_sequence = interpolation_pipe.interpolate_save_gpu(
            latent1,
            latent2,
            prompt1,
            prompt2,
            guide_prompt=guide_prompt,
            negative_prompt=negative_prompt,
            size=size,
            num_inference_steps=num_inference_steps,
            warmup_ratio=warmup_ratio,
            early=early,
            late=late,
            alpha=alpha,
            beta=beta,
        )
        smoothness, _, _ = compute_smoothness_and_consistency(
            interpolation_sequence, lpips_model
        )
        return smoothness

    # Add prior into selection of alpha and beta
    # We firstly compute the interpolated images with t=0.5
    images = interpolation_pipe.interpolate_single(
        0.5,
        latent1,
        latent2,
        prompt1,
        prompt2,
        guide_prompt=guide_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        warmup_ratio=warmup_ratio,
        early=early,
        late=late,
    )
    # We compute the perceptual distances of the interpolated images (t=0.5) to the source image
    distances = compute_lpips(images, lpips_model)
    # We compute the init_smoothness as the smoothness when alpha=beta to avoid recomputation
    init_smoothness, _, _ = compute_smoothness_and_consistency(images, lpips_model)
    # If perceptual distance to the first source image is smaller, alpha should be larger than beta
    large_alpha_prior = distances[0] < distances[1]

    # Baysian optimization configuration
    num_warmup_steps = warmup_ratio * num_inference_steps
    if p_min is None:
        p_min = 1
    if p_max is None:
        p_max = num_warmup_steps
    pbounds = {"alpha": (p_min, p_max), "beta": (p_min, p_max)}
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.1)
    optimizer = BayesianOptimization(
        f=get_smoothness,
        pbounds=pbounds,
        random_state=1,
        bounds_transformer=bounds_transformer,
        allow_duplicate_points=True,
    )
    alpha_init = [p_min, (p_min + p_max) / 2, p_max]
    beta_init = [p_min, (p_min + p_max) / 2, p_max]

    # Initial probing
    for alpha in alpha_init:
        for beta in beta_init:
            optimizer.probe(params={"alpha": alpha, "beta": beta}, lazy=False)
            latest_result = optimizer.res[-1]  # Get the last result
            latest_score = latest_result["target"]
            if latest_score >= target_score:
                return alpha, beta

    # Start optimization
    for _ in range(n_iter):  # Max iterations
        optimizer.maximize(init_points=0, n_iter=1)  # One iteration at a time
        max_score = optimizer.max["target"]  # Get the highest score so far
        if max_score >= target_score:
            print(f"Stopping early, target of {target_score} reached.")
            break  # Exit the loop if target is reached or exceeded

    results = optimizer.max
    alpha = results["params"]["alpha"]
    beta = results["params"]["beta"]
    return alpha, beta


def generate_beta_tensor(
    size: int, alpha: float = 3, beta: float = 3
) -> torch.FloatTensor:
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
    prob_values = [i / (size - 1) for i in range(size)]
    inverse_cdf_values = beta_distribution.ppf(prob_values, alpha, beta)

    # Converting to a PyTorch tensor
    return torch.tensor(inverse_cdf_values, dtype=torch.float32)
