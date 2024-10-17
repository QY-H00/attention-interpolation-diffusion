import numpy as np
import torch
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from lpips import LPIPS
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_distribution

from transformers import CLIPImageProcessor, CLIPModel
from utils import compute_lpips, compute_smoothness_and_consistency


class BetaPriorPipeline:
    def __init__(self, pipe, model_ID="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_ID)
        self.preprocess = CLIPImageProcessor.from_pretrained(model_ID)
        self.pipe = pipe

    def _compute_clip(self, embedding_a, embedding_b):
        similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
        return 1 - similarity_score[0]

    def _get_feature(self, image):
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                image = self.preprocess(image, return_tensors="pt", do_rescale=False).pixel_values
            else:
                image = self.preprocess(image, return_tensors="pt").pixel_values
            embedding = self.model.get_image_features(image)
        return embedding

    def _update_alpha_beta(self, xs, ds):
        uniform_point = []
        ds_sum = sum(ds)
        for i in range(len(ds)):
            uniform_point.append(ds[i] / ds_sum)
        uniform_point = [0] + uniform_point
        uniform_points = np.cumsum(uniform_point)

        xs = np.asarray(xs)
        uniform_points = np.asarray(uniform_points)

        def beta_cdf(x, alpha, beta_param):
            return beta_distribution.cdf(x, alpha, beta_param)

        initial_guess = [1.0, 1.0]
        bounds = ([1e-6, 1e-6], [np.inf, np.inf])
        params, covariance = curve_fit(
            beta_cdf, xs, uniform_points, p0=initial_guess, bounds=bounds
        )

        fitted_alpha, fitted_beta = params
        return fitted_alpha, fitted_beta

    def _add_next_point(
        self,
        ds,
        xs,
        images,
        features,
        alpha,
        beta_param,
        prompt_start,
        prompt_end,
        negative_prompt,
        latent_start,
        latent_end,
        num_inference_steps,
        uniform=False,
        **kwargs
    ):
        idx = np.argmax(ds)
        A = xs[idx]
        B = xs[idx+1]
        F_A = beta_distribution.cdf(A, alpha, beta_param)
        F_B = beta_distribution.cdf(B, alpha, beta_param)

        # Compute the target CDF for t
        F_t = (F_A + F_B) / 2

        # Compute the value of t using the inverse CDF (percent point function)
        t = beta_distribution.ppf(F_t, alpha, beta_param)

        if uniform:
            idx = np.argmax(np.array(xs) - np.array([0] + xs[:-1])) - 1
            t = (xs[idx] + xs[idx+1]) / 2

        if t < 0 or t > 1:
            return xs, False

        ims = self.pipe.interpolate_single(
                t,
                prompt_start=prompt_start,
                prompt_end=prompt_end,
                negative_prompt=negative_prompt,
                latent_start=latent_start,
                latent_end=latent_end,
                early='fused_outer',
                num_inference_steps=num_inference_steps,
                **kwargs
            )

        added_image = ims.images[1]
        added_feature = self._get_feature(added_image)
        d1 = self._compute_clip(features[idx], added_feature)
        d2 = self._compute_clip(features[idx+1], added_feature)

        images.insert(idx+1, ims.images[1])
        features.insert(idx+1, added_feature)
        xs.insert(idx+1, t)
        del ds[idx]
        ds.insert(idx, d1)
        ds.insert(idx+1, d2)
        return xs, True

    def explore_with_beta(
        self,
        prompt_start,
        prompt_end,
        negative_prompt,
        latent_start,
        latent_end,
        num_inference_steps=28,
        exploration_size=16,
        init_alpha=3,
        init_beta=3,
        uniform=False,
        **kwargs
    ):
        xs = [0.0, 0.5, 1.0]
        images = self.pipe.interpolate_single(
            0.5,
            prompt_start=prompt_start,
            prompt_end=prompt_end,
            negative_prompt=negative_prompt,
            latent_start=latent_start,
            latent_end=latent_end,
            early='fused_outer',
            num_inference_steps=num_inference_steps,
            **kwargs
        )
        images = images.images
        images = [images[0], images[1], images[2]]
        features = [self._get_feature(image) for image in images]
        ds =[self._compute_clip(features[0], features[1]), self._compute_clip(features[1], features[2])]
        alpha = init_alpha
        beta_param = init_beta
        print("Alpha:", alpha, "| Beta:", beta_param, "| Current Coefs:", xs, "| Current Distances:", ds)
        while len(xs) < exploration_size:
            xs, flag = self._add_next_point(
                ds,
                xs,
                images,
                features,
                alpha,
                beta_param,
                prompt_start,
                prompt_end,
                negative_prompt,
                latent_start,
                latent_end,
                num_inference_steps,
                uniform=uniform,
                **kwargs
            )
            if not flag:
                break
            alpha, beta_param = self._update_alpha_beta(xs, ds)
            if uniform:
                alpha = 1
                beta_param = 1
            print(f"--------Exploration: {len(xs)} / {exploration_size}--------")
            print("Alpha:", alpha, "| Beta:", beta_param, "| Current Coefs:", xs, "| Current Distances:", ds)

        return images, features, ds, xs, alpha, beta_param

    def extract_uniform_points(self, ds, interpolation_size):
        expected_dis = sum(ds) / (interpolation_size - 1)
        current_sum = 0
        output_idxs = [0]
        for idx, d in enumerate(ds):
            current_sum += d
            if current_sum >= expected_dis:
                output_idxs.append(idx)
                current_sum = 0
        return output_idxs

    def extract_uniform_points_plus(self, features, interpolation_size):
        weights = -1 * np.ones((len(features), len(features)))
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                weights[i][j] = self._compute_clip(features[i], features[j])
        m = len(features)
        n = interpolation_size
        _, best_path = self.find_minimal_spread_and_path(n, m, weights)
        print("Optimal smooth path:", best_path)
        return best_path

    def find_minimal_spread_and_path(self, n, m, weights):
        # Collect all unique edge weights, excluding non-existent edges (-1)
        W = sorted({
            weights[i][j] for i in range(m - 1) for j in range(i + 1, m) if weights[i][j] != -1
        })
        min_weight = W[0]
        max_weight = W[-1]

        low = 0.0
        high = max_weight - min_weight
        epsilon = 1e-6  # Desired precision

        best_D = None
        best_path = None

        while high - low > epsilon:
            D = (low + high) / 2
            result = self.is_path_possible(D, n, m, weights, W)
            if result is not None:
                # A valid path is found
                high = D
                best_D = D
                best_path = result
            else:
                low = D

        return best_D, best_path

    def is_path_possible(self, D, n, m, weights, W):
        for w_min in W:
            w_max = w_min + D
            if w_max > W[-1]:
                break

            # Dynamic Programming to check for a valid path
            dp = [[None] * (n + 1) for _ in range(m)]
            dp[0][1] = (float('-inf'), float('inf'), [0])  # Start from x1 with path length 1

            for l in range(1, n):
                for i in range(m):
                    if dp[i][l] is not None:
                        max_w, min_w, path = dp[i][l]
                        for j in range(i + 1, m):
                            w = weights[i][j]
                            if w != -1 and w_min <= w <= w_max:
                                # Update max and min weights along the path
                                new_max_w = max(max_w, w)
                                new_min_w = min(min_w, w)
                                new_diff = new_max_w - new_min_w
                                if new_diff <= D:
                                    dp_j_l_plus_1 = dp[j][l + 1]
                                    if dp_j_l_plus_1 is None or new_diff < (dp_j_l_plus_1[0] - dp_j_l_plus_1[1]):
                                        dp[j][l + 1] = (new_max_w, new_min_w, path + [j])

            if dp[m - 1][n] is not None:
                # Reconstruct the path
                _, _, path = dp[m - 1][n]
                return path  # Return the path if found

        return None  # Return None if no valid path is found

    def generate_interpolation(
        self,
        prompt_start,
        prompt_end,
        negative_prompt,
        latent_start,
        latent_end,
        num_inference_steps=28,
        exploration_size=16,
        init_alpha=3,
        init_beta=3,
        interpolation_size=7,
        uniform=False,
        **kwargs
    ):
        images, features, ds, xs, alpha, beta_param = self.explore_with_beta(
            prompt_start,
            prompt_end,
            negative_prompt,
            latent_start,
            latent_end,
            num_inference_steps,
            exploration_size,
            init_alpha,
            init_beta,
            uniform=uniform,
            **kwargs
        )
        # output_idx = self.extract_uniform_points(ds, interpolation_size)
        output_idx = self.extract_uniform_points_plus(features, interpolation_size)
        output_images = []
        for idx in output_idx:
            output_images.append(images[idx])

        # for call_back
        self.images = images
        self.ds = ds
        self.xs = xs
        self.alpha = alpha
        self.beta_param = beta_param

        return output_images


def bayesian_prior_selection(
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
        Black-box objective function of Bayesian Optimization.
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
        smoothness, _, _ = compute_smoothness_and_consistency(interpolation_sequence, lpips_model)
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

    # Bayesian optimization configuration
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

def generate_beta_tensor(size: int, alpha: float = 3, beta: float = 3) -> torch.FloatTensor:
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
