import numpy as np

# Add common distributions so they can be imported directly; assigned to a variable since used in help messages
__all__ = [
    "constant",
    "exponential",
    "gamma",
    "lognormal",
    "normal",
    "poisson",
    "uniform",
]


class Distribution:
    def __init__(self, dist_type, **params):
        """
        Initializes the distribution.

        :param dist_type: Type of distribution (e.g., "poisson", "uniform", "gamma", etc.).
        :param params: Parameters specific to the chosen distribution.
        """
        self.dist_type = dist_type
        self.params = params

        # Validate the distribution type
        supported_distributions = {
            "constant",
            "exponential",
            "gamma",
            "lognormal",
            "normal",
            "poisson",
            "uniform",
        }
        if dist_type not in supported_distributions:
            raise ValueError(f"Unsupported distribution: {dist_type}. Supported: {supported_distributions}")

    def sample(self, size=1):
        """
        Generates random samples from the specified distribution.

        :param size: Number of samples to generate.
        :return: A NumPy array of sampled values.
        """
        if self.dist_type == "constant":
            return np.full(size, self.pars.get("value", 1))
        elif self.dist_type == "exponential":
            return np.random.exponential(self.params.get("scale", 1.0), size)
        elif self.dist_type == "gamma":
            return np.random.gamma(self.params.get("shape", 2.0), self.params.get("scale", 1.0), size)
        elif self.dist_type == "lognormal":
            return np.random.lognormal(self.params.get("mean", 1.0), self.params.get("sigma", 0.5), size)
        elif self.dist_type == "normal":
            return np.random.normal(self.params.get("mean", 0.0), self.params.get("std", 1.0), size)
        elif self.dist_type == "poisson":
            return np.random.poisson(self.params.get("lam", 5), size)
        elif self.dist_type == "uniform":
            return np.random.randint(self.params.get("min", 2), self.params.get("max", 10), size)
        else:
            raise ValueError(f"Unsupported distribution type: {self.dist_type}")

    def __call__(self, size=1):
        """
        Allows the instance to be called like a function to sample values.

        :param size: Number of samples to generate.
        :return: A NumPy array of sampled values.
        """
        return self.sample(size)

    def __repr__(self):
        return f"Distribution(type={self.dist_type}, params={self.params})"


# Define helper functions for common distributions
def constant(value):
    return Distribution("constant", value=value)


def exponential(scale):
    return Distribution("exponential", scale=scale)


def gamma(shape, scale):
    return Distribution("gamma", shape=shape, scale=scale)


def lognormal(mean, sigma):
    return Distribution("lognormal", mean=mean, sigma=sigma)


def normal(mean, std):
    return Distribution("normal", mean=mean, std=std)


def poisson(lam):
    return Distribution("poisson", lam=lam)


def uniform(min, max):
    return Distribution("uniform", min=min, max=max)
