"""Print the installed versions used by the assignment, without unrelated packages."""
from importlib.metadata import version

for name in ("jax", "jaxlib", "equinox", "chex", "optax", "numpy", "scipy", "ml_dtypes",
             "opt_einsum", "jaxtyping", "wadler-lindig", "absl-py", "toolz", "typing_extensions",
             "matplotlib", "imageio", "imageio-ffmpeg", "pillow", "pytest"):
    print(f"{name}=={version(name)}")
