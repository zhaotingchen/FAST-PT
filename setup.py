from setuptools import setup, find_packages

setup(
    name='fastpt',
    description=(
        "FAST-PT is a code to calculate quantities in cosmological "
        "perturbation theory at 1-loop (including, e.g., corrections to the "
        "matter power spectrum)."),
    author="Joseph E. McEwen, Xiao Fang, and Jonathan Blazek (blazek@berkeley.edu)",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib'],
)
