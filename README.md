# FAST-PT

FASTPT is a code to calculate quantities in cosmological perturbation theory at 1-loop (including, e.g., corrections to the matter power spectrum). 
The code utilizes Fourier methods combined with analytic expressions to reduce the computation time
to scale as N log N, where N is the number of grid points in the input linear power spectrum. 

FAST-PT way to get started: 

* Make sure you have current numpy, scipy, and matplotlib
* download FAST-PT (or clone the repo)
* in terminal type the following:
* python FASTPT.py
* (hopefully you get a plot)

See the user_manual.pdf for more details. 

Our papers (arXiv:1603.04826 and arXiv:1609.05978) describe the FAST-PT algorithm and implementation. Please cite these papers when using FAST-PT in your research.

