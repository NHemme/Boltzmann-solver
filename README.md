# Welcome to the Boltzmann solver

This repository provides a Boltzmann solver for studying the evolution of dark sector particles in the early universe. The solver is designed to compute the relic abundance of particles as a function of temperature, incorporating various processes such as 3-2 processes and annihilation to Standard Model particles.

The implementation is based on the methodology described in [arXiv:2311.17157](https://arxiv.org/pdf/2311.17157), which explores the dynamics of a strongly interacting dark sector with light vector mesons during freeze-out and its implications for cosmology. The solver allows for the inclusion of different interaction rates and model parameters.

**Some notes on the physics behind the solver:**

It has been designed to determined the relic abundance of dark pions $\pi_D$ originating from an $SU(N_{C_D})$ with $SU(N_{f_D})$ quark-like fermions, specifically intended for a model that considers light rho/vector mesons $\rho_D$. Therefore, the ratio of the masses $r_m=\tfrac{m_{\rho_D}}{m_{A}}$ is an important parameter, but the code can easily be adapted to other models. The dark pions have been renamed to particle A, and the dark rho mesons to particle B.
