import numpy as np
from scipy.integrate import solve_ivp
from functionsFile import *
from __main__ import g_starContent, outputPath

m_pl = 1.22E19  # Planck mass in GeV

class BSolution:
    """
    A class to solve the Boltzmann equation for dark matter relic abundance.
    """

    def __init__(self, rm, mA, name=None, Nf=3, Nc=3, exclB=False, inclThermW=False, inclAnn=False, kappa=0):
        """
        Initialize the BSolution object with model parameters.

        Parameters:
        - rm: Ratio of m_rho/m_A
        - fA: Decay constant (optional, used to calculate mA if not provided)
        - mA: Mass of the dark matter particle
        - name: Name of the solution (optional)
        - Nf: Number of dark flavors
        - Nc: Number of dark colors
        - exclB: Whether to exclude 3A-A+B process (appropriate e.g. if mB > 2*mA)
        - inclThermW: Whether to include thermal width effects
        - inclAnn: Whether to include annihilation processes
        - kappa: Kinetic mixing parameter between rho_D and SM photon
        """
        self.Nf = Nf
        self.Nc = Nc
        self.name = name
        self.rm = rm
        self.mA = mA
        self.fA = mA / (7.79 * (1 / rm) + 0.57 * ((1 / rm) ** 2))
        self.kappa = kappa
        self.gi = Nf**2 - 1  # Degrees of freedom of A
        self.exclB = exclB
        self.inclThermW = inclThermW
        self.inclAnn = inclAnn

        # Variables to store results
        self.x_freeze = None
        self.solution = None
        self.solutions_xopt = None

    def dWdx(self, x, W):
        """
        Differential equation for dW/dx, where W = log(Y).

        Parameters:
        - x: Dimensionless inverse temperature (mA/T)
        - W: Logarithm of the yield Y=n/s, dimensionless

        Returns:
        - dW/dx: The rate of change of W with respect to x
        """
        T = self.mA / x  # Temperature
        g_star_sqrt = gStarSqrt(T)  # Effective degrees of freedom
        s = (2 * np.pi**2 / 45) * hEff(T) * T**3  # Entropy density
        n = self.gi * (self.mA * T / (2 * np.pi))**(3 / 2) * np.exp(-x)  # Number density of A
        Y_eq = n / s  # Equilibrium yield
        W_eq = np.log(Y_eq)  # Logarithm of equilibrium yield

        # Calculate dW/dx based on included processes
        if self.exclB:
            # Only 3A -> 2A process
            σvv = crossSectionPionsOnly(self.fA, self.mA, x, Nc=self.Nc, Nf=self.Nf)
            difference = np.sqrt(np.pi / 45) * m_pl * s * self.mA / (x**2) * g_star_sqrt * σvv * (np.exp(W_eq + W) - np.exp(2 * W))
        else:
            # 3A -> A+B process and optionally 3A -> 2A process
            σvv = crossSection(self.fA, self.mA, self.rm, Nf=self.Nf, inclThermW=self.inclThermW, x=x)
            dPionsRho = np.sqrt(np.pi / 45) * m_pl * s * self.mA / (x**2) * g_star_sqrt * σvv * (np.exp(2 * W_eq) - np.exp(2 * W))
            dPionsOnly = 0
            if self.Nf > 2:
                σvv = crossSectionPionsOnly(self.fA, self.mA, x, Nc=self.Nc, Nf=self.Nf)
                dPionsOnly = np.sqrt(np.pi / 45) * m_pl * s * self.mA / (x**2) * g_star_sqrt * σvv * (np.exp(W_eq + W) - np.exp(2 * W))
            difference = dPionsOnly + dPionsRho

        if self.inclAnn:
            # Include 2A -> e+e- via off-shell B
            σvAnn = crossSectionAnn(self.mA, self.rm, x, self.Nc, self.Nf, self.kappa)
            dPionsAnn = np.sqrt(np.pi / 45) * m_pl * self.mA / (x**2) * g_star_sqrt * σvAnn * (np.exp(2 * W_eq - W) - np.exp(W))
            difference += dPionsAnn

        return difference

    def solve(self, x_init, x_inf):
        """
        Solve the Boltzmann equation using the initial and final x values.

        Parameters:
        - x_init: Initial value of x (mA/T)
        - x_inf: Final value of x (mA/T)
        """
        T = self.mA / x_init
        gi = self.gi
        s = T**3 * (2 * hEff(T) * (np.pi**2) / 45)  # Entropy density
        n = gi * (self.mA * T / (2 * np.pi))**(3 / 2) * np.exp(-x_init)  # Number density
        self.Y_init = n / s  # Initial yield

        # Solve the differential equation
        self.solution = solve_ivp(self.dWdx, [x_init, x_inf], [np.log(self.Y_init)], method='Radau', max_step=1)

    def solve_xopt(self, x_init, x_inf, step=2, n_iterations=20, tol=2E-3, printIterations=False):
        """
        Optimized solver for the Boltzmann equation with iterative adjustment of x_init.

        Parameters:
        - x_init: Initial value of x (mA/T)
        - x_inf: Final value of x (mA/T)
        - step: Step size for adjusting x_init
        - n_iterations: Maximum number of iterations
        - tol: Tolerance for convergence
        - printIterations: Whether to print iteration details
        """
        T = self.mA / x_init
        s = T**3 * (2 * hEff(T) * (np.pi**2) / 45)  # Entropy density
        n = self.gi * (self.mA * T / (2 * np.pi))**(3 / 2) * np.exp(-self.mA / T)  # Number density
        Y_init = n / s  # Initial yield

        diff, diff_sign = 1, 1
        SOL0 = solve_ivp(self.dWdx, [x_init, x_inf], [np.log(Y_init)], method='Radau', max_step=1)
        solutions = [SOL0]
        x_initList = [x_init]

        for i in range(1, n_iterations):
            x_init = x_init - step
            x_initList.append(x_init)

            T = self.mA / x_init
            s = T**3 * (2 * hEff(T) * (np.pi**2) / 45)
            n = self.gi * (self.mA * T / (2 * np.pi))**(3 / 2) * np.exp(-self.mA / T)
            Y_init = n / s

            SOLi = solve_ivp(self.dWdx, [x_init, x_inf], [np.log(Y_init)], method='Radau')
            Yi_inf = np.exp(SOLi.y[0][-1])
            Y0_inf = np.exp(solutions[i - 1].y[0][-1])
            solutions.append(SOLi)

            diff = (Yi_inf - Y0_inf) / Yi_inf

            if printIterations:
                print('i=%i, x_init = %.2f, step = %.2E, Y0_inf = %.4E, Yi_inf = %.4E, diff = %.2E, status = %i' %
                      (i, x_init, step, Y0_inf, Yi_inf, diff, solutions[-1].status))

            if solutions[-1].status != 0:
                print('status = %i' % (solutions[-1].status))
                step = -step * 0.5
                continue

            if diff_sign != np.sign(diff):
                step = -step * 0.5

            diff_sign = np.sign(diff)
            if abs(diff) < tol:
                self.solution = solutions[-2]  # Use the previous solution
                self.x_freeze = x_initList[-2]
                if printIterations:
                    print('Optimisation was successful.')
                break

            if i == n_iterations:
                print('Optimisation was not successful. Will use a low default x_init.')
                self.solution = solve_ivp(self.dWdx, [10, x_inf], [np.log(self.Y_init)], method='Radau')

        self.solutions_xopt = solutions

        if diff > tol:
            print('Optimisation was not successful. Increase n_iterations, lower tol or change x_init.')

