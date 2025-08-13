import importlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from scipy.optimize import root_scalar
from matplotlib import rc
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from IPython.display import display, clear_output
from __main__ import g_starContent, outputPath

from solutionClass import BSolution
import solutionClass
importlib.reload(solutionClass)

m_pl = 1.22E19 # [GeV]

###################################################################################################################
# This section includes basic and important functions for the solutionClass to work
###################################################################################################################

gStarSqrt = interpolate.interp1d(g_starContent[0].tolist(), g_starContent[1].tolist())
hEff = interpolate.interp1d(g_starContent[0].tolist(), g_starContent[2].tolist())

def crossSection(fpi, mpi, r, Nf = 3, inclThermW = True, x = None):
    mrho = r * mpi
    Sa, Sb = 6, 1
    N = Nf #Is this right?
    
    p4_tilde = np.sqrt((9*mpi**2-(mrho+mpi)**2)*(9*mpi**2-(mrho-mpi)**2))/(6*mpi)
    crossSection_factor = 1/(32*np.pi*Sa*Sb*mpi**3) * p4_tilde/(np.sqrt(p4_tilde**2+mpi**2)+np.sqrt(p4_tilde**2+mrho**2))
    kappa = 2*(37*Nf**4 - 30*Nf**2 + 18) / (3*Nf*(Nf**2-1)**2)
    y = mrho**2/(4*mpi**2)
    
    if inclThermW == True:
        Gamma = 8 * np.pi * (Nf**2 - 1)/(x**2) * np.exp(-x) * mpi**3 * 3*mpi**2 / (64*np.pi*fpi**4)

        M2_total_nomT1 = (-20*mpi**2*mrho**2 + mrho**4 + 64*mpi**4) * (4*Gamma**2*mpi**2 + mrho**4)
        M2_total_nomT2 = (5*Gamma**2*Nf**4*(13*mrho**2 + 8*mpi**2)**2 + 64*mpi**2*(4*mpi**2*(245*Nf**4 - 114*Nf**2 + 36)*mrho**2 + (821*Nf**4 - 168*Nf**2 + 36)*mrho**4 + 8*mpi**4*(37*Nf**4 - 30*Nf**2 + 18)))
        M2_total_denom = 96*fpi**6*Nf*((Nf**2-1)**2) * (Gamma**2 + 64*mpi**2)*((mrho**2 + 2*mpi**2)**2) * (9*(Gamma**2)*(mpi**2) + 4*((mrho**2-4*mpi**2)**2))
                          
        M2_total = (M2_total_nomT1 * M2_total_nomT2) / M2_total_denom                  
                          
    else:         
        M2_total_nomT1 = (16*mpi**2-mrho**2)
        M2_total_nomT2 = ((821*N**4-168*N**2+36)*mrho**8 + 4*mpi**2*(245*N**4 - 114*N**2 + 36)*mrho**6 + 8*mpi**4*(37*N**4 - 30*N**2 + 18)*mrho**4)
        M2_total_denom = (384*fpi**6*N*(N**2-1)**2*(4*mpi**2-mrho**2)*(mrho**2+2*mpi**2)**2)
                          
        M2_total = M2_total_nomT1 * M2_total_nomT2 / M2_total_denom
    
    crossSection = M2_total * crossSection_factor 

    return crossSection



def crossSectionPionsOnly(fpi, mpi, x, Nc = 3, Nf = 3):
    kappa = Nf**2 * (Nf**2 - 4) / ((Nf**2 - 1)**2)
    xi = mpi/fpi
    
    nom = 5 * np.sqrt(5) * Nc**2 * kappa * xi**10
    denom = 1536 *np.pi**5 * Nf * mpi**5 * x**2
    crossSection = 1/3 * nom / denom

    return crossSection


def crossSectionAnn(mpi,r,x,Nc,Nf,kappa=1): # Not finished
    mrho = r * mpi
    a = 1/137
    traceID = Nf*(Nf**2-1)/4
    Npi = Nf**2-1
    aD = 1/(4*np.pi)
    g_l = mrho**2/(10E3*5.7) * 2 * kappa # Defined in Eq. C3 of arxiv:2311.17157 using g=5.7, e=1, e_D=1, mZ'=1TeV

    A_ann = 4*np.pi * traceID * g_l**2 * aD * a / (3*Npi**2)
    s_int = integrate.quad(lambda s: s**(3/4)*(s-1)**(3/2)*np.exp(-2*x*np.sqrt(s))/((s-mrho**2/(4*mpi**2))**2), 1, np.inf)[0]
    crossSection = 4/(np.sqrt(np.pi)) * x**(3/2)/(mpi**2) * A_ann * np.exp(2*x) * s_int
    
    return crossSection



###################################################################################################################
# This section is concerned about the plotting of a model's distribution as a function of x=mA/T
# Each model is an instance of the solutionClass
# The plotBoltzmannModels function can plot several models, defined by whether they include the 3A-AB process or not.
###################################################################################################################
def makeEqDistribution(mA, Nf,
                       x_init=1, 
                       x_inf=100):
    '''
    This function will take the dark pion mass and calculate the Y equilibrium distribution.
    It returns two list; the x-values and Y equilibrium distribution values

    Parameters:
    - mA: Mass of the dark matter candidate in GeV
    - Nf: Number of dark flavors
    - x_init: Initial x value (mA/T)
    - x_inf: Final x value (mA/T)
    '''
    debugFile = open('./outputs/debugFile.txt','w') # Can be deleted
    step = 0.5
    
    gi = Nf**2-1
    T_init = mA/x_init
    h_eff = hEff(T_init)
    Y_init = (1/(2*np.pi))**(3/2) / (2*(np.pi**2)/45) * gi/h_eff * x_init**(3/2) * np.exp(-x_init)
    
    xList = np.arange(x_init,x_inf+step,step)
    YeqList = []
    i = 0
    for x in xList:
        T_i = mA/x
        h_eff = hEff(T_i)
        Y_i = (1/(2*np.pi))**(3/2) / (2*(np.pi**2)/45) * gi/h_eff * x**(3/2) * np.exp(-x)
        YeqList.append(Y_i)

    return xList, YeqList


def plotBoltzmannModels(modelsAB=[], modelsA=[], x_init=10, x_inf=50, fileName='BoltzmannSolution'):
    """
    This function is a wrapper for the plotBoltzmann function that uses a BSolution object.
    It extracts the necessary parameters from the model and calls plotBoltzmann.

    Parameters:
    - modelsAB: List of BSolution objects (already solved) that assume inclusion of the 3A-AB process
    - modelsA: List of BSolution objects (already solved) that only include the 3A-2A process
    - x_init: Initial x value (mA/T)
    - x_inf: Final x value (mA/T)
    - fileName: Name of the output file to save the plot
    """

    # Check if both modelsAB and modelsA are empty
    if not modelsAB and not modelsA:
        raise ValueError("Both modelsAB and modelsA are empty. Please provide at least one model.")
        return
    mA_original = False  # Used to check if all models have the same mA value

    # Check if modelsAB and modelsA are lists, if not convert them to lists
    if not isinstance(modelsAB, list): modelsAB = [modelsAB]
    if not isinstance(modelsA, list): modelsA = [modelsA]

    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.setp(ax.spines.values(), lw=0.2)

    # Cosmetic settings
    plt.rcParams.update({'font.size': 18})
    linelist = ['-', '--', '-.', ':']
    if len(modelsAB) > len(linelist):
        raise ValueError("Too many models provided. Please reduce the number of models or increase the linelist size.")
        return
    blueColor = '#2c7bb6'
    redColor = '#d7191c'

    # Set to track added labels - will prevent duplicate labels in the legend
    # Since we iterate over modelsAB first, it will choose blue legend labels first
    added_labels = set()

    # Plot each model that includes 3A-AB process
    if modelsAB:
        i_counter = 0  # Used to iterate through the linelist

        for model in modelsAB:
            mA = model.mA

            # Check if the model has the same mA value as other models
            if mA_original == False: mA_original = mA
            elif mA_original != mA:
                raise ValueError("All models must have the same mA value.")
                return

            # Generate the label
            label = r'$m_{B} / m_{A} = %.2f$ ($\xi  = $%.2f)' % (model.rm, model.mA / model.fA)

            # Plot the solution of the model
            ax.plot(model.solution.t, np.exp(model.solution.y[0]),
                    color=blueColor, linewidth=2, linestyle=linelist[i_counter],
                    label=label if label not in added_labels else None)

            # Add the label to the set
            added_labels.add(label)
            i_counter += 1

        # Plot each model that only has 3A-2A process
    if modelsA:
        i_counter = 0  # Used to iterate through the linelist

        for model in modelsA:
            mA = model.mA

            # Check if the model has the same mA value as other models
            if mA_original == False: mA_original = mA
            elif mA_original != mA:
                raise ValueError("All models must have the same mA value.")
                return

            # Generate the label
            label = r'$m_{B} / m_{A} = %.2f$ ($\xi  = $%.2f)' % (model.rm, model.mA / model.fA)

            # Plot the solution of the model
            ax.plot(model.solution.t, np.exp(model.solution.y[0]),
                    color=redColor, linewidth=2, linestyle=linelist[i_counter],
                    label=label if label not in added_labels else None)

            # Add the label to the set
            added_labels.add(label)
            i_counter += 1

    # Calculate and plot the equilibrium distribution
    xeqList, YeqList = makeEqDistribution(mA_original, model.Nf, x_init, x_inf)
    ax.plot(xeqList, YeqList, dashes=[8, 4], color='grey', linewidth=1.5)

    # Calculate and plot the relic density line
    relic = []
    for xi in xeqList:
        relic.append(0.12 / (2970 * mA) * 1.054E-5)
    ax.plot(xeqList, relic, color='black', linewidth=1.5, dashes=[6, 2, 2, 2], zorder=0.5)

    # Axis settings and labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([5, 10, 20, 30, 40, 50, 100, 200])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_formatter('')
    ax.set_ylim(1e-10, 1e-4)
    ax.set_xlim(x_init, x_inf)

    ax.set_xlabel('$x=m_{A}/T$', fontsize=18)
    ax.set_ylabel('$Y_{A}=n_{A}/s$', fontsize=18)

    # Add legend
    ax.legend(fontsize=16, title=r'$m_{A} = %.0f$ MeV, $N_{f_D} = %i, N_{c_D} = %i$' % (mA_original * 1000, model.Nf, model.Nc), title_fontsize=16, alignment='left', frameon=False)

    # Add text annotations for the processes
    if modelsAB:
        fig.text(0.7, 0.261, r'$3A \to AB \text{ or } 2A$', fontsize=16, color=blueColor, rotation=0)
    if modelsA:
        fig.text(0.7, 0.575, r'$3A \to 2 A$ only', fontsize=16, color=redColor, rotation=0)

    # Add text for relic density and equilibrium distribution
    fig.text(0.25, 0.37, r'$\Omega_{DM} h^2 = 0.12$', fontsize=16, color="black", rotation=0)
    fig.text(0.62, 0.2, r'$Y_{A}^\mathrm{eq}$', fontsize=16, color="grey", rotation=0)

    plt.tight_layout()
    plt.savefig('%s%s.pdf' % (outputPath, fileName))
    plt.show()


def modelResults(solutionObjects):
    nObjects = 1
    if isinstance(solutionObjects,list): nObjects = len(solutionObjects)

    for i in range(nObjects):
        SOLi = solutionObjects[i]
        mpi = SOLi.mpi
        g_0 = float(g_starContent[1].tolist()[0])
        s_0 = mpi**3*(2*g_0*(np.pi**2)/45)
        Hm = 1.67*np.sqrt(g_0)*mpi**2/m_pl
        g_x1 = gStarSqrt(mpi)
        s_x1 = mpi**3*(2*g_x1*(np.pi**2)/45)
        #l = s_x1*SOLi.crossSection/Hm

        Y_inf = np.exp(SOLi.solution.y[0][-1])
        omega_dark = Y_inf * 2970 * mpi / 1.054E-5 # s_0 = 2970 cm^-3, rho_crit = 1.054*10^-5 h^2 GeV cm^-3
        
        print('''
########################################################################################################
Results for model "%s" with m_π = %i GeV and r = %.2f:
--------------------------------------------------------------------------------------------------------
Y_inf = %.2E 
Ω_d*h^2 = Y_inf * s_0 * m / rho_crit = %.4f
########################################################################################################
        '''%(SOLi.name, mpi, SOLi.r, Y_inf, omega_dark))

        
        
###################################################################################################################
# This section includes a function to make a line plot of ξ (m_π/f_π) as a function of m_π that gives the 
# correct relic abundance. It includes the Bullet cluster constraints.
###################################################################################################################
    
def find_mass(r, Nf, Nc, pionsOnly = False, inclThermW = True, inclAnn = False, kappa = 0):

    def omegah2(m):
        model = BSolution(r, mpi = m, name='Test', Nf=Nf, Nc=Nc, pionsOnly = pionsOnly, inclThermW = inclThermW, inclAnn = inclAnn, kappa = kappa)
        x1=10
        model.solve(x_init=x1, x_inf=50)
        om = np.exp(model.solution.y[0][-1]) * 2970 * model.mpi / 1.054E-5 
        return om - 0.12
    
    sol = root_scalar(omegah2, bracket=(0.01,1), method='brentq')
    
    return sol.root


def find_bc_constraints(r, bound = 2):

    def bc(m):
        fpi = m * (0.129*r - 0.013)
        crossSection = 3/(64*np.pi) * m**2/(fpi**4) * (2E-14)**2 # GeV^2 to cm^2
        m = m*1.8E-24 # GeV to g
        return crossSection/m - bound
    
    sol = root_scalar(bc, bracket=(0.01,100), method='brentq')
    
    return sol.root


def findkappa(model,abundance,tol=0.1,x_init=10,x_inf=50):

    def kappaTol(kappa):
        model.kappa = kappa
        model.solve(x_init=x_init, x_inf=x_inf)
        om = np.exp(model.solution.y[0][-1]) * 2970 * model.mpi / 1.054E-5
        
        return abundance - (1+tol)*om

    sol = root_scalar(kappaTol, bracket=(1E-9,1), method='brentq')
    
    return sol.root


def makeFig3(inputR = np.arange(1.6,2,0.1),
             x_init = 30, 
             x_inf = 100,
             Nf = 2,
             Nc = 3,
             pionsOnly = False,
             inclThermW = True,
             inclAnn = False, 
             kappa = 0,
             fileName = 'fig3rep'):
    
    # Set up arrays and files to write to
    Mpi = np.empty(shape=[0,1])
    R = np.empty(shape=[0,1])
    Fpi = np.empty(shape=[0,1])
    Xi = np.empty(shape=[0,1])
    outputFile = '%s%s'%(outputPath,fileName)
    
    # Set up to record progress during run of code
    n_points = len(inputR)
    n_progr = 0
    
    # Begin loop through 2D grid
    for i in range(len(inputR)):
        r = inputR[i]
        mpi = find_mass(r, Nf = Nf, Nc = Nc, pionsOnly = pionsOnly, inclThermW = inclThermW, inclAnn = inclAnn, kappa = kappa)
        fpi = mpi * (0.18287*r - 0.01940) / np.sqrt(2)
        xi = mpi/fpi
        
        Mpi = np.append(Mpi, mpi)
        Fpi = np.append(Fpi, fpi)
        Xi = np.append(Xi, xi)
        
        n_progr = n_progr + 1
        progress = 100/n_points*n_progr
        clear_output(wait=True)
        display('Have performed %.2f%% of the scan'%(progress))
    
    df = pd.DataFrame(np.array([inputR, Mpi, Fpi, Xi])).T
    df.columns = ['m_ρ/m_π', 'm_π', 'f_π', 'ξ']
    df.to_csv(outputFile)
    
    return df


def plotFig3(dataframes, labels, figureName = 'fig3'):
    if not isinstance(dataframes,list): dataframes = [dataframes]
    if not isinstance(labels,list): labels = [labels]
            
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Axis settings and save the figure
    plt.grid(True)
    ax.set_xlabel('$m_π$ [MeV]', fontsize = 30)
    ax.set_ylabel('ξ', fontsize = 30, rotation=0)
    ax.set_xscale('log')
    
    for i in range(len(dataframes)):
        df = dataframes[i]
        label = labels[i]
        color = ['turquoise','dodgerblue','goldenrod','orangered'][i]
        ax.plot(df['m_π']*1000, df['ξ'], label=label, linewidth=2.0, color = color)
       
    
    secax_y = ax.secondary_yaxis('right',functions=(lambda y: 0.106086 + 7.73344/y, lambda y: np.sqrt(2)/(0.18287*y-0.01940)))
    secax_y.set_ylabel('$m_ρ/m_π$', fontsize = 30, rotation=0)
    
    bc_m = []
    # Put BC constraints on
    for i in range(len(df['ξ'])):
        xi = df['ξ'][i]
        r = 0.106086 + 7.73344/xi
        mpi = find_bc_constraints(r)
        bc_m.append(mpi*1000)
        
    
    ax.plot(bc_m, df['ξ'], color = 'k', linewidth=2.0)
    fig.text(0.28, 0.23, 'Bullet Cluster', fontsize = 20,
                     bbox=dict(boxstyle="square",
                     ec='grey',
                     fc='white', alpha=0.0))
    
    # Fill left of BC constraints
    fillx = np.append(bc_m,[0])
    filly1 = np.append(df['ξ'],min(dataframes[0]['ξ']))
    filly2 = max(dataframes[0]['ξ'])*len(fillx)
    ax.fill_between(fillx, filly1, filly2, alpha=0.4, color='grey')
    
    ax.set_ylim(min(dataframes[0]['ξ']),max(dataframes[0]['ξ']))
    
    # Set ticks and legend
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.grid(True)
    ax.legend(fontsize = 20)
    
    plt.savefig('%s%s.pdf'%(outputPath, figureName))
    plt.show()
    
    
    
###################################################################################################################
# This section has the function plotXopt that illustrates how the solve_xopt function of the solutionClass works 
###################################################################################################################    
        
def plotXopt(model, x_init, x_inf):
    n_solutions = len(model.solutions_xopt)
    m = model.mpi
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for i in range(n_solutions):
        SOLi = model.solutions_xopt[i]
        line = ['-','--'][i%2]
        ax.plot(SOLi.t, np.exp(SOLi.y[0]), linestyle = line, alpha = 0.7, label = i)
        
    xeqList, YeqList = makeEqDistribution(m, x_init, x_inf/2)
    ax.plot(xeqList,YeqList, '--', color='k', alpha = 0.9, label = 'Y_eq (m=%i) GeV'%(m))

    # Axis settings and save the figure
    ax.legend()
    plt.xscale('log')
    plt.yscale('log')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid(True)
    ax.set_xticks(np.arange(x_init,x_inf+1,10))
    ax.set_xlabel('x=m/T')
    ax.set_ylabel('Y=n/s')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    secax_x = ax.secondary_xaxis('top',functions=(lambda x: m/x,lambda x: m/x))
    secax_x.set_xlabel('Temperature T [GeV]')


    plt.savefig('%sx_initOpt.pdf'%(outputPath))
    plt.show()

