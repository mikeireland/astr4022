"""
This is a solver for the grey atmosphere, using the discrete ordinate method.
There is 1 first order differential equation per ray direction mu, that 
includes an integral over the specific intensity I.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
plt.ion()

#Make an evenly spaced grid of \mu points, and a theta grid for plotting convenience.
#The splu function that is part of the boundary value solver has a time required 
#proportional to something like the 4th power of n! We get about 5 correct decimal 
#places with n=140.
n = 70
mu_g = (2*np.arange(2*n) + 1 - 2*n)/(2*n)
theta_deg_g = np.degrees(np.arccos(mu_g))

#Prepare the optical depth list and Intensity list
taus = np.array([0,0.1,0.2, 0.5,1,1.5,2,3,5,10])
Iinit=[]

#Prepare the figure and printout
plt.figure(1)
plt.clf()
print("tau  H(tau) [Eddington]")

#Iterate through optical depths, finding the Eddington approximate solution as an initial
#guess, and making a plot.
for tau in taus:
	this_I = 3*(tau + 2/3 + mu_g)
	mu_neg = mu_g[mu_g<0]
	this_I[mu_g<0] = 3*((tau + 2/3 + mu_neg)*(1-np.exp(tau/mu_neg)) + tau*np.exp(tau/mu_neg))
	Iinit += [this_I] 
	plt.plot(theta_deg_g, this_I, label=r'$\tau$={:.1f}'.format(tau))
	print('{:5.2f} {:.3f}'.format(tau, 0.5*np.sum(this_I*mu_g)/n))
plt.legend()
plt.xlabel(r'$\theta$ (deg)')
plt.ylabel('I/H')
plt.axis([0,180,0,12])

#Create the 2 functions needed for the boundary value problem.
#Note that these use global variables n and mu_g
def Ideriv(tau, I):
	"""For solve_bvp - compute the specific intensity derivative,
    equal to (I-J)/mu."""
	#As we don't have parameters, re-compute mu
	mu_g_rep = np.repeat(mu_g, I.shape[1]).reshape(I.shape)
	J = np.sum(I, axis=0)/n/2
	return (I-J)/mu_g_rep
	
def bc(Ia, Ib):
	"""Fix the lower and upper boundary conditions. The 
    lower boundary is for upgoing radiation [n:], the upper for
	downgoing radiation [:n]."""
	J = np.sum(Ib, axis=0)/n/2
	mu_g = (2*np.arange(2*n) + 1 - 2*n)/(2*n)
	return np.concatenate((J + 3*mu_g[n:] - Ib[n:], Ia[:n]))
	
#Solve the boundary problem!
Iinit = np.array(Iinit).T
res = solve_bvp(Ideriv, bc, taus, Iinit, tol=0.01)

#Plot the new intensities from our numerical solution
for ix, tau in enumerate(taus):
	this_I = [np.interp(tau, res.x, res.y[i,:]) for i in np.arange(2*n)]
	plt.plot(theta_deg_g, this_I, 'C{:d}--'.format(ix))

#Compute the Js and Hs to examine the solution
ntau = res.y.shape[1]
Js = np.zeros(ntau)
Hs = np.zeros(ntau)
for i in range(ntau):
	Js[i] = 0.5*np.sum(res.y[:,i])/n
	Hs[i] = 0.5*np.sum(mu_g*res.y[:,i])/n
print('\n Maximum H departure from 1.0 [BVP solution]: {:.1e}\n'.format(np.max(np.abs(Hs-1))))

#Now also plot the q(tau), and print Tabld 17.3 in the text
plt.figure(2)
plt.clf()
qs = Js/3-res.x
plt.plot(res.x, qs)
plt.xlabel(r'$\tau$')
plt.ylabel(r'Hopf $q$ function')
plt.axis([0,2,.57,.71])
plt.tight_layout()
print('tau   q(tau)')
for tau in [0,.01,.02,.04,.06,.1,.2,.4,.6,1,1.5,2]:
	print("{:.2f} {:.5f}".format(tau, np.interp(tau, res.x, qs)))