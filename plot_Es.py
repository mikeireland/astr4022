from scipy.special import expn
import numpy as np
import matplotlib.pyplot as plt
tau = np.linspace(.001,5,200)

plt.figure(1)
plt.clf()
for i in range(1,5):
	plt.plot(tau, expn(i, tau), label=r'E$_{:d}$'.format(i))
plt.plot(tau, np.exp(-tau), '--', label=r'exp(-$\tau$)')
plt.xlabel(r'$\tau$')
plt.ylabel(r'E$_n$')
plt.legend()
plt.axis([0,3,0,1.2])