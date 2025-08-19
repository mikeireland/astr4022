import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expn

def lambda_matrix(tau_grid):
	"""
	Compute the Lambda operator as a matrix based on a tau grid.
	
	Parameters
	----------
	tau_grid: (n_tau) numpy.array
		A grid of optical depth points, starting at 0 and monotonically increasing.
	
	Returns
	-------
	lambda_mat: (n_tau) numpy.array
		A matrix defined so that to obtain vector of mean intensities J from a vector 
		of source functions S we simply compute: J=numpy.dot(lambda_mat, S)
	"""
	#Fill our final result matrix with zeros
	lambda_mat = np.zeros( (len(tau_grid), len(tau_grid)) )
	
	#Create a delta-tau grid
	delta_tau = tau_grid[1:] - tau_grid[:-1]
	
	#For simplicity and readability, just go through one layer at a time.
	for j in range(len(tau_grid)):
		#Create E2 and E3 vectors
		E2_vect = expn(2,np.abs(tau_grid - tau_grid[j]))
		E3_vect = expn(3,np.abs(tau_grid - tau_grid[j]))
		
		#Add the contribution from the i'th layer, for upwards going rays
		lambda_mat[j,j:-1] +=  E2_vect[j:-1] - (E3_vect[j:-1] - E3_vect[j+1:])/delta_tau[j:] 
		
		#Add the contribution from the i+1'th layer, for upwards going rays
		lambda_mat[j,j+1:]  += -E2_vect[j+1:]  + (E3_vect[j:-1] - E3_vect[j+1:])/delta_tau[j:]
		
		#Add the contribution from the i'th layer, for downwards going rays
		lambda_mat[j,1:j+1] +=  E2_vect[1:j+1] - (E3_vect[1:j+1] - E3_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the i-1'th layer, for downwards going rays
		lambda_mat[j,:j]  += -E2_vect[:j]  + (E3_vect[1:j+1] - E3_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the lower boundary condition
		lambda_mat[j,-1] +=  E3_vect[-1]/delta_tau[-1] + E2_vect[-1]
		lambda_mat[j,-2] += -E3_vect[-1]/delta_tau[-1]
	return 0.5*lambda_mat

		