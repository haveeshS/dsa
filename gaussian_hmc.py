import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats
import seaborn as sns

N = 20000# Number of Samples 
m = 2 # Mean of Gaussian PDF
s = 2 # Variance of Normal PDF
x_i = 0

sns.set()

samples = []

def U(x):

	P = np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))
	return -np.log(P)

def K(p):

	return 0.5*p**2 

def grad_U(x):

	h = 1e-3 

	dU = (U(x+h)-U(x))/h 

	return dU

def grad_K(p):

	return p 

N_h = 10

x_c = x_i 	

for j in range(N): 

	samples.append(x_c)

	p_c = np.random.normal(loc=0.0,scale=1.0)

	p = p_c
	x = x_c 

	ep = 0.6

	p = p - 0.5*ep*grad_U(x)

	for i in range(N_h):

		x = x + ep*grad_K(p)

		if i!=N_h-1:

			p = p - ep*grad_U(x)

	p = p - 0.5*ep*grad_U(x)

	# print('x',x,U(x),grad_U(x))
	# print('p',p,K(p),grad_K(p))

	# p = -p

	r = np.random.uniform()

	if np.exp(-U(x)-K(p)+U(x_c)+K(p_c))>r:
		x_c = x 
		p_c = p 

samples = np.array(samples)

# print(samples)

print(np.mean(samples))

t = np.linspace(-6,10)

plt.plot(t,np.exp(-U(t)),label='The Target Distribution')
plt.xlabel('$x$')
plt.ylabel('$p(x)$')
plt.legend()
plt.hist(samples,50,density=True)
plt.title('Histogram of samples created by Hamiltonian Monte Carlo')
plt.show()