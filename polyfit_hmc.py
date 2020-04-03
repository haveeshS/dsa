import numpy as np
import corner
import matplotlib.pyplot as plt
import emcee as em
import scipy.optimize as opt

np.random.seed(123)

# Choose the "true" parameters.
a_true = -0.959
b_true = 4.294
c_true = -2.174
d_true = 1.687
e_true = 0.247

# Generate some synthetic data from the model.
N = 69
x = np.sort(2 * np.random.rand(N))
yerr = 1 + np.random.rand(N)
y = np.polynomial.polynomial.polyval(x,(a_true,b_true,c_true,d_true,e_true))
y += np.abs(0.2 * y) * np.random.randn(N)
y += yerr * np.random.randn(N)

# print(yerr)

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
# x0 = np.linspace(0, 10, 500)
# plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

def lnprior((a,b,c,d,e)):
	if -5.0<a<5.0 and -5.0<b<5.0 and -5.0<c<7.0 and -5.0<d<8.0 and -5.0<e<5.0:
		ln1=0

	else:
		ln1=-np.inf

	# return ln1+(H0-H0p)**2/(2*(s0p**2))
	return ln1

def U(q):

	lp = lnprior(q)
	if not np.isfinite(lp):
		return -np.inf	
# 
	y_th = np.polynomial.polynomial.polyval(x,q)
	c=np.sum(((y-y_th)/yerr)**2)
	L = -0.5*(np.sum(np.log(2*np.pi*(yerr**2))))+c 
	return L+lp
	# return L

def K(p):

	return 0.5*np.sum(p**2)

def grad_U(q):

	h = 1e-3 

	dU = np.ones(q.size)

	for i in range(q.size):

		temp = np.array(q)
		temp[i] = q[i] + h 

		dU[i] = (U(temp)-U(q))/h

	return dU 

def grad_K(p):

	return p

N_s = 10000

N_h = 10

q_i = np.array([-0.5,3.0,-1.0,1.0,1.0])
# q_i = np.array([0,0,0,0,0])
# q_i = np.array([-0.9,2.1,-0.1,3.1,1.1])

q_c = q_i 

p_c = np.random.random_sample((q_c.size))

samples = []

# print(U(q_i),grad_U(q_i))

for j in range(N_s):

	samples.append(q_c)

	p_c = np.random.normal(loc=0,scale=1,size=q_c.size)

	# p_c = np.random.random_sample((q_c.size))

	p = p_c
	q = q_c

	# print(grad_U(q),grad_K(p))

	ep = 1e-2

	p = p - 0.5*ep*grad_U(q)

	for i in range(N_h):

		q = q + ep*grad_K(p)

		if i!=N_h-1:

			p = p - ep*grad_U(q)

	p = p - 0.5*ep*grad_U(q)	

	# p = -p

	# print(np.exp(-U(q)-K(p))/np.exp(-U(q_c)-K(p_c)))
	# print(q)

	# print(grad_U(q),grad_K(p))

	r = np.random.uniform()

	# print(np.exp(-U(q)-K(p)+U(q_c)+K(p_c)))

	if np.exp(-U(q)-K(p)+U(q_c)+K(p_c))>r:

		q_c = q 
		p_c = p 

samples = np.array(samples)

# print(samples)

a_hmc = np.mean(samples[:,0])
b_hmc = np.mean(samples[:,1])
c_hmc = np.mean(samples[:,2])
d_hmc = np.mean(samples[:,3])
e_hmc = np.mean(samples[:,4])

coeff = np.array([a_hmc,b_hmc,c_hmc,d_hmc,e_hmc])

print(coeff)

x_t = np.linspace(0,2)
y_t = np.polynomial.polynomial.polyval(x_t,coeff)
y_1 = np.polynomial.polynomial.polyval(x_t,(a_true,b_true,c_true,d_true,e_true))

plt.plot(x_t,y_t,color='b',label='The best fit model using HMC')
plt.plot(x_t,y_1,color='orange',label='The true model')
plt.legend()
plt.show()

fig = corner.corner(samples,labels=["$a$", "$b$", "$c$", "$d$", "$e$"])
# fig.savefig("triangle.png")
plt.show()

T = np.ones(N_s)

for i in range(N_s):

	T[i] = i 

# plt.plot(T,samples[:,0])
# plt.show()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, sharey=True)
ax1.plot(T,samples[:,0])
ax2.plot(T,samples[:,1])
ax3.plot(T,samples[:,2])
ax4.plot(T,samples[:,3])
ax5.plot(T,samples[:,4])

ax1.set_ylabel('a')
ax2.set_ylabel('b')
ax3.set_ylabel('c')
ax4.set_ylabel('d')
ax5.set_ylabel('e')
ax5.set_xlabel('Steps')
# f.set_xlabel('Steps')

plt.show()

