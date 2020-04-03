import numpy as np
import corner
import matplotlib.pyplot as plt

D = np.loadtxt('hznodups.dat', delimiter=' ')

z = D[:,0]
H = D[:,1]
s = D[:,2]

H0p = 68
s0p = 2.8

def lnprior((m,l,H0)):
	if 0.0<m<1.0 and 0.0<l<1.0 and 60.0<H0<80:
		ln1=0

	else:
		ln1=-np.inf

	# return ln1+(H0-H0p)**2/(2*(s0p**2))
	return ln1

		

def U((m,l,H0)):
	lp = lnprior((m,l,H0))
	if not np.isfinite(lp):
		return -np.inf	

	if m>0 and l>0:
		H_th = H0*((m*((1+z)**3)+l)**0.5)
		c=np.sum(((H-H_th)/s)**2)
		L = -0.5*(np.sum(np.log(2*np.pi*(s**2))))+c
		return L+lp

def K((p1,p2,p3)):
	return 0.5*(p1**2 + p2**2 + p3**2)

def grad_U((m,l,H0)):

	h = 1e-3

	Um = (U((m+h,l,H0))-U((m,l,H0)))/h
	Ul = (U((m,l+h,H0))-U((m,l,H0)))/h
	UH = (U((m,l,H0+h))-U((m,l,H0)))/h

	return np.array([Um,Ul,UH])

def grad_K((p1,p2,p3)):

	h = 1e-3

	K1 = (K((p1+h,p2,p3))-K((p1,p2,p3)))/h
	K2 = (K((p1,p2+h,p3))-K((p1,p2,p3)))/h
	K3 = (K((p1,p2,p3+h))-K((p1,p2,p3)))/h

	return np.array([K1,K2,K3])

q_i = np.array([0.5,0.5,70])

# print(U(q_i))
# print(grad_U(q_i))

p = np.random.random_sample((3,))

# print(K(p))
# print(grad_K(p))

N_s = 20000

N_h = 10

q_c = q_i 
p_c = np.random.random_sample((q_c.size))


samples = []

print('Initial U',U(q_c))

for i in range(N_s):

	samples.append(q_c)

	p_c = np.random.normal(loc=0.0,scale=1.0,size=q_c.size)

	p = p_c
	q = q_c

	# print(grad_U(q),grad_K(p))

	ep = 1e-3

	p = p - 0.5*ep*grad_U(q)

	for i in range(N_h):

		q = q + ep*grad_K(p)

		if i!=N_h-1:

			p = p - ep*grad_U(q)

	p = p - 0.5*ep*grad_U(q)	

	# p = -p

	# print(q)
	# print(p)

	r = np.random.uniform()

	# print(np.exp(-U(q)-K(p))/np.exp(-U(q_c)-K(p_c)),r)

	if np.exp(-U(q)-K(p))/np.exp(-U(q_c)-K(p_c))>r:

		q_c = q 
		p_c = p 

samples = np.array(samples)


# plt.hist(samples[:,0],50)
# plt.show()

def chisq((m,l,H0)):
	Ht = H0*np.sqrt(m*((1+z)**3)+l)
	c=np.sum(((H-Ht)/s)**2)
	return c

# print(samples)

smples_c = np.delete(samples,(0),axis=0)
smples_c = np.delete(smples_c,(1),axis=0)

# print(smples_c)

# print(np.mean(s[:,0]),np.mean(s[:,1]),np.mean(s[:,2]))

om = np.mean(smples_c[:,0])
ol = np.mean(smples_c[:,1])
oh = np.mean(smples_c[:,2])

print(om,ol,oh)
print(chisq((om,ol,oh)))

fig = corner.corner(smples_c,labels=["$\\Omega_m$", "$\\Omega_{\\Lambda}$", "$H_0$"])
fig.savefig("triangle.png")
plt.show()

m_hmc, l_hmc, H0_hmc = map(lambda v: (v[1],v[1]-v[0],v[2]-v[1]),zip(*np.percentile(samples,[16,50,84],axis=0)))

print(m_hmc)
print(l_hmc)
print(H0_hmc)

# plt.hist(smples_c[:,0],50)
# plt.show()

# plt.hist(smples_c[:,1],50)
# plt.show()

# plt.hist(smples_c[:,2],50)
# plt.show()