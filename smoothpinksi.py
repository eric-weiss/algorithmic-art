import numpy as np
from matplotlib import pyplot as pp
from colorsys import hls_to_rgb

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    arg = arg - np.min(arg)
    arg = 2.0*np.pi*arg / (np.max(arg) - np.min(arg))

    # r=r-np.mean(r)
    # r=r-np.max(r)/2.

    # r=1.*r/np.sqrt(np.var(r))
    # arg=1.*arg/np.sqrt(np.var(arg))

    r=r-np.min(r)
    r=r/(np.max(r)-np.min(r))


    h = (arg + np.pi*0.)  / (2.0 * np.pi)
    l = (1.0 - 1.0/(1.0 + np.exp(-r)) )
    # l = (1.0/(1.0 + np.exp(-r)) )

    s = 1.0

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

def plotc(x, fname_save=None):
    if len(x.shape)==1:
        pp.plot(x.real)
        pp.plot(x.imag)
    else:
        pp.imshow(colorize(x), interpolation='Nearest')
        # pp.savefig('/home/swoop/Desktop/imtest.eps', format='eps', dpi=1000)

    if fname_save is not None:
        pp.xticks([])
        pp.yticks([])
        pp.savefig(fname_save, dpi=400)
    pp.show()





ndim=3*600   #dimensionality of complex vector
nt=ndim    #number of time steps to calculate
#
# ndim=1024   #dimensionality of complex vector
# nt=1024     #number of time steps to calculate


##   Initialization of the complex state vector    ##

Z=np.ones(ndim,dtype=complex)
# Z=np.arange(ndim,dtype=complex)*1.+0.0
# Z=-np.exp(2.j*Z*np.pi*np.random.rand(1))
Z=np.exp(2.j*Z*np.pi/2.)
# Z[0::3]*=1.j
# Z[1::3]*=-1.


Z[ndim/2]=-Z[ndim/2]

# Z*=np.exp(1.j*np.pi*(np.random.randint(low=0, high=4, size=ndim)/2.)*np.random.binomial(1,100./ndim,size=(ndim)))
# Z=np.exp(2.j*np.pi*(1.+Z/(float(ndim)))) + np.exp(1.j*np.pi*(np.random.randint(low=0, high=4, size=ndim)/2.))*np.random.binomial(1,10./ndim,size=(ndim))


Zhist=[]
Zhist.append(Z)
Zhist.append(Z)
########################################################


## Construction of the dynamics matrix M ##

#different types of random initializations
M=np.exp(np.random.rand(3,3)*2.j*np.pi)    #Initializes to totally random complex numbers on the unit circle
# M=np.exp(np.random.randint(0,8,size=(3,3))*1.j*np.pi/2.0)   #Initializes to a discrete set of values on the circle
M=np.dot(M,M.T.conjugate())
M=M*(np.random.rand(3,3)>.1)   ###### This zeros out some of the values (randomly)


# M = np.asarray([ [ -1.0, 1.0, 1.0 ],
#                  [  0.0, 1.0, -1.0],
#                  [  0.0, -1.0, 1.0] ], dtype=complex).T



####  The rest were discovered experimentally


  #energy sword
# [[  0.00000000e+00 +0.00000000e+00j,   0.00000000e+00 +0.00000000e+00j,
#     1.01000000e+00 +0.00000000e+00j],
#  [  0.00000000e+00 +0.00000000e+00j,   6.18446634e-17 +1.01000000e+00j,
#     0.00000000e+00 +0.00000000e+00j],
#  [ -1.01000000e+00 +1.23689327e-16j,  -0.00000000e+00 +0.00000000e+00j,
#     1.01000000e+00 +0.00000000e+00j]]

# M=np.asarray([[-1. +0.0j,  0. +0.0j,  0. +0.00000000e+00j],
#               [ 0. +0.0j,  1. +0.0j, -1. +1.22464680e-16j],
#               [ 1. +0.0j, -1. +0.0j,  1. +0.0j]], dtype=complex)
#
#
# M=1.0*np.asarray([[  1.00000000e+00+0.j,   1.00000000e+00+0.j,   0.00000000e+00+0.j],
#               [  6.12323400e-17+1.j,   1.00000000e+00+0.j,   6.12323400e-17+1.j],
#               [  0.00000000e+00+0.j,   1.00000000e+00+0.j,   6.12323400e-17+1.j]], dtype=complex)
#
#
#
M=1.*np.asarray([[  1.00000000e+00 +0.00000000e+00j,   0.00000000e+00 +0.00000000e+00j, -1.83697020e-16 -1.00000000e+00j],
                   [  6.12323400e-17 +1.00000000e+00j,  -1.00000000e+00 +1.22464680e-16j, -1.83697020e-16 -1.00000000e+00j],
                   [  6.12323400e-17 +1.00000000e+00j,   0.00000000e+00 +0.00000000e+00j, 1.00000000e+00 +0.00000000e+00j]], dtype=complex)
#
# M=1.*np.asarray([[  1.00000000e+00 +0.00000000e+00j,  -1.83697020e-16 -1.00000000e+00j,
#     1.00000000e+00 +0.00000000e+00j],
#  [  1.00000000e+00 +0.00000000e+00j,  -1.00000000e+00 +1.22464680e-16j,
#     0.00000000e+00 +0.00000000e+00j],
#  [ -0.00000000e+00 +0.00000000e+00j,   6.12323400e-17 +1.00000000e+00j,
#    -1.83697020e-16 -1.00000000e+00j]], dtype=complex)
#
# #
# M=1.*np.asarray([[  1.00000000e+00 +0.00000000e+00j,   6.12323400e-17 +1.00000000e+00j, -0.00000000e+00 +0.00000000e+00j],
#                  [  0.00000000e+00 +0.00000000e+00j,  -0.00000000e+00 +0.00000000e+00j, -1.00000000e+00 +1.22464680e-16j],
#                  [  1.00000000e+00 +0.00000000e+00j,  -1.83697020e-16 -1.00000000e+00j, 0.00000000e+00 -0.00000000e+00j]], dtype=complex)
#
# M=1.4*np.asarray([[  1.00000000e+00 +0.00000000e+00j,   6.12323400e-17 +1.00000000e+00j,
#     0.00000000e+00 +0.00000000e+00j],
#  [  0.00000000e+00 +0.00000000e+00j,   0.00000000e+00 +0.00000000e+00j,
#    -1.00000000e+00 +1.22464680e-16j],
#  [  1.00000000e+00 +0.00000000e+00j,  -1.83697020e-16 -1.00000000e+00j,
#     0.00000000e+00 +0.00000000e+00j]], dtype=complex)
#
# M=np.asarray([[  0.00000000e+00 +0.00000000e+00j,   0.00000000e+00 -0.00000000e+00j,
#    -9.99000000e-01 +0.00000000e+00j],
#  [ -0.00000000e+00 +0.00000000e+00j,   1.83513323e-16 +9.99000000e-01j,
#     0.000000001.j*e+00 +0.00000000e+00j],
#  [  9.99000000e-01 -1.22342215e-16j,  -0.00000000e+00 +0.00000000e+00j,
#    -9.99000000e-01 +0.00000000e+00j]], dtype=complex)
# M=np.asarray([[-0.-1.j,  0.+1.j, -0.-1.j],
#  [-1.+0.j,  0.+1.j,  0.+1.j],
#  [-1.+0.j, -1.+0.j,  1.+0.j]])
#################################################################################

# These last few lines are random things I thought to play around with

# Try commenting them in/out

#################################################################################


#a type of normalization of the columns of M
M=M/np.sqrt(np.sum(np.abs(M)**2,axis=0,keepdims=True))

#this value can be changed to values close but not equal to 1.0 - sometimes gives cool results

# M=np.asarray([[  1.01000000e+00+0.j,     0.00000000e+00+0.j,     6.18446634e-17+1.01j],
#  [  1.01000000e+00+0.j,     6.18446634e-17+1.01j,   1.01000000e+00+0.j  ],
#  [ -1.85533990e-16-1.01j,   6.18446634e-17+1.01j,   6.18446634e-17+1.01j]], dtype=complex)

#sometimes using the transpose of M can be interesting
# M=M.T+M
# M=M.conjugate()

print M
rate_adjust = 0.15
M=M*rate_adjust

## Main calculation loop
cvals=[]
# cvals.append(np.zeros((ndim,3)))
# cvals.append(np.ones((ndim,3)))
cvals.append(np.zeros((ndim)))
cvals.append(np.ones((ndim)))
for k in range(nt):
    #

    #The two lines below are just random experiments
    # Z_angles = np.angle(Z*Zhist[-2])
    # Z_angles = np.angle(Z)*np.sqrt(np.abs(Z))

    # This is the original rule - if using either of the above lines this should be commented out
    Z_angles = np.angle(Z)
    # Z_angles=Z

    #These next lines implement the dynamics and save the new vector in Zhist
    Z_angle_3stack = np.stack([np.roll(Z_angles, -1), Z_angles, np.roll(Z_angles, 1)])  #(3, ndim)
    # Znew = np.exp(np.sum(np.dot(M, Z_angle_3stack),axis=0))  #(3, ndim)
    u = np.exp(np.dot(M, Z_angle_3stack)*1.j)  #(3, ndim)
    alpha=1.0
    Znew = (np.sum(u, axis=0)**alpha)*Z/np.abs(Z)
    # Znew/= np.abs(Znew)
    # Znew=np.round(Znew,2)

    Znew_1_3=np.roll(Znew[::3],3)
    Znew_2_3=np.roll(Znew[1::3],-3)

    Znew[::3]=Znew_1_3
    Znew[1::3]=Znew_2_3
    Zhist.append(Znew)


    #If you don't want values to wrap around the edges of the vectors, then uncomment these two lines
    # Znew[-1]=np.exp(-1.j * 1. * np.pi)
    # Znew[0]=np.exp(-1.2j * 1. * np.pi)

    Z=Znew
    cvals.append(np.ones((ndim))*(k+2))
    # cvals.append(np.ones((ndim,3))*(k+2))


Zhist=np.asarray(Zhist,dtype=complex)
cvals=np.asarray(cvals,dtype=np.float32)


#####  PLOTTING

#plots the entire history
# plotc(Zhist[:,::3])
# plotc(Zhist[:,1::3])
# plotc(Zhist[:,2::3])
plotc(Zhist)

#plots every other value of the history matrix
# Zhist=Zhist[1:]*Zhist[:-1].conjugate()
# cvals=cvals[:-1]
# Zhist.real=-np.sqrt(np.abs(Zhist.real))*np.sign(Zhist.real)
# plotc(Zhist)
# plotc(Zhist.imag)
# Zf,vs,f=np.linalg.svd(np.dot(Zhist.T.conjugate(),Zhist))
# vs=1./np.sqrt(vs)
# plotc(np.dot(Zf,np.dot(np.diag(vs),np.dot(Zf,Zhist.T))))

# idxs=np.random.permutation(range(nt))
# cvals=cvals[idxs]
# Zhist=Zhist[idxs]
# cvals=cvals[:512]
# Zhist=Zhist[:1]

cvals=cvals/np.max(cvals)


# cvals[:,:,2] = 1.0 - cvals[:,:,2]
# cvals[:,:,3]=0.8
print np.max(cvals), np.min(cvals)
fig=pp.figure(figsize=(12,12))

# pp.scatter(Zhist[:].flatten().real, Zhist[:].flatten().imag, s=1*(1.1-(cvals**2)), c=(cvals**2), lw=0, alpha=0.6)
# pp.scatter(Zhist[:].flatten().real, Zhist[:].flatten().imag, s=1*(1.1-(cvals**1)), c=(cvals**2), lw=0, alpha=0.5)
# fig=pp.figure()
ax = fig.add_subplot(111, axisbg='black')
Zhist=Zhist[:nt/2]
cvals=cvals[:nt/2]
# cvals+=0.52
# ax.scatter(Zhist[:].flatten().imag, Zhist[:].flatten().real, s=600.0-599.0*cvals**3/np.max(cvals**3), c=(cvals**3.5), lw=0, alpha=0.5,cmap='terrain')
ax.scatter(Zhist[:].flatten().real, Zhist[:].flatten().imag, s=5.0+0.0*np.sin(cvals*np.pi), c=(cvals**6.0), lw=0, alpha=0.75,cmap='terrain')
# pp.xlim([1.0,5])
# pp.ylim([-2,2])
# pp.xlim([2.65,3.42])
# pp.ylim([-.65,.65])
pp.xticks([])
pp.yticks([])
# pp.show()
pp.savefig('/home/swoop/Desktop/scatternew.png', format='png', dpi=72*4, bbox_inches='tight', pad_inches=0)
# pp.scatter(Zhist[:].flatten().real, Zhist[:].flatten().imag, s=1, c=(cvals**2)/np.sqrt(float(ndim)), lw=0, alpha=0.5)
# pp.scatter(Zhist[:].flatten().real, Zhist[:].flatten().imag, s=10, c=cvals.reshape((nt+2)*ndim,3), alpha=0.8)
# pp.scatter(Zhist[:].flatten().real, Zhist[:].flatten().imag, s=2, alpha=0.2)
# pp.show()

# r=np.log(np.abs(Zhist[:])**2).flatten()
# phi=np.angle(Zhist[:]).flatten()
# pp.scatter(r*np.cos(phi), r*np.sin(phi), s=1, c=(cvals**1), lw=0, alpha=0.7)
# pp.scatter(r*np.cos(phi), r*np.sin(phi), s=1, c=r, lw=0, alpha=0.7)
# pp.show()


