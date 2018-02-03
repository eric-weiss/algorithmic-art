import numpy as np
from matplotlib import pyplot as pp
import theano
import theano.tensor as T
import scipy
import numpy as np
from matplotlib import pyplot as pp
from colorsys import hls_to_rgb
import sys
sys.path.append('../sequence_encoding/')

from scipy import signal as spsg

def colorize(z, smap=None):
    r = np.absolute(z)
    # r = np.log(r+1.)
    arg = np.angle(z)
    arg/=2.0*np.pi
    h = arg
    # r=r-np.max(r)/2
    # r-=np.min(r)
    r/=(np.max(r))
    # l=r*0.6+0.1
    l=0.5
    # l=1.0*r

    if smap is None:
        s=0.9
        # s=1.0*r+0.
    else:
        s=smap.real
        s=s-np.min(s)
        s=s/(np.max(s)-np.min(s))


    c = np.vectorize(hls_to_rgb) (h, l, s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

def plotc(x, smap=None):
    if len(x.shape)==1:
        pp.plot(x.real,linewidth=4)
        pp.plot(x.imag,linewidth=4)
    else:
        pp.imshow(colorize(x, smap), interpolation='Nearest')
    pp.show()


# class quaternion():
#
#     def __init__(self, rvals=None, ivals=None):
#         if not rvals:
#             rvals_init=np.ones()

def qu_mult(a, b):
    c_0 = a[0]*b[0]-a[1]*(b[1].conjugate())
    c_1 = a[1]*(b[0].conjugate())+b[1]*a[0]
    return np.asarray([c_0, c_1])

def quantize_phase(x_in, n_angles):
    norms=np.absolute(x_in)
    norms=np.round(norms*2.0,0)/2.0
    if n_angles>0:
        angs = (np.angle(x_in)) * float(n_angles) / (2.0*np.pi)
        xout = np.exp(2.j * (np.pi * np.round(angs,0) *0.125 / float(n_angles) ))
    return xout#*norms


def quantize_hurwitz(x_in):
    return np.ceil(x_in.real*2.0)/2.0+1.j*np.ceil(x_in.imag*2.0)/2.0

def qu_conj(x_in):
    return np.asarray([x_in[0].conjugate(), -x_in[1]])


def normalize(x_in):
    norms = np.sqrt(np.sum(x_in*x_in.conjugate(), axis=0, keepdims=True).real)
    return x_in/norms

# plotc(np.round(2.0*np.exp(1.j*np.linspace(-2.*np.pi,2.*np.pi,1024)),0))



nx=5**3

nstates=2

nt=256

shifts=[1,2,3]

rad=nx/1

x=np.ones((2,nx),dtype=complex)*1.
# x[0,1]=1.j
# x[1,:]=0.
# x[0,0]=0.5-0.5j
# x[1,0]=-0.5+0.5j
# print x[:,0]
# for i in range(24):
#     x[:,0]=qu_mult(x[:,0], qu_conj(x[:,1]))
#     print qu_mult(qu_conj(x[:, 1]), x[:, 0])
#     print x[:,0]

# exit()
# x[0,800]=0.5
# x[1,800]=0.5j
# x[0,:nx/2]=0.
# x[1,:nx/2]=1.
# x[:,nx/2-4:nx/2+4]=-0.5-0.5j
# x[:,nx/4-4:nx/4+4]=-0.5-0.5j


# print qu_mult(x[:,2],qu_conj(x[:,0]))
# exit()

# x[0,:]=np.exp(1.j*(np.pi*np.random.randint(0,4,(nx)))/2.0)
# x[0,0:rad]*=np.exp(1.j*(np.pi*np.arange(rad)/float(rad)))
# x[1,0:rad]=np.exp(11.j*(np.pi*np.arange(rad)/float(rad)))
# x[0,0:rad]=np.exp(13.j*(np.pi*np.arange(rad)/float(rad)))
# x[1,:]=0.1
# x[0,:]=0.
# x[1,:]=1.
x[1,nx/2]=np.exp(1.j*np.pi/3.0)
# x[0,nx/2]=-np.exp(1.j*np.pi/1.0)
# x[1,nx/2]=np.exp(-2.j*np.pi/6.0)
# x[1,nx/2]=np.exp(1.j*np.pi/4.0)
# x[1,nx/2]-=np.exp(1.j*np.pi/8.0+0.j*np.pi/2.0)
# x[0,0]=1.j
# x[1,0]=-1.j
# x=normalize(x)
x=quantize_phase(x,4)
x = normalize(x)
print x
xhist=[]

# norm0=np.sqrt(np.sum(np.sum(np.abs(x)**2)))
# x/=norm0
for t in range(nt):
    xhist.append(x*1.0)

    dots=np.roll(x, 1, axis=1)*np.roll(x, -1, axis=1)*np.roll(x, 1, axis=0)
    dots=quantize_phase(dots,124)
    # dots = normalize(dots)
    # xnew=qu_mult(x,qu_conj(dots))
    xnew=qu_mult(dots, qu_mult(x,qu_conj(dots)))
    # trig=np.exp(-1.0*(dots[0]*dots[1].conjugate()).real**2+1.2j*(x[0]*dots[1].conjugate()).imag**1)
    # xnew = 0.02**np.log(dots+1.0)*(1.0-trig)+x*trig
    # dots=quantize_phase(dots)
    # dots[0] *= 1e-0
    # xnew = quantize_phase(xnew,4)
    xnew = normalize(xnew)

    # dots[0,:]=1.0
    # dots = normalize(dots)
    # norms = np.sqrt(np.sum(dots * qu_conj(dots), axis=0))
    # xnew=qu_mult(dots, qu_mult(x, qu_conj(dots)))
    # xnew=qu_mult(qu_conj(dots), xnew)
    # x = normalize(xnew)
    # xnew/=np.sqrt(np.sum(np.sum(np.abs(xnew) ** 2)))/norm0
    x=xnew*1.0
    # x = normalize(xnew)
    # x = quantize_hurwitz(x)
    # print xnew

    # x/=np.sqrt(np.sum(np.abs(x)**2))/norm0
    # x = quantize_phase(x, nstates)
    # x=np.roll(x,1,axis=1)
    # plotc(xnew)
    # x=1.0*xnew

xhist.append(x*1.0)
xhist=np.asarray(xhist)
# xhist=xhist[:,:,::2]
plotc((np.angle(xhist[:,1].T)+1.j*np.angle(xhist[:,0].T)).T)
# pp.matshow((np.angle(xhist[:,0].T))); pp.show()
plotc(xhist[:,1].T)
plotc(xhist[:,0].T)
# plotc(xhist[:,1].real.T)
# plotc(xhist[:,1].imag.T)
diffs=xhist[1:,0]-xhist[:-1,0]
# plotc((np.sqrt(diffs*diffs.conjugate())).T)
# plotc( diffs[::1].T**2)
# plotc( xhist[:,0,:].T)
# plotc( xhist[:,1,:].T)
# plotc( xhist[:,0].T+xhist[:,1].T)
# plotc( xhist[:,0].T*np.exp(2.j*np.random.rand()+np.random.randn())+xhist[:,1].T*np.exp(2.j*np.random.rand()+np.random.randn()))
# plotc( np.fft.fft(xhist[:,1].T,axis=0))
# plotc( np.fft.fft(xhist[:,0].T,axis=0))
# plotc( np.fft.fft(xhist[:,1].T+xhist[:,0].T,axis=0))
# plotc( prods.T)
# plotc(xhist[:,1].T)