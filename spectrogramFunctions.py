import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import cmocean
import seaborn as sns
import scipy.stats as spst
import sys
from hermite_functions import hermite_functions
import matplotlib.path as mpltPath
from math import atan2
from scipy.spatial import KDTree
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.pyplot import figure
from pandas import DataFrame
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error


# #testing
# np.random.seed(12) # to reproduce paper results
# base = 128 # controls the length of the simulation
#
# # signal parameters
# duration = (base - 20)
#
#
# # simulate
def get_area(points):
    #function that calculates the are of a triangle
    x = points[:, 0]
    y = points[:, 1]

    area = 0.5 * ((x[0] * (y[1] - y[2])) + (x[1] * (y[2] - y[0])) + (x[2] * (y[0] - y[1])))
    return area

def maxTriDistance(points):
    #This function calculates the lenght triangle side given the vertices coordinates of the tringle
    dista = np.linalg.norm(points[0] - points[1])
    distb = np.linalg.norm(points[1] - points[2])
    distc = np.linalg.norm(points[2] - points[0])

    maxSide = max(dista, distb, distc)
    return maxSide

def extr2minth(M,th=1e16):
    #This function calculates the local minima of 2 dimensional array
    #Th is used as a threshold. The function will return only local minima less than threshold th

    C,R = M.shape

    Mid_Mid = np.zeros((C,R), dtype=bool)

    for c in range(1, C-1):
        for r in range(1, R-1):
            T = M[c-1:c+2,r-1:r+2]
            Mid_Mid[c, r] = (np.min(T) == T[1, 1]) * (np.min(T) < th)
            #Mid_Mid[c, r] = (np.min(T) == T[1, 1])

    x, y = np.where(Mid_Mid)
    return x, y

def extr2maxth(M,th=-1e16):
    # This function calculates the local maxima of 2 dimensional array
    # Th is used as a threshold. The function will return only local maxima greater than threshold th

    C,R = M.shape

    Mid_Mid = np.zeros((C,R), dtype=bool)

    for c in range(1, C-1):
        for r in range(1, R-1):
            T = M[c-1:c+2,r-1:r+2]
            Mid_Mid[c, r] = (np.max(T) == T[1, 1]) * (np.min(T) > th)
            #Mid_Mid[c, r] = (np.max(T) == T[1, 1])

    x, y = np.where(Mid_Mid)
    return x, y

def whiteNoiseSpectrogram(base, mode=complex):

    #function that caclulates and plots the zeros of white noise spectrogram made by :https://github.com/jflamant/2018-zeros-spectrogram-white-noise

    N = 4 * base
    Nfft = 2 * base
    t = np.arange(N)

    if mode == 'complex':
        w = np.random.randn(N) + 1j * np.random.randn(N)
    elif mode == 'real':
        w = np.random.randn(N)
    else:
        raise ValueError('mode should be either real or complex')
    g = sg.gaussian(Nfft, np.sqrt((Nfft) / 2 / np.pi))
    g = g / g.sum()
    _, _, stft = sg.stft(w, window=g, nperseg=Nfft, noverlap=Nfft - 1, return_onesided=False)

    Sww_t = np.abs(stft) ** 2
    print("STFT computed")

    tmin = base
    tmax = 3 * base

    Sww = Sww_t[:, tmin:tmax + 1]

    # detection
    th = 1e-14
    y, x = extr2minth(Sww, th)

    u = (np.array(x)) / np.sqrt(2 * base)
    v = (np.array(y)) / np.sqrt(2 * base)

    pos = np.zeros((len(x), 2))
    pos[:, 0] = u
    pos[:, 1] = v

    return pos, [Sww, x, y]

def spectrogramSignalImpulseAndChirp(SNR, duration, viz=False, shrink=True):
    base = 128
    b = 150
    a = base - b


    N = 4 * base
    Nfft = 2 * base
    t = np.arange(N)

    # Noise only
    w = np.random.randn(N)
    # window
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()

    # bounds for detection (larger)
    fmin = 0
    fmax = base

    #tmin = 2*base - (base-trunc) // 2
    #tmax = 2*base + (base-trunc) // 2
    tmin = base
    tmax = 3*base

    #chirp
    duration = int(np.floor(duration))
    if duration > base:
        raise ValueError('Duration should be lesser than base')


    start_s = 2*base-duration //2 +20
    end_s = 2*base+duration //2  +20
    chirp = np.zeros(N)

    impulse = np.zeros(512)
    impulse[200:201] = 7 * sg.tukey(201 - 200)

    freq = (a + b*t[start_s:end_s]/N)* t[start_s:end_s]/N
    chirp[start_s:end_s] =  sg.tukey(duration)*np.cos(2*np.pi*freq)
    chirp[200:201]=impulse[200:201]

    x0 = np.sqrt(2*SNR)*chirp + w

    #spectro = STFT(x0, g, Nfft)
    f, ti, spectro = sg.stft(x0, window=g, nperseg=Nfft, noverlap=Nfft-1)
    Sww_t = abs(spectro)**2
    #print("STFT computed")


    Sww = Sww_t[fmin:fmax+1, tmin:tmax+1]


    # detection
    y0, x0 = extr2minth(Sww)
    if shrink is True:
        # boundary conditions
        side = 110 # size of square; equivalent to trunc
        fmin_b = (max(0, (base-side)//2))
        fmax_b = (min(base, (base+side)//2))
        tmin_b = (base-side//2)
        tmax_b = (base+side//2)

        mask = (y0 > fmin_b)*(y0 < fmax_b)*(x0 > tmin_b)*(x0 < tmax_b)
        u = x0[mask]/np.sqrt(2*base)
        v = y0[mask]/np.sqrt(2*base)
    else:
        u = x0/np.sqrt(2*base)
        v = y0/np.sqrt(2*base)

    pos = np.zeros((len(u), 2))
    pos[:, 0] = u
    pos[:, 1] = v

    if viz is True:
        side = 110 # size of square; equivalent to trunc
        fmin = (max(0, (base-side)//2))/np.sqrt(2*base)
        fmax = (min(base, (base+side)//2))/np.sqrt(2*base)
        tmin = (base-side//2)/np.sqrt(2*base)
        tmax = (base+side//2)/np.sqrt(2*base)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.imshow(np.log10(Sww), origin='lower', extent=[0, (2*base)/np.sqrt(2*base), 0, (base)/np.sqrt(2*base)], cmap=cmocean.cm.deep)
        ax.scatter(pos[:, 0], pos[:, 1], color='w', s=40)

        ax.set_xlim([tmin, tmax])
        ax.set_ylim([fmin, fmax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.subplots_adjust(left=0.04, bottom=0.05)

        return Sww, pos, spectro, chirp , f, ti

    else:
        return pos

def spectrogramHermiteSignal(SNR, duration, viz=False, shrink=True):
    base = 128
    b = 150
    a = base - b

    N = 4 * base
    Nfft = 2 * base
    t = np.arange(N)

    # Noise only
    w = np.random.randn(N)
    # window
    g = sg.gaussian(Nfft, np.sqrt((Nfft) / 2 / np.pi))
    g = g / g.sum()

    # bounds for detection (larger)
    fmin = 0
    fmax = base

    # tmin = 2*base - (base-trunc) // 2
    # tmax = 2*base + (base-trunc) // 2
    tmin = base
    tmax = 3 * base

    # chirp
    duration = int(np.floor(duration))
    if duration > base:
        raise ValueError('Duration should be lesser than base')

    start_s = 2 * base - duration // 2
    end_s = 2 * base + duration//2 

    hermite = np.zeros(N)
    x=np.arange(-10,10,0.2)

    hermF = hermite_functions(100, x, all_n=False)





    hermite[start_s:end_s] = sg.tukey(duration) * hermF
    hermite=hermite*1/np.max(np.abs(hermite))

    x0 = np.sqrt(2 * SNR) * hermite + w
    # spectro = STFT(x0, g, Nfft)
    _, _, spectro = sg.stft(x0, window=g, nperseg=Nfft, noverlap=Nfft - 1)
    Sww_t = abs(spectro) ** 2
    # print("STFT computed")

    Sww = Sww_t[fmin:fmax + 1, tmin:tmax + 1]

    # detection
    th = 1e-14
    y0, x0 = extr2minth(Sww, th)
    if shrink is True:
        # boundary conditions
        side = 110  # size of square; equivalent to trunc
        fmin_b = (max(0, (base - side) // 2))
        fmax_b = (min(base, (base + side) // 2))
        tmin_b = (base - side // 2)
        tmax_b = (base + side // 2)

        mask = (y0 > fmin_b) * (y0 < fmax_b) * (x0 > tmin_b) * (x0 < tmax_b)
        u = x0[mask] / np.sqrt(2 * base)
        v = y0[mask] / np.sqrt(2 * base)
    else:
        u = x0 / np.sqrt(2 * base)
        v = y0 / np.sqrt(2 * base)

    pos = np.zeros((len(u), 2))
    pos[:, 0] = u
    pos[:, 1] = v

    if viz is True:
        side = 110  # size of square; equivalent to trunc
        fmin = (max(0, (base - side) // 2)) / np.sqrt(2 * base)
        fmax = (min(base, (base + side) // 2)) / np.sqrt(2 * base)
        tmin = (base - side // 2) / np.sqrt(2 * base)
        tmax = (base + side // 2) / np.sqrt(2 * base)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.imshow(np.log10(Sww), origin='lower',
                  extent=[0, (2 * base) / np.sqrt(2 * base), 0, (base) / np.sqrt(2 * base)], cmap=cmocean.cm.deep)
        ax.scatter(pos[:, 0], pos[:, 1], color='w', s=40)

        ax.set_xlim([tmin, tmax])
        ax.set_ylim([fmin, fmax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.subplots_adjust(left=0.04, bottom=0.05)

        return Sww, pos, spectro, hermite

    else:
        return pos

def findDelauneyMask(Sww, pos_exp, th_side ,th_area=0.4 , base=128, viz= False, triDist=False):

    #This is a funtion that is finding mask based on the zeros of the spectrogram
    #It use the algorithm that Patrick Flandrin propose on his paper "Time–Frequency Filtering Based on Spectrogram Zeros"
    #First, the function calculates the delauney triangulation of the spertrcogram zeros
    #Then, the function keeps only the triangles with sides greater than 'side_th' and area greater than 'th_area'
    #The function group the keeped triqngles in a binary mask end returns it


    # define a deleauney triangulation with zeros
    tri=Delaunay(pos_exp)
    # define a grid corresponding to the time-frequency paving
    vecx = (np.arange(0, Sww.shape[0]) / np.sqrt(2 * base))
    vecy = (np.arange(0, Sww.shape[1]) / np.sqrt(2 * base))
    g = np.transpose(np.meshgrid(vecy, vecx))

    #result = kdpos.query_ball_point(g, radi_seg).T
    empty = np.zeros(len(vecx) * len(vecy), dtype='bool')
    g = g.reshape(len(vecx) * len(vecy), 2)
    triangleLength = np.zeros(len(tri.simplices))
    bigTriangles = np.zeros(len(tri.simplices), dtype='bool')


    for i in range(len(tri.simplices)):
        maxSide = maxTriDistance(pos_exp[tri.simplices[i]])
        area= get_area(pos_exp[tri.simplices[i]])
        triangleLength[i]=maxSide
        if maxSide > th_side and area>th_area :
            bigTriangles[i] = 'True'

            path = mpltPath.Path(pos_exp[tri.simplices[i]])
            # inside2 = path.contains_points(grid[0,:,:])
            empty = empty | path.contains_points(g)

    empty = empty.reshape(len(vecy), len(vecx)).T
    #empty = np.zeros(result.shape, dtype=bool)
    #for i in range(len(vecx)):
     #   for j in range(len(vecy)):
      #      empty[i,j] = len(result[i, j]) < 1


    # then plot
    if viz is True:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(g[..., 0], g[..., 1], s=0.1, color='w')
        ax.imshow(empty, extent=[0, (2*base)/np.sqrt(2*base), 0, (base)/np.sqrt(2*base)], origin='lower', cmap=cmocean.cm.deep_r)
        ax.scatter(pos_exp[:,0], pos_exp[:, 1], color='w', s=30)

        side = 110*(base/128)
        fmin = (max(0, (base-side)//2))/np.sqrt(2*base)
        fmax = (min(base, (base+side)//2))/np.sqrt(2*base)
        tmin = (base-side//2)/np.sqrt(2*base)
        tmax = (base+side//2)/np.sqrt(2*base)

        ax.set_xlim([tmin, tmax])
        ax.set_ylim([fmin, fmax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        fig.subplots_adjust(left=0.04, bottom=0.05)
        points=pos_exp
        plt.triplot(points[:, 0], points[:, 1], tri.simplices)
        plt.plot(points[:, 0], points[:, 1], 'o')
        plt.show()

    if triDist is True:
        return empty,triangleLength
    else:
        return empty

def findEnergyThresholdMask(Sww,th,base=128, viz=False):

    # This is a function that uses the traditional way of masking in time-frequency plane.
    # The function keeps only the values of spectrogram that are greater than an energy threshold “th”

    # define a grid corresponding to the time-frequency paving
    vecx = (np.arange(0, Sww.shape[0]) / np.sqrt(2 * base))
    vecy = (np.arange(0, Sww.shape[1]) / np.sqrt(2 * base))
    g = np.transpose(np.meshgrid(vecy, vecx))

    empty = np.zeros(Sww.shape, dtype='bool')

    for i in range(empty.shape[0]):
        for j in range(empty.shape[1]):
            if Sww[i,j]>th:
                empty[i,j]=True

    # then plot
    if viz is True:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(g[..., 0], g[..., 1], s=0.1, color='w')
        ax.imshow(empty, extent=[0, (2 * base) / np.sqrt(2 * base), 0, (base) / np.sqrt(2 * base)], origin='lower',
                  cmap=cmocean.cm.deep_r)

        side = 110 * (base / 128)
        fmin = (max(0, (base - side) // 2)) / np.sqrt(2 * base)
        fmax = (min(base, (base + side) // 2)) / np.sqrt(2 * base)
        tmin = (base - side // 2) / np.sqrt(2 * base)
        tmax = (base + side // 2) / np.sqrt(2 * base)

        ax.set_xlim([tmin, tmax])
        ax.set_ylim([fmin, fmax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        fig.subplots_adjust(left=0.04, bottom=0.05)

        plt.show()
    return empty

def findDelauneyMaskPlusVoronoiTessellation(Sww, pos_exp, radi_seg,pos_vor,base=128):
    # This function is using the same method as “findDelaunayMask” to calculate the mask based on a Delauney triangulation
    # Also, the function calculates the Voronoi Tesselation of the spectral maxima

    # define a Delauney triangulation with zeros
    tri=Delaunay(pos_exp)
    # define a Voronoi tessellation with maximus
    vor = Voronoi(pos_vor)

    # define a grid corresponding to the time-frequency paving
    vecx = (np.arange(0, Sww.shape[0]) / np.sqrt(2 * base))
    vecy = (np.arange(0, Sww.shape[1]) / np.sqrt(2 * base))
    g = np.transpose(np.meshgrid(vecy, vecx))

    #result = kdpos.query_ball_point(g, radi_seg).T
    empty = np.zeros(len(vecx) * len(vecy), dtype='bool')
    g = g.reshape(len(vecx) * len(vecy), 2)
    bigTriangles = np.zeros(len(tri.simplices), dtype='bool')
    th = radi_seg
    for i in range(len(tri.simplices)):
        maxSide = maxTriDistance(pos_exp[tri.simplices[i]])
        if maxSide > th:
            bigTriangles[i] = 'True'
            path = mpltPath.Path(pos_exp[tri.simplices[i]])
            # inside2 = path.contains_points(grid[0,:,:])
            empty = empty | path.contains_points(g)

    empty = empty.reshape(len(vecy), len(vecx)).T
    #empty = np.zeros(result.shape, dtype=bool)
    #for i in range(len(vecx)):
     #   for j in range(len(vecy)):
      #      empty[i,j] = len(result[i, j]) < 1


    # then plot
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(g[..., 0], g[..., 1], s=0.1, color='w')
    ax.imshow(empty, extent=[0, (2*base)/np.sqrt(2*base), 0, (base)/np.sqrt(2*base)], origin='lower', cmap=cmocean.cm.deep_r)
    ax.scatter(pos_exp[:,0], pos_exp[:, 1], color='w', s=30)

    side = 110 * (base/128)
    fmin = (max(0, (base-side)//2))/np.sqrt(2*base)
    fmax = (min(base, (base+side)//2))/np.sqrt(2*base)
    tmin = (base-side//2)/np.sqrt(2*base)
    tmax = (base+side//2)/np.sqrt(2*base)

    ax.set_xlim([tmin, tmax])
    ax.set_ylim([fmin, fmax])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(left=0.04, bottom=0.05)
    points=pos_exp
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    voronoi_plot_2d(vor,ax)

    plt.show()

    return empty

def reconstructionSignal(sub_mask, stft, chirp, base=128, viz=False, fake_SNR=1, real_Signal=False):
    # this function is computing the istft given the stft of a signal and a binary mask
    # it can be also used to visualize the results

    if real_Signal is True:
        amp=np.sqrt(2*fake_SNR)
    else:
        amp=1

    scale=base/128
    side = 110*scale
    side=int(np.floor(side))
    fmin = max(0, (base-side)//2)
    fmax = min(base, (base+side)//2)
    tmin = base-side//2
    tmax = base+side//2
    # sub mask : check which points are in the convex hull
    vec1 = (np.arange(0, side))
    vec2 = (np.arange(0, tmax-tmin))
    g = np.transpose(np.meshgrid(vec1, vec2))


    # create a mask
    sub_mask = sub_mask[fmin:fmax, tmin:tmax]
    mask = np.zeros(stft.shape, dtype=bool)
    mask[fmin:fmax, base+tmin:base+tmax] = sub_mask
    
    # reconstruction
    Nfft = 2*base # as in the simulation
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()
    t, xorigin = sg.istft(stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)
    t, xr = sg.istft(mask*stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)

    if viz is True:
        xorigin = xorigin * amp
        xr = xr * amp
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(t, scale*xorigin+15*scale)
        ax.plot(t, scale*chirp, color='k')
        ax.plot(t, scale*xr-15*scale, color='g')

        ax.set_xlim(162*scale, 350*scale)
        ax.set_ylim(-25*scale, 25*scale)
        sns.despine(offset=10)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ax.text((350+162)*scale/2, -26.5*scale,r'$\mathsf{time}$', fontsize=24, ha='center', backgroundcolor='white')
        ax.text(158.5*scale, 0*scale,r'$\mathsf{amplitude}$', fontsize=24, ha='center', va='center', rotation=90, backgroundcolor='white')

        ax.text(350*scale, 19*scale, r'$\mathsf{noisy\: signal}$', color='b', fontsize=20, ha='right', backgroundcolor='white' )
        ax.text(350*scale, 2*scale, r'$\mathsf{original}$', color='k', fontsize=20, ha='right', backgroundcolor='white' )
        ax.text(350*scale, -13*scale, r'$\mathsf{reconstructed}$', color='g', fontsize=20, ha='right', backgroundcolor='white' )
        #ax.legend(fontsize=18, loc=5)
        fig.tight_layout()
        fig.subplots_adjust(left=0.06, bottom=0.06)

    return xr, xorigin, chirp

def show_clusters(X,cluster):
    #this functions is used to visualize the clusters of a DBSCAN clustering

    df= DataFrame(dict(x=X[:,0],y=X[:,1], label=cluster))
    colors= {-1: 'red',0:'blue', 1:'orange',2:'green', 3:'skyblue',4:'yellow',5:'purple',6:'pink',7:'black',8:'brown',9:'grey',10:'coral',11:'teal',12:'khaki',13:'violet',14:'springgreen',15:'fuchsia',16:'lightsteelblue',17:'tan',18:'tomato',19:'palegreen',20:'midnightblue',21: 'burlywood',22:'peru',23:'maroon',24:'rosybrown', 25:'aqua',26:'orchid',27:'darkolivegreen',28:'turquoise',29:'navajowhite',30:'crimson'}
    fig, ax = plt.subplots(figsize=(8,8))
    grouped = df.groupby('label')


    for key, group in grouped:
        group.plot(ax=ax, kind='scatter',x='x',y='y', label=key, color= colors[key])
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.show()

def clusterMask(cluster, maskLabel, X, delauneyMaskShape):
    # function that creates a new mask for the prefered cluster
    new_mask = np.zeros(delauneyMaskShape, dtype='bool').T
    preferedCluster = X[cluster == maskLabel]
    preferedCluster = tuple(map(tuple, preferedCluster))
    for i in range(len(preferedCluster)):
        new_mask[preferedCluster[i]] = True

    return new_mask.T

def fromMaskToX(delauneyMask):
    # this fucntion transform a delauney Mask to a proper form for DBSCAN algorithm
    X = np.where(delauneyMask.T)
    X = np.asarray(X).T
    return X

def spectrogramSignal(SNR, signal,std=1,zeros_th=1e16, viz=False, shrink=True, voronoi=False):

    # This function takes a synthetic signal “signal” and add noise to it.
    # The function returns its spectrogram and signals stft transformation.
    # The function can be also used to visualize the zeros of the spectrogram on time-frequency plane.
    snr= np.power(10, SNR/10)
    duration= len(signal)
    #signal normalization
    signal=signal- np.mean(signal)
    # signal calibration
    signal=np.array(signal)
    c=1/(np.max(np.abs(signal)))
    signal=signal*c

    base = 128
    side=110
    while base<duration:
        base=base*2
        side=side*2

    scale=int(base/128)
    N = 4 * base
    Nfft = 2 * base
    t = np.arange(N)

    # Noise only
    w = np.random.normal(0, std, N)
    #w = np.random.randn(N)
    # window
    g = sg.gaussian(Nfft, np.sqrt((Nfft) / 2 / np.pi))
    g = g / g.sum()

    # bounds for detection (larger)
    fmin = 0
    fmax = base

    tmin = base
    tmax = 3 * base


    # chirp
    duration = int(np.floor(duration))
    if duration > base:
        raise ValueError('Duration should be lesser than base')

    start_s = 2 * base - duration // 2
    end_s = 2 * base + duration // 2
    if duration % 2 == 1:
        end_s=end_s+1

    chirp = np.zeros(N)


    chirp[start_s:end_s] = sg.tukey(duration) * signal

    x0 = np.sqrt(2 * snr) * chirp + w
    # spectro = STFT(x0, g, Nfft)
    _, _, spectro = sg.stft(x0, window=g, nperseg=Nfft, noverlap=Nfft - 1)
    Sww_t = abs(spectro) ** 2
    # print("STFT computed")

    Sww = Sww_t[fmin:fmax + 1, tmin:tmax + 1]

    # detection
    y0, x0 = extr2minth(Sww, zeros_th)

    # Voronoi
    if voronoi is True:
        y1, x1 = extr2maxth(Sww)
        pos_vor0=[y1, x1]

    if shrink is True:
        # boundary conditions
        side = 110*scale  # size of square; equivalent to trunc
        fmin_b = (max(0, (base - side) // 2))
        fmax_b = (min(base, (base + side) // 2))
        tmin_b = (base - side // 2)
        tmax_b = (base + side // 2)

        mask = (y0 > fmin_b) * (y0 < fmax_b) * (x0 > tmin_b) * (x0 < tmax_b)
        u = x0[mask] / np.sqrt(2 * base)
        v = y0[mask] / np.sqrt(2 * base)

        if voronoi is True:
            u1 = x1[mask] / np.sqrt(2 * base)
            v1 = y1[mask] / np.sqrt(2 * base)

    else:
        u = x0 / np.sqrt(2 * base)
        v = y0 / np.sqrt(2 * base)
        if voronoi is True:
            u1 = x1 / np.sqrt(2 * base)
            v1 = y1 / np.sqrt(2 * base)

    pos = np.zeros((len(u), 2))
    pos[:, 0] = u
    pos[:, 1] = v
    if voronoi is True:
        pos_vor = np.zeros((len(u1), 2))
        pos_vor[:, 0] = u1
        pos_vor[:, 1] = v1

    if viz is True:
        # size of square; equivalent to trunc
        fmin = (max(0, (base - side) // 2)) / np.sqrt(2 * base)
        fmax = (min(base, (base + side) // 2)) / np.sqrt(2 * base)
        tmin = (base - side // 2) / np.sqrt(2 * base)
        tmax = (base + side // 2) / np.sqrt(2 * base)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.imshow(np.log10(Sww), origin='lower',extent=[0, (2 * base) / np.sqrt(2 * base), 0, (base) / np.sqrt(2 * base)], cmap=cmocean.cm.deep)
        ax.scatter(pos[:, 0], pos[:, 1], color='w', s=40)

        ax.set_xlim([tmin, tmax])
        ax.set_ylim([fmin, fmax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        fig.subplots_adjust(left=0.04, bottom=0.05)
        # tri = Delaunay(pos)
        # plt.triplot(pos[:, 0], pos[:, 1], tri.simplices,color = 'green')
        # plt.show()

        if voronoi is True:
            # plt.triplot(points[:, 0], points[:, 1], tri.simplices, linestyle='--')
            vor = Voronoi(pos_vor)
            voronoi_plot_2d(vor, ax)

        if voronoi is True:
            return Sww, pos, spectro, chirp, base, pos_vor0
        else:
            return Sww, pos, spectro, chirp, base

    else:
        if voronoi is True:
            return Sww, pos, spectro, chirp,  base, pos_vor0
        else:
            return Sww, pos, spectro, chirp,  base

def findVoronoiMask(Sww, pos_vor, th=2, base=128, viz=False):

    #This function creates a mask based on the Voronoi tessellation of spectrograms local maxima

    #get Local Maxima that are greater than th
    index1=np.where(Sww[tuple(pos_vor)]>th)
    index=index1[0]
    # define a deleauney triangulation with zeros
    y1 = pos_vor[0]
    x1 = pos_vor[1]

    u1 = x1 / np.sqrt(2 * base)
    v1 = y1 / np.sqrt(2 * base)

    pos = np.zeros((len(u1), 2))
    pos[:, 0] = u1
    pos[:, 1] = v1

    # find Voronoi tesselation
    vor=Voronoi(pos)
    # define a grid corresponding to the time-frequency paving
    vecx = (np.arange(0, Sww.shape[0]) / np.sqrt(2 * base))
    vecy = (np.arange(0, Sww.shape[1]) / np.sqrt(2 * base))
    g = np.transpose(np.meshgrid(vecy, vecx))

    #result = kdpos.query_ball_point(g, radi_seg).T
    empty = np.zeros(len(vecx) * len(vecy), dtype='bool')
    g = g.reshape(len(vecx) * len(vecy), 2)
    for i in range(len(index)):

            path = mpltPath.Path(vor.vertices[vor.regions[vor.point_region[index[i]]]])
            # inside2 = path.contains_points(grid[0,:,:])
            empty = empty | path.contains_points(g)

    empty = empty.reshape(len(vecy), len(vecx)).T
    #empty = np.zeros(result.shape, dtype=bool)
    #for i in range(len(vecx)):
     #   for j in range(len(vecy)):
      #      empty[i,j] = len(result[i, j]) < 1

    if viz is True:
        # then plot
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(g[..., 0], g[..., 1], s=0.1, color='w')
        ax.imshow(empty, extent=[0, (2*base)/np.sqrt(2*base), 0, (base)/np.sqrt(2*base)], origin='lower', cmap=cmocean.cm.deep_r)
        ax.scatter(pos_exp[:,0], pos_exp[:, 1], color='w', s=30)

        side = 110*(base/128)
        fmin = (max(0, (base-side)//2))/np.sqrt(2*base)
        fmax = (min(base, (base+side)//2))/np.sqrt(2*base)
        tmin = (base-side//2)/np.sqrt(2*base)
        tmax = (base+side//2)/np.sqrt(2*base)

        ax.set_xlim([tmin, tmax])
        ax.set_ylim([fmin, fmax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        fig.subplots_adjust(left=0.04, bottom=0.05)
        points=pos_exp
        plt.plot(points[:, 0], points[:, 1], 'o')
        voronoi_plot_2d(vor, ax)

        plt.show()


    return empty

def maskAddition(cluster, masks, X, delauneyMaskShape):
    #function that adds the masks labeled in list 'masks'
    new_mask=np.zeros(delauneyMaskShape, dtype='bool')
    for i in masks:
            tempMask = clusterMask(cluster, i, X, delauneyMaskShape)
            new_mask = new_mask | tempMask

    return  new_mask

def realSignal(signal, zeros_th=1e16,fake_noise=False ,viz=False, shrink=False, voronoi=False):
    # This function is used to compute the stft of a real signal.
    # It can also be used to visualize the spectrograms zeros.
    # It has the option fake_noise = True/False. That is used as noise padding to help the Delaunay algorithm produce better results.

    duration = len(signal)
    # signal normalization
    signal = signal - np.mean(signal)
    # signal calibration
    c = 1 / (np.max(np.abs(signal)))
    signal = signal * c


    base=128
    while base<duration:
        base=base*2

    scale = int(base / 128)
    side = 110*scale

    N = 4 * base
    Nfft = 2 * base
    t = np.arange(N)

    # bounds for detection (larger)
    fmin = 0
    fmax = base

    # tmin = 2*base - (base-trunc) // 2
    # tmax = 2*base + (base-trunc) // 2
    tmin = base
    tmax = 3 * base

    #fake noise
    std=0.1
    #std less than 1 so the zeros will have higher density
    if fake_noise is True:
        w = np.random.normal(0, std, N)
    else:
        w=np.zeros(N)
    # window
    g = sg.gaussian(Nfft, np.sqrt((Nfft) / 2 / np.pi))
    g = g / g.sum()
    # chirp
    duration = int(np.floor(duration))

    #if duration > base:
     #   raise ValueError('Duration should be lesser than base')
    start_s = 2 * base - duration // 2
    end_s = 2 * base + duration // 2
    if duration % 2 == 1:
        end_s=end_s+1
    chirp = np.zeros(N)


    chirp[start_s:end_s] = sg.tukey(duration) * signal
    w[start_s:end_s]=0
    x0 = np.sqrt(2) * chirp+w/4
    # spectro = STFT(x0, g, Nfft)
    _, _, spectro = sg.stft(x0, window=g, nperseg=Nfft, noverlap=Nfft - 1)
    Sww_t = abs(spectro) ** 2
    # print("STFT computed")

    Sww = Sww_t[fmin:fmax + 1, tmin:tmax + 1]

    # zeros detection
    y0, x0 = extr2minth(Sww, zeros_th)

    # Voronoi
    if voronoi is True:
        y1, x1 = extr2maxth(Sww, th)
        pos_vor0 = [y1, x1]

    if shrink is True:
        # boundary conditions
        side = 110 * scale  # size of square; equivalent to trunc
        fmin_b = (max(0, (base - side) // 2))
        fmax_b = (min(base, (base + side) // 2))
        tmin_b = (base - side // 2)
        tmax_b = (base + side // 2)

        mask = (y0 > fmin_b) * (y0 < fmax_b) * (x0 > tmin_b) * (x0 < tmax_b)
        u = x0[mask] / np.sqrt(2 * base)
        v = y0[mask] / np.sqrt(2 * base)

        if voronoi is True:
            u1 = x1[mask] / np.sqrt(2 * base)
            v1 = y1[mask] / np.sqrt(2 * base)

    else:
        u = x0 / np.sqrt(2 * base)
        v = y0 / np.sqrt(2 * base)
        if voronoi is True:
            u1 = x1 / np.sqrt(2 * base)
            v1 = y1 / np.sqrt(2 * base)

    pos = np.zeros((len(u), 2))
    pos[:, 0] = u
    pos[:, 1] = v
    if voronoi is True:
        pos_vor = np.zeros((len(u1), 2))
        pos_vor[:, 0] = u1
        pos_vor[:, 1] = v1

    if viz is True:
        # size of square; equivalent to trunc
        fmin = (max(0, (base - side) // 2)) / np.sqrt(2 * base)
        fmax = (min(base, (base + side) // 2)) / np.sqrt(2 * base)
        tmin = (base - side // 2) / np.sqrt(2 * base)
        tmax = (base + side // 2) / np.sqrt(2 * base)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.imshow(np.log10(Sww), origin='lower',extent=[0, (2 * base) / np.sqrt(2 * base), 0, (base) / np.sqrt(2 * base)], cmap=cmocean.cm.deep)
        ax.scatter(pos[:, 0], pos[:, 1], color='w', s=40)

        ax.set_xlim([tmin, tmax])
        ax.set_ylim([fmin, fmax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        fig.subplots_adjust(left=0.04, bottom=0.05)

        if voronoi is True:
            # plt.triplot(points[:, 0], points[:, 1], tri.simplices, linestyle='--')
            vor = Voronoi(pos_vor)
            voronoi_plot_2d(vor, ax)

        if voronoi is True:
            return Sww, pos, spectro, chirp, base, pos_vor0
        else:
            return Sww, pos, spectro, chirp, base

    else:
        if voronoi is True:
            return pos, base, spectro, chirp, base, pos_vor0
        else:
            return pos, base, spectro, chirp, base

def keepOnlyUsefullMasks(cluster, base, X ):
    # This function is used to cut off the masks that are near the edges of the spectrogram.
    # It finds the centroids of every mask and cuts those ones that are out of bounds (fmin:fmax, tmin:tmax)

    scale = base/128
    side = 110*scale

    fmin = (max(0, (base - side)))
    fmax = (min(base, (base + side)//2 -scale*10))
    tmin = (base - side // 2)
    tmax = (base + side // 2)

    centroids=np.zeros([np.max(cluster),2])
    clusters_to_keep=[]

    for maskLabel in range(np.max(cluster)):
        points_of_cluster = X[cluster == maskLabel]
        centroids[maskLabel]=np.mean(points_of_cluster, axis=0)
        #identifying if
        if centroids[maskLabel,0]>tmin and centroids[maskLabel,0]<tmax and centroids[maskLabel,1]>fmin and centroids[maskLabel,1]<fmax:
            clusters_to_keep.append(maskLabel)

    return clusters_to_keep

#hanford gravitational wave signal GW150914

# hanford=np.loadtxt('hanford.txt', delimiter=' ')
# signal1=hanford[:,1]
# signal=signal1[::10]
# zeros_th=1.5e-04
#
# Sww, pos_exp, stft, chirp, b = filteredSignal(signal, zeros_th, viz=True)
#
# th=1.7
# delauney_mask = findDelauneyMask(Sww, pos_exp,th ,base=b, viz=True)
#
# X=fromMaskToX(delauney_mask)
#
# clustering= DBSCAN(eps=1, min_samples=3).fit(X)
# cluster=clustering.labels_
#
# show_clusters(X, cluster)
#
# cluster_mask= clusterMask(cluster,10,X, delauney_mask.shape)
#
# SNR=1
# amp = np.sqrt(2*SNR)
# xr, xor, x_true =reconstructionSignal(cluster_mask, stft, amp*chirp,base=b, viz= True)

#mean_squared_error(x_true, xr)

#xr, xor, x_true =reconstructionSignal(cluster_mask, stft, amp*chirp,base=b, viz= True)
#x=np.arange(-35,35,0.1)
#hermF = hermite_functions(400, x, all_n=False)
# SNR=20
#
# #chirp
# fmin = 0
# fmax = base
#
#
# tmin = base
# tmax = 3 * base
# t = np.arange(4*base)
# b = 150
# a = base - b
#
# #chirp
# duration = 100
# start_s = 2 * base - duration // 2
# end_s = 2 * base + duration // 2
# chirp= np.zeros(duration)
# N=4*base
# freq = (a + b * t[start_s:end_s] / N) * t[start_s:end_s] / N
# chirp = sg.tukey(duration) * np.cos(2 * np.pi * freq)
#
# signal=chirp
#
# # bat chirp
# #f = open('batChirp.txt', 'r+')
# #lines = [line for line in f.readlines()]
# #f.close()
# #for i in range(len(lines)):
#  #   lines[i]=lines[i].rstrip("\n")
#
# #lines = [float(i) for i in lines]
# #signal=lines
#
# #triangle threshold side performance
# side_th =np.arange(1, 2.5, 0.05)
# mse=np.zeros(len(side_th))
# j=0
# std=1
# Sww, pos_exp, stft, chirp, b = realSignal(SNR, signal, std, viz=False, shrink=False, voronoi=False)
#
# for i in side_th:
#
#     #Deleauney mask
#
#     delauney_mask = findDelauneyMask(Sww, pos_exp, i,base=b, viz=False)
#
#     X=fromMaskToX(delauney_mask)
#
#     clustering= DBSCAN(eps=1, min_samples=3).fit(X)
#     cluster=clustering.labels_
#
#    # show_clusters(X, cluster)
#
#     #cluster_mask= clusterMask(cluster,8,X,delauney_mask.shape)
#
#     masks=keepOnlyUsefullMasks(cluster,b,X)
#
#     added_mask= maskAddition(cluster, masks, X ,delauney_mask.shape)
#     # reconstruction
#     amp = np.sqrt(2*SNR)
#     #xr, xor, x_true =reconstructionSignal(energy_mask, stft, amp*chirp,base=b, viz= True)
#
#     #mean_squared_error(x_true, xr)
#
#     xr, xor, x_true =reconstructionSignal(added_mask, stft, amp*chirp,base=b, viz= False)
#     mse[j]=mean_squared_error(x_true, xr)
#     j=j+1
#
# # and plot the results
# fig1, ax1 = plt.subplots()
# ax1.plot(side_th,mse)
# ax1.set_title("Mean squared error performance of triangle side threshold")
# ax1.set_xlabel("side threshold")
# ax1.set_ylabel("MSE")
#
#
# #standar deviation performance
# std = np.arange(0.2, 2.5, 0.1)
# side_th=1.8
# mse=np.zeros(len(std))
# j=0
# mse2=np.zeros(len(std))
# th=0.4
# for i in std:
#     Sww, pos_exp, stft, chirp, b = realSignal(SNR, signal, i, viz=False, shrink=False, voronoi=False)
#     # Deleauney mask
#
#     delauney_mask = findDelauneyMask(Sww, pos_exp, side_th, base=b, viz=False)
#
#     # voronoi_mask= findVoronoiMask(Sww,pos_vor,base=b,th=1)
#     energy_mask= findEnergyThresholdMask(Sww,th,b,viz=False)
#     # convex hull
#     # hull,points= getConvexHull(Sww, pos_exp, empty_mask)
#
#     X = fromMaskToX(delauney_mask)
#
#     clustering = DBSCAN(eps=1, min_samples=3).fit(X)
#     cluster = clustering.labels_
#
#     # show_clusters(X, cluster)
#
#     # cluster_mask= clusterMask(cluster,8,X,delauney_mask.shape)
#
#     masks = keepOnlyUsefullMasks(cluster, b, X)
#
#     added_mask = maskAddition(cluster, masks, X, delauney_mask.shape)
#     # reconstruction
#     amp = np.sqrt(2 * SNR)
#     xr, xor, x_true =reconstructionSignal(energy_mask, stft, amp*chirp,base=b, viz= False)
#
#     mse2[j]= mean_squared_error(x_true, xr)
#
#     xr, xor, x_true = reconstructionSignal(added_mask, stft, amp*chirp, base=b, viz=False)
#     mse[j] = mean_squared_error(x_true, xr)
#     j = j + 1
#
# # and plot the results
# fig2, ax2 = plt.subplots()
# ax2.plot(std,mse)
# ax2.plot(std,mse2)
# ax2.set_title("Mean squared error performance with different noise standar deviation")
# ax2.set_xlabel("standar deviation")
# ax2.set_ylabel("MSE")
# plt.legend(["deleuney threshold","energy mask"])
#
# #energy threshold perfonmacne
# th=np.arange(0,3,1/25)
# mse=np.zeros(len(th))
# j=0
# std=1
# for i in th:
#     Sww, pos_exp, stft, chirp, b = realSignal(SNR, signal,std , viz=False, shrink=False, voronoi=False)
#     energy_mask= findEnergyThresholdMask(Sww,i,b,viz=False)
#     amp = np.sqrt(2 * SNR)
#     xr, xor, x_true = reconstructionSignal(energy_mask, stft, amp * chirp, base=b, viz=False)
#     mse[j] = mean_squared_error(x_true, xr)
#     j = j+1
#
# # and plot the results
# fig3, ax3 = plt.subplots()
# ax3.plot(th,mse)
# ax3.set_title("Mean squared error performance of energy mask threshold")
# ax3.set_xlabel("Energy threshold")
# ax3.set_ylabel("MSE")
#
#
#
# # SNR performance
# SNR=np.arange(0,40)
# std = 1
# side_th=1.8
# mse=np.zeros(len(SNR))
# j=0
# mse2=np.zeros(len(SNR))
# th=0.4
# for i in SNR:
#     Sww, pos_exp, stft, chirp, b = realSignal(i, signal, std, viz=False, shrink=False, voronoi=False)
#     # Deleauney mask
#
#     delauney_mask = findDelauneyMask(Sww, pos_exp, side_th, base=b, viz=False)
#
#     # voronoi_mask= findVoronoiMask(Sww,pos_vor,base=b,th=1)
#     energy_mask= findEnergyThresholdMask(Sww,th,b,viz=False)
#     # convex hull
#     # hull,points= getConvexHull(Sww, pos_exp, empty_mask)
#
#     X = fromMaskToX(delauney_mask)
#
#     clustering = DBSCAN(eps=1, min_samples=3).fit(X)
#     cluster = clustering.labels_
#
#     # show_clusters(X, cluster)
#
#     # cluster_mask= clusterMask(cluster,8,X,delauney_mask.shape)
#
#     masks = keepOnlyUsefullMasks(cluster, b, X)
#
#     added_mask = maskAddition(cluster, masks, X, delauney_mask.shape)
#     # reconstruction
#     amp = np.sqrt(2 * i)
#     xr, xor, x_true =reconstructionSignal(energy_mask, stft, amp*chirp,base=b, viz= False)
#
#     mse2[j]= mean_squared_error(x_true, xr)
#
#     xr, xor, x_true = reconstructionSignal(added_mask, stft, amp*chirp, base=b, viz=False)
#     mse[j] = mean_squared_error(x_true, xr)
#     j = j + 1
# # and plot the results
# fig4, ax4 = plt.subplots()
# ax4.plot(SNR,mse)
# ax4.plot(SNR,mse2)
# ax4.set_title("Mean squared error performance with different SNR")
# ax4.set_xlabel("SNR")
# ax4.set_ylabel("MSE")
# plt.legend(["deleuney threshold","energy mask"])

#side_th = 1.9
#delauney_mask2 =  findDelauneyMask(Sww2, pos_exp2, side_th,base=b2)


#fname=delauney_mask
#blur_radius = 1.0
#threshold = 3
#img = Image.fromarray(delauney_mask)
#img = np.asarray(img)
#print(img.shape)

#imgf = ndimage.gaussian_filter(img, blur_radius)
#threshold = 3

# find connected components
#labeled, nr_objects = ndimage.label(imgf > threshold)
#print("Number of objects is {}".format(nr_objects))
# Number of objects is 4

#plt.imshow(labeled)

#plt.show()
#f = open('batChirp.txt', 'r+')
#my_file_data = f.read()
#f.close()
#f = open('batChirp.txt', 'r+')
#lines = [line for line in f.readlines()]
#f.close()
#for i in range(len(lines)):
 #   lines[i]=lines[i].rstrip("\n")

#lines = [float(i) for i in lines]
#signal=lines
#Sww, pos_exp, stft, chirp, b= realSignal(SNR, signal= lines , viz=True, shrink=False)

#radi_seg = 1.8 # fix segmentation radius
#delauney_mask =  findDelauneyMask(Sww, pos_exp, radi_seg,base=b)
#testing DBSCAN clustering

#df=DataFrame(dict(x=X[:,0],y=X[:,1]))
#fig, ax= plt.subplots(figsize=(8,8))
#df.plot(ax=ax,kind='scatter',x='x',y='y')
#plt.xlabel('X_1')
#plt.ylabel('X_2')
#plt.show

#X=fromMaskToX(delauney_mask)

#clustering= DBSCAN(eps=1, min_samples=3).fit(X)
#cluster=clustering.labels_

#show_clusters(X, cluster)

#cluster_mask= clusterMask(cluster,9,X,delauney_mask.shape)

#amp = np.sqrt(2*SNR)
#reconstructionSignal(cluster_mask, stft, amp*chirp, base=b)




