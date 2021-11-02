import spectrogramFunctions as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from sklearn.cluster import DBSCAN
import time

duration=np.arange(100,800,100)
timeFlandrin=np.zeros(len(duration))
timeEnergy=np.zeros(len(duration))
k=0
for dur in duration:
    SNR = 15
    base = 128
    while base < dur:
        base = base * 2
    # chirp

    # chirp
    fmin = 0
    fmax = base

    tmin = base
    tmax = 3 * base
    t = np.arange(4 * base)
    b = 150
    a = base - b

    # chirp
    start_s = 2 * base - dur // 2
    end_s = 2 * base + dur // 2
    chirp = np.zeros(dur)
    N = 4 * base
    freq = (a + b * t[start_s:end_s] / N) * t[start_s:end_s] / N
    chirp = sg.tukey(dur) * np.cos(2 * np.pi * freq)

    signal = chirp

    reps=10

    c = 0
    std = 1
    side_th = 1.7
    SNR = 15
    snr= np.power(10, SNR/10)



    start = time.process_time()
    for j in range(reps):


        Sww, pos_exp, stft, chirp, b = sf.spectrogramSignal(SNR, signal, std, viz=False, shrink=False, voronoi=False)

        # Delauney mask
        delauney_mask = sf.findDelauneyMask(Sww, pos_exp, side_th, base=b, viz=False)

        X = sf.fromMaskToX(delauney_mask)

        clustering = DBSCAN(eps=1, min_samples=3).fit(X)
        cluster = clustering.labels_

        # show_clusters(X, cluster)

        # cluster_mask= clusterMask(cluster,8,X,delauney_mask.shape)

        masks = sf.keepOnlyUsefullMasks(cluster, b, X)

        added_mask = sf.maskAddition(cluster, masks, X, delauney_mask.shape)
        # reconstruction
        amp = np.sqrt(2 * snr)
        # xr, xor, x_true =reconstructionSignal(energy_mask, stft, amp*chirp,base=b, viz= True)

        # mean_squared_error(x_true, xr)


        xr, xor, x_true = sf.reconstructionSignal(added_mask, stft, amp * chirp, base=b, viz=False)

    timeFlandrin[k]= time.process_time() - start



    start = time.process_time()
    th=0.5
    for j in range(reps):


        Sww, pos_exp, stft, chirp, b = sf.spectrogramSignal(SNR, signal, std, viz=False, shrink=False, voronoi=False)

        # Delauney mask
        energy_mask = sf.findEnergyThresholdMask(Sww, th, b, viz=False)
        amp = np.sqrt(2 * snr)
        xr, xor, x_true = sf.reconstructionSignal(energy_mask, stft, amp * chirp, base=b, viz=False)
        # reconstruction
        amp = np.sqrt(2 * snr)
        # xr, xor, x_true =reconstructionSignal(energy_mask, stft, amp*chirp,base=b, viz= True)

        # mean_squared_error(x_true, xr)


        xr, xor, x_true = sf.reconstructionSignal(added_mask, stft, amp * chirp, base=b, viz=False)

    timeEnergy[k]= time.process_time() - start
    k=k+1

fig,ax= plt.subplots()
ax.plot(duration, timeEnergy/10)
ax.plot(duration, timeFlandrin/10)
ax.set_title("Σύγκριση ταχύτητας αλγορίθμου ενέργειας και αλγορίθμου Flandrin")
ax.set_xlabel("Αριθμος δειγμάτων σήματος (samples)")
ax.set_ylabel("Μέση ταχύτητα αλγορίθμων (s)")
ax.legend([ "energy mask","Flandrin Algorithm"])