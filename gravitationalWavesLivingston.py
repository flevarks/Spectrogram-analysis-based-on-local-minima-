import  spectrogramFunctions as sf
import numpy as np
from scipy.signal import decimate
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt




# Livingston gravitational wave signal GW150914
# load the signal
hanford = np.loadtxt('Livingston.txt', delimiter=' ')
signal1 = hanford[:,1]

numHanford= np.loadtxt('LivingstonNum.txt', delimiter=' ')
numHan=numHanford[:, 1]
numHan=numHan[::16]
# filtering and resampling 1/16

signal = decimate(signal1,16,ftype='iir')

# finding spectrogram
zeros_th = 1.5e-03
area_th = 0
Sww, pos_exp, stft, chirp, b = sf.realSignal(signal, zeros_th,fake_noise= True,  viz=True)

thR=1.85

# finding mask based Delaunay transformation of spectrogram zeros
delauney_mask = sf.findDelauneyMask(Sww, pos_exp,thR,area_th ,base=b, viz=True)

# Mask clustering with DBSCAN algorithm

X = sf.fromMaskToX(delauney_mask)

clustering = DBSCAN(eps=1, min_samples=2).fit(X)
cluster = clustering.labels_

sf.show_clusters(X, cluster)

# choose the masks that we keep

masks = [8]
added_mask = sf.maskAddition(cluster, masks, X, delauney_mask.shape)


# cluster_mask= sf.clusterMask(cluster,10,X, delauney_mask.shape)

# calculating istft and plotting the results

SNR=20
amp = np.sqrt(2*SNR)
xr, xor, x_true =sf.reconstructionSignal(added_mask, stft, amp*chirp,base=b, viz= True, fake_SNR=SNR, real_Signal=True)

base=b
tmin=base
tmax=base*3
duration= len(signal)

#   raise ValueError('Duration should be lesser than base')

time = hanford[:,0]
time=time[::16]


start_s = 2 * base - duration // 2
end_s = 2 * base + duration // 2
if duration % 2 == 1:
    end_s = end_s + 1

x_true = np.array(x_true)
c = 1 / (np.max(np.abs(x_true)))
x_true = x_true * c


xr = np.array(xr)
c = 1 / (np.max(np.abs(xr)))
xr = xr * c


numHan = np.array(numHan)
c = 1 / (np.max(np.abs(numHan)))
x_true = numHan * c

fig, ax = plt.subplots()
ax.plot(hanford[:,0], hanford[:,1])
ax.set_xlabel("time [s]")
ax.set_ylabel("strain [e-21]")


fig, ax = plt.subplots()
ax.plot(time, x_true[start_s:end_s])
ax.set_xlabel("time [s]")
ax.set_ylabel("strain [e-21]")


fig, ax = plt.subplots()
ax.plot(time, xr[start_s:end_s])
ax.plot(time[0:end_s-start_s],numHan)
ax.set_xlabel("time [s]")
ax.set_ylabel("strain [e-21]")
ax.set_title("???????????????? ???????????????????????????????????? ?????????????? ???? ???????????????????? ???????????????????? ?????????????????? ??????????????, Livingston.")
ax.legend(["Reconstructed signal", "Numerical Relativity"])



fig, ax = plt.subplots()
ax.plot(time, x_true[start_s:end_s]-xr[start_s:end_s])
ax.set_xlabel("time [s]")
ax.set_ylabel("strain [e-21]")


f=np.arange(0,200,1)

fig, ax = plt.subplots()
ax.pcolormesh(time, f, np.abs(stft[0:200,start_s:end_s]),shading='auto')
ax.set_yticklabels([])
ax.set_xlabel("time (s)")
ax.set_ylabel("frequency")
ax.set_title("???????????????????????????? ?????????????????? ??????????????, Livingston")
