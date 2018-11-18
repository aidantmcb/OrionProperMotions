from astropy.io import fits
from astropy.table import Table, QTable
import astropy.table as table
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from math import sqrt, log10
from astropy.io import ascii
import astropy.io

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

#Open the appropriate fits file, extract relevant data, close file
#hdul = fits.open('c:\\users\\sahal\\2018TrinaryWork\\gaiadr2_OrionTables.fits')
hdul = fits.open('gaiadr2_OrionTables.fits')
pmra_all = hdul[1].data['pmra'] / 3600000 #pmra is in mas/yr - converts to deg/yr
pmdec_all = hdul[1].data['pmdec'] / 3600000 #pmdec is in mas/yr - converts to deg/yr
radial_velocity_all = hdul[1].data['radial_velocity'] #radial velocity is in mas/yr? or parsec/yr?
source_id_all = hdul[1].data['source_id']
ra_all = hdul[1].data['ra'] #ra in deg
dec_all = hdul[1].data['dec'] #dec in deg
parallax_all = hdul[1].data['parallax'] #parallax in mas

G_all = hdul[1].data['phot_g_mean_mag'] #magnitudes in G filter
Gb_all = hdul[1].data['phot_bp_mean_mag']#magnitudes in Gbp filter
Gr_all = hdul[1].data['phot_rp_mean_mag']#magnitudes in Brp filter

bp_rp_all = hdul[1].data['bp_rp']#color difference b-r
#This is the same as subtracting Gb - Gr

ra_err_all = hdul[1].data['ra_error']
dec_err_all = hdul[1].data['dec_error']
pmra_err_all = hdul[1].data['pmra_error']
pmdec_err_all = hdul[1].data['pmdec_error']

hdul.close()

#Get the indices ii of all values for stars that satisfy appropriate photometric conditions
ii = []

Mg_all = [G + 5 - 5*(np.log10(1000/p)) for G, p in zip(G_all, parallax_all)]

for i in range(len(source_id_all)):
    if( (Mg_all[i]<2.46*(bp_rp_all[i])+2.76 and .3<(bp_rp_all[i])<1.8)
       or (Mg_all[i]<2.8*(bp_rp_all[i])+2.16 and 1.8<(bp_rp_all[i]))):
       #or (Mg_all[i]>2.14*(bp_rp_all[i])-.57 and .5<(bp_rp_all[i])<1.2)
       #or (Mg_all[i]>1.11*(bp_rp_all[i])+.66 and 1.2<(bp_rp_all[i])<3)):
        ii.append(i)

#Redefine our data as all stars that fit photometric parameters of young stars
pmra = pmra_all[ii]
pmdec = pmdec_all[ii]
radial_velocity = radial_velocity_all[ii]
source_id = source_id_all[ii]
ra = ra_all[ii]
dec = dec_all[ii]
parallax = parallax_all[ii]

G = G_all[ii]
Gb = Gb_all[ii]
Gr = Gr_all[ii]

bp_rp = bp_rp_all[ii]

ra_err = ra_err_all[ii]
dec_err = dec_err_all[ii]
pmra_err = pmra_err_all[ii]
pmdec_err = pmdec_err_all[ii]

###############################################

fnam = 'c:\\users\\sahal\\2018trinarywork\\jupyterfiles\\timecoords.npz'
inVals = np.load(fnam) #has entries ['time'] and ['ind']

tts = inVals['time']
iind = inVals['ind']

tts = np.flip(tts,0)
iind = np.flip(iind,0)


ta = Table([tts, np.array(iind)], names = ('times', 'inds'), meta = {'name': 'IndTable'})
ta.remove_rows(slice(0,4))
tts = ta['times']
iinds = ta['inds']

################################################################################
#Now match them up by distance:
################################################################################

def getCoords(ra,dec, pmra, pmdec, t):
    if t != 0:
        pmra_atT = pmra * t
        pmdec_atT = pmdec * t
    else:
        pmra_atT = pmra
        pmdec_atT = pmdec

    ra_atT = ra + pmra_atT
    dec_atT = dec + pmdec_atT
    return [ra_atT, dec_atT, pmra, pmdec]

def dist(ra, dec):
    return np.array([np.sqrt( (ra[i]-ra)**2 +  (dec[i]-dec)**2 ) for i in range(len(ra))])

arcsec = 1/3600
for mm in range(len(ta)):
    chunk = ta['inds'][mm]
    tim = ta['times'][mm]
    coords = getCoords(ra, dec, pmra, pmdec,tim)
    if not (len(chunk) > 2):
        ta['inds'][mm] = [list(chunk)]
    else:
        mat = dist(coords[0][chunk], coords[1][chunk])
        a, b = mat.shape
        sel = np.tril_indices(a)
        mat[sel] = 0

        rawlist = list(np.where((mat < arcsec) & (mat != 0)))
        rawlist = [list(i) for i in rawlist]
        xs = [[chunk[i], chunk[j]] for i, j in zip(rawlist[0], rawlist[1])]
        ta['inds'][mm] = xs

################################################################################
#Remove all redundant pairs
################################################################################

checkoutPairs = []
remRows = []

for nn in range(len(ta)):
    checks = ta['inds'][nn]
    checkstrs = [str(cc) for cc in checks]
    #print(checkstrs)
    if set(checkstrs).issubset(set(checkoutPairs)):
        remRows.append(nn)
    else:
        keepvals = []
        for i in range(len(checkstrs)):
            if checkstrs[i] not in checkoutPairs:
                keepvals.append(checks[i])
            checkoutPairs.append(checkstrs[i])
        ta['inds'][nn] = keepvals
ta.remove_rows(remRows)

################################################################################
#Now we can make our plots
################################################################################


def getBox(ras, decs, pmras, pmdecs):
    maxDist = np.max([np.max(abs(pmras)),np.max(abs(pmdecs))])* 100


    xmin = np.mean(ras) - arcsec/2 - maxDist
    xmax = np.mean(ras) + arcsec/2 + maxDist

    ymin = np.mean(decs) - arcsec/2 - maxDist
    ymax = np.mean(decs) + arcsec/2 + maxDist

    return ([xmin, xmax], [ymin, ymax])



with PdfPages('ProxPlots.pdf') as pdf:
    for i in range(len(ta)): #DO NOT FORGET this is a small selection
        currentInd = ta['inds'][i]
        currentT = ta['times'][i]
        tiCo = getCoords(ra, dec, pmra, pmdec, currentT)
        for pair in currentInd:
            fig, axs = plt.subplots(figsize = (14,14))
            for sp in pair:
                x1 = tiCo[0][sp]
                y1 = tiCo[1][sp]
                xo1 = tiCo[2][sp]
                yo1 = tiCo[3][sp]

                axs.plot(x1, y1, '.')
                axs.plot([x1, x1+100*xo1],[y1,y1+100*yo1])

            axs.set_xlabel('Right Ascension (Ticks: 100 mas)')
            axs.set_ylabel('Declination (Ticks: 100 mas)')
            axs.set_title('Pair of stars projected at time ' + str(currentT) + ' years')

            xstops, ystops = getBox(tiCo[0][pair], tiCo[1][pair], tiCo[2][pair], tiCo[3][pair])
            axs.set_xlim(xstops)
            axs.set_ylim(ystops)

            axs.xaxis.set_ticks(np.arange(xstops[0], xstops[1], arcsec*.1), True)
            axs.xaxis.set_ticks(xstops, False)
            axs.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

            axs.yaxis.set_ticks(np.arange(ystops[0], ystops[1], arcsec*.1), True)
            axs.yaxis.set_ticks(ystops, False)
            axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

            axs.set_aspect('equal')
            pdf.savefig()
            plt.close()
