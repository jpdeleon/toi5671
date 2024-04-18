#https://ui.adsabs.harvard.edu/abs/2020MNRAS.496.5176D/abstract
import Comove

#targname="TOI-5671"
targname="2MASS J14040226+3837056"

##Alternative is to use coordinates, default is [None,None] in which case the targname is used to get coordinates
#rd = ['14:04:02.21','38:37:05.99']

radvel=-1.75 #from IRD

output = Comove.binprob(targname, targfilt='J', targDmag=2, targDmagerr=0.5, targsep=10, targDPM=5,targDPMerr=1,targDPI=None,Pradvel=None,Pdist=None,Pdisterr=None,PdistU=None,PdistL=None,Pmass=None,PT=None)
