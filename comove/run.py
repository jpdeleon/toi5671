import Comove

## Define the inputs:
##An target star name searchable on simbad. If coordinates are used, it will just be used as the default results directory
targname="TOI-5671"

##Alternative is to use coordinates, default is [None,None] in which case the targname is used to get coordinates
#rd = [None,None]
rd = ['14:04:02.21','38:37:05.99']

##input target star radial velocity to calulate 3D space velocities
#radvel=-19.82 ##km/s
radvel=-1.75 #from IRD

##Neighbour velocity difference limit, and on sky search radius
vlim=5.0 ##km/s
srad=25.0 ##parsecs (spherical radius around target)


##This line runs the entire code. Set showplots=True to interactively plot, otherwise they are only saved as pngs
##Set verbose=True to see LOTS of print output
output_location = Comove.findfriends(targname,radvel,velocity_limit=vlim,search_radius=srad,radec=rd,output_directory=None,verbose=False,showplots=False)

