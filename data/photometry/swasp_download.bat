#!/bin/sh

# If wget is not installed on your system,
# please refer to http://irsa.ipac.caltech.edu/docs/batch_download_help.html.
#
# Windows users: the name of wget may have version number (ie: wget-1.10.2.exe)
# Please rename it to wget in order to successfully run this script
# Also the location of wget executable may need to be added to the PATH environment.
#
wget -O '1SWASP J140402.34+383711.2.fits' 'http://exoplanetarchive.ipac.caltech.edu:80/data/ETSS//SuperWASP/FITS/DR1/tile210126/1SWASP J140402.34+383711.2.fits' -a search_39953320.log
wget -O '1SWASP_J140402.34+383711.2_lc.tbl' 'http://exoplanetarchive.ipac.caltech.edu:80/data/ETSS//SuperWASP/TBL/DR1/tile210126/1SWASP_J140402.34+383711.2_lc.tbl' -a search_39953320.log
