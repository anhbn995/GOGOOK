import gdal,gdalconst
import os

cmd='rio cogeo create {0}_sigma0-4326.tif {0}_sigma0_cog.tif'.format(fname)
!{cmd}