import numpy as np
from CDLconfig import *

def getHardSphereRadiusFromGaussian( gaussianRadius ):
  
  return gaussianRadius * np.power( 3.0*np.sqrt(np.pi)/4., 1.0/3.0 )  