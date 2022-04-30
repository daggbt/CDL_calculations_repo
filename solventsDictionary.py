import numpy as np
import CDLconfig as config

pcSolvent = config.Solvent()
pcSolvent.name = "Propylene_Carbonate"
pcSolvent.radiusAng = 2.75
pcSolvent.dielectric = 66.14
pcSolvent.reducedDielectric = 0

waterSolvent = config.Solvent()
waterSolvent.name = "Water"
waterSolvent.radiusAng = 1.715
waterSolvent.dielectric = 78
waterSolvent.reducedDielectric = 0


