import numpy as np 
import CDLconfig as config
import scipy.constants as sc

def getHardSphereRadiusFromGaussian( gaussianRadius ):
  #return gaussianRadius * numpy.cbrt( 3.0*numpy.sqrt(numpy.pi)/4. )
  # where did cbrt disappear to???
  return gaussianRadius * np.power( 3.0*np.sqrt(np.pi)/4., 1.0/3.0 )  

NaIon = config.Ion()
NaIon.name="Na"
NaIon.charge=1
NaIon.gaussianRadiusAng=0.607246  # Na+ bare
NaIon.radiusAng = getHardSphereRadiusFromGaussian( NaIon.gaussianRadiusAng )
NaIon.polarizability = 1.0015 *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

hydNaw3Ion = config.Ion()
hydNaw3Ion.name="hydNaw3"
hydNaw3Ion.charge=1
hydNaw3Ion.gaussianRadiusAng=2.24981  # hydNaw3
hydNaw3Ion.radiusAng = getHardSphereRadiusFromGaussian( hydNaw3Ion.gaussianRadiusAng )
hydNaw3Ion.polarizability = 0  *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

hydroniumIon = config.Ion()
hydroniumIon.name="hydronium"
hydroniumIon.charge=1
hydroniumIon.gaussianRadiusAng=0.973925  # hydronium
hydroniumIon.radiusAng = getHardSphereRadiusFromGaussian( hydroniumIon.gaussianRadiusAng )
hydroniumIon.polarizability = 0  *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

simpleOHion = config.Ion()
simpleOHion.name="OH"
simpleOHion.charge=-1
simpleOHion.gaussianRadiusAng=1.25953  # simple OH-
simpleOHion.radiusAng = getHardSphereRadiusFromGaussian( simpleOHion.gaussianRadiusAng )
NaIon.polarizability = 0  *1.6487773e-41    # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

hydOHw3Ion = config.Ion()
hydOHw3Ion.name="hydOHw3"
hydOHw3Ion.charge=-1
hydOHw3Ion.gaussianRadiusAng=2.39243  # hydOHw3
hydOHw3Ion.radiusAng = getHardSphereRadiusFromGaussian( hydOHw3Ion.gaussianRadiusAng )
hydOHw3Ion.polarizability = 0  *1.6487773e-41    # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

ClIon = config.Ion()
ClIon.name="Cl"
ClIon.charge=-1
ClIon.gaussianRadiusAng=1.86058  # Cl[alt]
ClIon.radiusAng = getHardSphereRadiusFromGaussian( ClIon.gaussianRadiusAng )
# ~ ClIon.polarizability = 33.7  *1.6487773e-41    # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available
ClIon.polarizability =3.45  * 4*np.pi*sc.epsilon_0 *1e-30   # in Angostrom^3 polarizability data from Phys. Chem. Chem. Phys., 2013, 15, 2703

hydLi5Ion = config.Ion()
hydLi5Ion.name="hydLi5Ion"
hydLi5Ion.charge=1
hydLi5Ion.gaussianRadiusAng=2.56195
hydLi5Ion.radiusAng = getHardSphereRadiusFromGaussian( hydLi5Ion.gaussianRadiusAng )
hydLi5Ion.polarizability = 0.193   *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

hydNa3Ion = config.Ion()
hydNa3Ion.name="hydNa3Ion"
hydNa3Ion.charge=1
hydNa3Ion.gaussianRadiusAng=2.24981
hydNa3Ion.radiusAng = getHardSphereRadiusFromGaussian( hydNa3Ion.gaussianRadiusAng )
hydNa3Ion.polarizability = 1.0015    *1.6487773e-41  # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

hydOH3Ion = config.Ion()
hydOH3Ion.name="hydOH3Ion"
hydOH3Ion.charge=-1
hydOH3Ion.gaussianRadiusAng=2.39243
hydOH3Ion.radiusAng = getHardSphereRadiusFromGaussian( hydOH3Ion.gaussianRadiusAng )
hydOH3Ion.polarizability = 0   *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

hydOHIon = config.Ion()
hydOHIon.name="hydOHIon"
hydOHIon.charge=-1
hydOHIon.gaussianRadiusAng= 0#2.39243
hydOHIon.radiusAng = 3 #getHardSphereRadiusFromGaussian( hydOH3Ion.gaussianRadiusAng ) From Jacob Israelachvili, Intermolecular and surface forces (III Edition), 2011
hydOHIon.polarizability = 0   *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

########################################################################################Added from Drew's paper########################################################################################

liIon = config.Ion()
liIon.name="Li"
liIon.charge=1
liIon.gaussianRadiusAng=0.38467
liIon.radiusAng = getHardSphereRadiusFromGaussian( liIon.gaussianRadiusAng )
liIon.polarizability = 0.193   *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

potassiumIon = config.Ion()
potassiumIon.name="K"
potassiumIon.charge=1
potassiumIon.gaussianRadiusAng=3.010343  
potassiumIon.radiusAng = getHardSphereRadiusFromGaussian( potassiumIon.gaussianRadiusAng )
potassiumIon.polarizability = 5.47   *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

pf6Ion = config.Ion()
pf6Ion.name="PF_6"
pf6Ion.charge=-1
pf6Ion.gaussianRadiusAng=2.31
pf6Ion.ls = 1E-10*2.31*2/(24**0.5)
pf6Ion.radiusAng = getHardSphereRadiusFromGaussian( pf6Ion.gaussianRadiusAng )
pf6Ion.polarizability = 4.18  * 4*np.pi*sc.epsilon_0 *1e-30    # in Angostrom^3 polarizability data from Phys. Chem. Chem. Phys., 2013, 15, 2703

bf4Ion = config.Ion()
bf4Ion.name="BF_4"
bf4Ion.charge=-1
bf4Ion.gaussianRadiusAng=2.09
bf4Ion.radiusAng = getHardSphereRadiusFromGaussian( bf4Ion.gaussianRadiusAng )
bf4Ion.polarizability = 2.8  * 4*np.pi*sc.epsilon_0 *1e-30    # in Angostrom^3 polarizability data from Phys. Chem. Chem. Phys., 2013, 15, 2703

clo4Ion = config.Ion()
clo4Ion.name="ClO_4"
clo4Ion.charge=-1
clo4Ion.gaussianRadiusAng=2.17
clo4Ion.radiusAng = getHardSphereRadiusFromGaussian( clo4Ion.gaussianRadiusAng )
clo4Ion.polarizability = 6.02 * 4*np.pi*sc.epsilon_0 *1e-30     # in Angostrom^3 polarizability data from Phys. Chem. Chem. Phys., 2013, 15, 2703

bro4Ion = config.Ion()
bro4Ion.name="BrO_4"
bro4Ion.charge=-1
bro4Ion.gaussianRadiusAng=2.27
bro4Ion.radiusAng = getHardSphereRadiusFromGaussian( bro4Ion.gaussianRadiusAng )
bro4Ion.polarizability = 0   *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available

io4Ion = config.Ion()
io4Ion.name="IO_4"
io4Ion.charge=-1
io4Ion.gaussianRadiusAng=2.36
io4Ion.radiusAng = getHardSphereRadiusFromGaussian( io4Ion.gaussianRadiusAng )
io4Ion.polarizability = 0   *1.6487773e-41   # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available


####### radius measured using Avogadro software##########
bistriflimideIon = config.Ion()
bistriflimideIon.name="TFSI"
bistriflimideIon.charge=-1
bistriflimideIon.gaussianRadiusAng=3.42
bistriflimideIon.radiusAng = getHardSphereRadiusFromGaussian( bistriflimideIon.gaussianRadiusAng )
bistriflimideIon.polarizability = 13.59 * 4*np.pi*sc.epsilon_0 *1e-30     # in Angostrom^3 polarizability data from Phys. Chem. Chem. Phys., 2013, 15, 2703

emtIon = config.Ion()
emtIon.name="EM"
emtIon.charge=1
emtIon.gaussianRadiusAng=3.47
emtIon.radiusAng = getHardSphereRadiusFromGaussian( emtIon.gaussianRadiusAng )
emtIon.polarizability = 0  *1.6487773e-41    # polarizability data from Eur. Phys. J. D (2021) 75:46 Exptal value where available



