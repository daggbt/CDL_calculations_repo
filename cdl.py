from mpi4py import MPI as MPI4
import h5py
import mpi4py
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import scipy.constants as sc
import CDLconfig as config 
import solventsDictionary as Solvent
import ionsDictionary as Ion


##################### user defined parameters ###################################
class properties:
  electrolyteIons = [Ion.liIon, Ion.pf6Ion] #[Ion.potassiumIon, Ion.hydOHIon]
  solvent = Solvent.pcSolvent #pcSolvent
  bulkConcentration = 1  # in Molar i.e mol/l
  potentialDifference = 0#1000  # in mV
  singleElectrode = False
  plot = True

prop = properties()
cath, an = prop.electrolyteIons
temperature = 298.15
epsilon = sc.epsilon_0*prop.solvent.dielectric

########################## properties of the ions ###############################
def getDebyeLength(bulkConcentration,electrolyteIons,solvent):
  # ~ epsilon = getReducedDielectricConstant(prop.solvent,potential)
  debyeLength = np.sqrt(epsilon*sc.k*temperature/sc.e/sc.e/(bulkConcentration*cath.charge**2 + bulkConcentration*an.charge**2)/1000/sc.N_A)
  return debyeLength

def ionStericConcentration(ionRadiusAng):
  ionRadius = ionRadiusAng*1e-10
  ionVolume = 4*np.pi*ionRadius**3/3
  stericConcentration = 1/ionVolume/1000/sc.N_A
  return stericConcentration

def getThresholdPotential(ion,bulkConcentration):
  thresholdPotential = -sc.k*temperature*np.log(ionStericConcentration(ion.radiusAng)/bulkConcentration)/sc.e/ion.charge
  return thresholdPotential*1000                            # *1000 convert it to mV
  
def getChargeBalancePotential(potentialDifference):
  potentialDifference = potentialDifference/1000             # /1000 convert it to V from mV
  if prop.singleElectrode:
    leftPotential = potentialDifference
    
  else:
    if potentialDifference < 0:
      counter_ion = cath 
      co_ion = an
    else:
      counter_ion = an 
      co_ion = cath
    z1 = counter_ion.charge
    nu1 = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
    z2 = co_ion.charge
    nu2 = 2*prop.bulkConcentration/ionStericConcentration(co_ion.radiusAng)
    
    leftPotential = potentialDifference/2
    rightPotential = leftPotential - potentialDifference
    
    leftPotential_threshold = getThresholdPotential(an,prop.bulkConcentration)/1000        
    rightPotential_threshold = getThresholdPotential(cath,prop.bulkConcentration)/1000
    
    if abs(leftPotential) > abs(leftPotential_threshold) or abs(rightPotential) > abs(rightPotential_threshold):
      leftPotential = (sc.k*temperature/sc.e)*(potentialDifference*z2*nu1*sc.e/sc.k/temperature + nu1*(1-nu2/2)**2 - nu2*(1-nu1/2)**2 + nu2*np.log(2/nu1) - nu1*np.log(2/nu2))/(z2*nu1 - z1*nu2)
    
  return 1000*leftPotential          #convert it back to mV


def getEffectiveDielectricConstant(solvent,potential):
  if potential < 0:
    counter_ion = cath 
    co_ion = an
    
  else:
    counter_ion = an
    co_ion = cath
  epsilon = sc.epsilon_0*prop.solvent.dielectric
  ionVolume = 4*np.pi*counter_ion.radiusAng**3/3 * 1e-30
  polarizability_factor = counter_ion.polarizability/3/epsilon/ionVolume
  reducedDielectric = prop.solvent.dielectric * 1000*sc.N_A * ionStericConcentration(counter_ion.radiusAng)*ionVolume *  polarizability_factor
  
  # ~ reducedDielectric = prop.solvent.dielectric *(3*sc.epsilon_0 + 2*counter_ion.polarizability* ionStericConcentration(counter_ion.radiusAng)*1000*sc.N_A)/(3*sc.epsilon_0 - counter_ion.polarizability* ionStericConcentration(counter_ion.radiusAng)*1000*sc.N_A)
  
  if reducedDielectric == 0:
    reducedDielectric = prop.solvent.dielectric
  
  return reducedDielectric

def getStericLayerThickness(potential):
  debyeLength = getDebyeLength(prop.bulkConcentration,prop.electrolyteIons,prop.solvent)
  if potential < 0:
    counter_ion = cath 
    co_ion = an
    
  else:
    counter_ion = an
    co_ion = cath
    
  nu = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
  stericConcentration = ionStericConcentration(counter_ion.radiusAng)
  thresholdPotential = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000
  rho_cap = counter_ion.charge*sc.e*sc.Avogadro*stericConcentration*1000
      
  if(abs(potential/1000) < abs(thresholdPotential)):
    thickness_left = 0
  else:
    thickness_left = debyeLength*np.sqrt(2*nu)*(-1+0.5*nu+np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*potential/1000/sc.k/temperature+np.log(0.5*nu)))
    
  return thickness_left #(thickness_left,thickness_right)

def getChargeDensity(potential):
  debyeLength = getDebyeLength(prop.bulkConcentration,prop.electrolyteIons,prop.solvent)
  if potential < 0:
    counter_ion = cath 
    co_ion = an
    
  else:
    counter_ion = an
    co_ion = cath
    
  nu = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
  stericConcentration = ionStericConcentration(counter_ion.radiusAng)
  thresholdPotential = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000
  rho_bulk = counter_ion.charge*sc.e*sc.Avogadro*prop.bulkConcentration*1000
  
  if(abs(potential/1000) < abs(thresholdPotential)):
    chargeDensity = 0
    # ~ print("Electrode potential is less than the threshold value of {} mV".format(1000*thresholdPotential))
    
  else:
    chargeDensity = -2*rho_bulk*debyeLength*np.sqrt(2/nu)*np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*potential/1000/sc.k/temperature+np.log(0.5*nu))
    
  return chargeDensity


def getElectrostaticEnergy(potentialDifference):
  potential = getChargeBalancePotential(potentialDifference)
  
  if potential < 0:
    counter_ion = cath 
    co_ion = an
      
  else:
    counter_ion = an
    co_ion = cath
    
  potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000 
  
  if abs(potential/1000) < abs(potential_threshold):
    electrostaticFreeEnergy = 0
  else:
    
    stericConcentration = ionStericConcentration(counter_ion.radiusAng)
    rho_cap = counter_ion.charge*sc.e*sc.Avogadro*stericConcentration*1000
          
    electrodeChargeDensity = getChargeDensity(potential)
    reducedPotential = potential - 1000*electrodeChargeDensity*counter_ion.radiusAng*1e-10/sc.epsilon_0/prop.solvent.dielectric
    stericLayer_thickness = getStericLayerThickness(reducedPotential)
    chargeDensity = getChargeDensity(reducedPotential)
    
    
    electrostaticFreeEnergy = (0.5/epsilon)*((1/3)*rho_cap**2*stericLayer_thickness**3 + electrodeChargeDensity*rho_cap*stericLayer_thickness**2 + electrodeChargeDensity**2*stericLayer_thickness) + electrodeChargeDensity**2*counter_ion.radiusAng*1e-10/2/sc.epsilon_0/prop.solvent.dielectric
    # ~ electrostaticFreeEnergy = (0.5/epsilon)*((1/3)*rho_cap**2*stericLayer_thickness**3 + electrodeChargeDensity*rho_cap*stericLayer_thickness**2 + electrodeChargeDensity**2*stericLayer_thickness)
    
    if not prop.singleElectrode:
      rightPotential = potential - potentialDifference
      if rightPotential < 0:
        counter_ion = cath 
        co_ion = an
        
      else:
        counter_ion = an
        co_ion = cath
      
      stericConcentration = ionStericConcentration(counter_ion.radiusAng)
      potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
      rho_cap = counter_ion.charge*sc.e*sc.Avogadro*stericConcentration*1000
          
      electrodeChargeDensity = getChargeDensity(rightPotential)
      reducedPotential = rightPotential - 1000*electrodeChargeDensity*counter_ion.radiusAng*1e-10/sc.epsilon_0/prop.solvent.dielectric
      stericLayer_thickness = getStericLayerThickness(reducedPotential)
      chargeDensity = getChargeDensity(reducedPotential)
    
    
      electrostaticFreeEnergy += (0.5/epsilon)*((1/3)*rho_cap**2*stericLayer_thickness**3 + electrodeChargeDensity*rho_cap*stericLayer_thickness**2 + electrodeChargeDensity**2*stericLayer_thickness) + electrodeChargeDensity**2*counter_ion.radiusAng*1e-10/2/sc.epsilon_0/prop.solvent.dielectric
      
      # ~ electrostaticFreeEnergy += (0.5/epsilon)*((1/3)*rho_cap**2*stericLayer_thickness**3 + electrodeChargeDensity*rho_cap*stericLayer_thickness**2 + electrodeChargeDensity**2*stericLayer_thickness)
        
  return electrostaticFreeEnergy
  

def getElectrostaticEnergyDCA(potentialDifference):
  potential = getChargeBalancePotential(potentialDifference)
  
  if potential < 0:
    counter_ion = cath 
    co_ion = an
      
  else:
    counter_ion = an
    co_ion = cath
    
  potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000 
  
  if abs(potential/1000) < abs(potential_threshold):
    electrostaticFreeEnergy = 0
  else:
    
    stericConcentration = ionStericConcentration(counter_ion.radiusAng)
    rho_cap = counter_ion.charge*sc.e*sc.Avogadro*stericConcentration*1000
          
    electrodeChargeDensity = getChargeDensity(potential)
    reducedPotential = potential - 1000*electrodeChargeDensity*counter_ion.radiusAng*1e-10/sc.epsilon_0/prop.solvent.dielectric
    stericLayer_thickness = getStericLayerThickness(reducedPotential)
    chargeDensity = getChargeDensity(reducedPotential)
    
    
    electrostaticFreeEnergy = (0.5/epsilon)*((1/3)*rho_cap**2*stericLayer_thickness**3 + electrodeChargeDensity*rho_cap*stericLayer_thickness**2 + electrodeChargeDensity**2*stericLayer_thickness) + electrodeChargeDensity**2*counter_ion.radiusAng*1e-10/2/sc.epsilon_0/prop.solvent.dielectric
    # ~ electrostaticFreeEnergy = (0.5/epsilon)*((1/3)*rho_cap**2*stericLayer_thickness**3 + electrodeChargeDensity*rho_cap*stericLayer_thickness**2 + electrodeChargeDensity**2*stericLayer_thickness)
    
    if not prop.singleElectrode:
      rightPotential = potential - potentialDifference
      if rightPotential < 0:
        counter_ion = cath 
        co_ion = an
        
      else:
        counter_ion = an
        co_ion = cath
      
      stericConcentration = ionStericConcentration(counter_ion.radiusAng)
      potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
      rho_cap = counter_ion.charge*sc.e*sc.Avogadro*stericConcentration*1000
          
      electrodeChargeDensity = getChargeDensity(rightPotential)
      reducedPotential = rightPotential - 1000*electrodeChargeDensity*counter_ion.radiusAng*1e-10/sc.epsilon_0/prop.solvent.dielectric
      stericLayer_thickness = getStericLayerThickness(reducedPotential)
      chargeDensity = getChargeDensity(reducedPotential)
    
    
      electrostaticFreeEnergy += (0.5/epsilon)*((1/3)*rho_cap**2*stericLayer_thickness**3 + electrodeChargeDensity*rho_cap*stericLayer_thickness**2 + electrodeChargeDensity**2*stericLayer_thickness) + electrodeChargeDensity**2*counter_ion.radiusAng*1e-10/2/sc.epsilon_0/prop.solvent.dielectric
      
      # ~ electrostaticFreeEnergy += (0.5/epsilon)*((1/3)*rho_cap**2*stericLayer_thickness**3 + electrodeChargeDensity*rho_cap*stericLayer_thickness**2 + electrodeChargeDensity**2*stericLayer_thickness)
        
  return electrostaticFreeEnergy
  

def getEntropicEnergy(potentialDifference):
  potential = getChargeBalancePotential(potentialDifference)
  if potential < 0:
    counter_ion = cath 
    co_ion = an
      
  else:
    counter_ion = an
    co_ion = cath
    
  potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000
    
  if abs(potential/1000) < abs(potential_threshold):
    entropicFreeEnergy = 0
  else:
    electrodeChargeDensity = getChargeDensity(potential)
    reducedPotential = potential - 1000*electrodeChargeDensity*counter_ion.radiusAng*1e-10/sc.epsilon_0/prop.solvent.dielectric
    stericLayer_thickness = getStericLayerThickness(reducedPotential) 
    stericConcentration = ionStericConcentration(counter_ion.radiusAng)
    
    entropicFreeEnergy = 1000*sc.Avogadro*sc.Boltzmann*temperature*stericConcentration*(np.log(stericConcentration/prop.bulkConcentration)-1)*stericLayer_thickness
      
    if not prop.singleElectrode:
      rightPotential = potential - potentialDifference
      if rightPotential < 0:
        counter_ion = cath 
        co_ion = an
        
      else:
        counter_ion = an
        co_ion = cath
      
      electrodeChargeDensity = getChargeDensity(rightPotential)
      reducedPotential = rightPotential - 1000*electrodeChargeDensity*counter_ion.radiusAng*1e-10/sc.epsilon_0/prop.solvent.dielectric
      stericLayer_thickness = getStericLayerThickness(reducedPotential) 
      stericConcentration = ionStericConcentration(counter_ion.radiusAng)
            
      entropicFreeEnergy += 1000*sc.Avogadro*sc.Boltzmann*temperature*stericConcentration*(np.log(stericConcentration/prop.bulkConcentration)-1)*stericLayer_thickness
        
  return entropicFreeEnergy
  
def getStericEnergy(potentialDifference):
  potential = getChargeBalancePotential(potentialDifference)
  if potential < 0:
    counter_ion = cath 
    co_ion = an
      
  else:
    counter_ion = an
    co_ion = cath
    
  potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000 
  
  if abs(potential/1000) < abs(potential_threshold):
    stericFreeEnergy = 0
  else:
    
    stericConcentration = ionStericConcentration(counter_ion.radiusAng)
    
    potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
    rho_cap = counter_ion.charge*sc.e*sc.Avogadro*stericConcentration*1000
    mu_cap = potential_threshold*counter_ion.charge*sc.e
    
    electrodeChargeDensity = getChargeDensity(potential)
    reducedPotential = potential - 1000*electrodeChargeDensity*counter_ion.radiusAng*1e-10/sc.epsilon_0/prop.solvent.dielectric
    stericLayer_thickness = getStericLayerThickness(reducedPotential)
    
    c_eff = 1000*sc.Avogadro*stericConcentration - 1/(counter_ion.radiusAng * 1e-10)
    rho_eff = counter_ion.charge*sc.e*c_eff
    H_eff = stericLayer_thickness + counter_ion.radiusAng*1e-10
    R = counter_ion.radiusAng*1e-10
    print('H_eff',H_eff)
    
    stericFreeEnergy_cda = c_eff*mu_cap*H_eff - rho_eff*(reducedPotential/1000- electrodeChargeDensity*counter_ion.radiusAng*1e-10/2/sc.epsilon_0/prop.solvent.dielectric)*counter_ion.radiusAng*1e-10
    stericFreeEnergy = stericFreeEnergy_cda + (c_eff*mu_cap - rho_eff*(reducedPotential*H_eff/1000-(rho_eff*H_eff**3)/(6*epsilon) -(electrodeChargeDensity*H_eff**2)/(2*epsilon)) -(reducedPotential*R/1000-(rho_eff*R**3)/(6*epsilon) -(electrodeChargeDensity*R**2)/(2*epsilon))) + (1000*sc.Avogadro*stericConcentration*4*np.pi*counter_ion.radiusAng**3*1e-30/3 - 1)*stericLayer_thickness/prop.solvent.radiusAng/1e-10
    # ~ print('added term',(1000*sc.Avogadro*stericConcentration*4*np.pi*counter_ion.radiusAng**3*1e-30/3 - 1)*stericLayer_thickness/prop.solvent.radiusAng/1e-10)
    if not prop.singleElectrode:
      rightPotential = potential - potentialDifference
      if rightPotential < 0:
        counter_ion = cath 
        co_ion = an
        
      else:
        counter_ion = an
        co_ion = cath
        
      stericLayer_thickness = getStericLayerThickness(rightPotential)
      stericConcentration = ionStericConcentration(counter_ion.radiusAng)
      
      potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
      rho_cap = counter_ion.charge*sc.e*sc.Avogadro*stericConcentration*1000
      mu_cap = potential_threshold*counter_ion.charge*sc.e
      
      electrodeChargeDensity = getChargeDensity(rightPotential)
      reducedPotential = rightPotential - 1000*electrodeChargeDensity*counter_ion.radiusAng*1e-10/epsilon
      stericLayer_thickness = getStericLayerThickness(reducedPotential) 
      
      c_eff = 1000*sc.Avogadro*stericConcentration - 1/(counter_ion.radiusAng * 1e-10)
      rho_eff = counter_ion.charge*sc.e*c_eff 
      H_eff = stericLayer_thickness + counter_ion.radiusAng*1e-10
      R = counter_ion.radiusAng*1e-10
      
      stericFreeEnergy_cda = c_eff*mu_cap*H_eff - rho_eff*(reducedPotential/1000- electrodeChargeDensity*counter_ion.radiusAng*1e-10/2/sc.epsilon_0/prop.solvent.dielectric)*counter_ion.radiusAng*1e-10
      stericFreeEnergy = stericFreeEnergy_cda + (c_eff*mu_cap - rho_eff*(reducedPotential*H_eff/1000-(rho_eff*H_eff**3)/(6*epsilon) -(electrodeChargeDensity*H_eff**2)/(2*epsilon)) -(reducedPotential*R/1000-(rho_eff*R**3)/(6*epsilon) -(electrodeChargeDensity*R**2)/(2*epsilon))) + (1000*sc.Avogadro*stericConcentration*4*np.pi*counter_ion.radiusAng**3*1e-30/3 - 1)*stericLayer_thickness/prop.solvent.radiusAng/1e-10
      # ~ stericFreeEnergy += (c_eff*mu_cap - rho_eff*(reducedPotential/1000-(rho_eff*stericLayer_thickness**2)/(6*epsilon)-(electrodeChargeDensity*stericLayer_thickness)/(2*epsilon)))*stericLayer_thickness + (1000*sc.Avogadro*stericConcentration*4*np.pi*counter_ion.radiusAng**3*1e-30/3 - 1)*stericLayer_thickness/prop.solvent.radiusAng/1e-10
      # ~ stericFreeEnergy += (1000*sc.Avogadro*stericConcentration*mu_cap - rho_cap*(rightPotential/1000-(rho_cap*stericLayer_thickness**2)/(6*epsilon)-(electrodeChargeDensity*stericLayer_thickness)/(2*epsilon)))*stericLayer_thickness
    
  return stericFreeEnergy

def getTotalFreeEnergy(potentialDifference):
  potential = getChargeBalancePotential(potentialDifference)
  electrostatic = getElectrostaticEnergy(potentialDifference)
  entropic = getEntropicEnergy(potentialDifference)
  steric = getStericEnergy(potentialDifference)
  total = electrostatic + entropic + steric 
  return total


def getCompositeDiffuseLayerCapacitance(potential):
  
  debyeLength = getDebyeLength(prop.bulkConcentration,prop.electrolyteIons,prop.solvent)
  
  if potential < 0:
    counter_ion = cath 
    co_ion = an
      
  else:
    counter_ion = an
    co_ion = cath
   
  nu = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
  potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
  epsilon = getEffectiveDielectricConstant(prop.solvent,potential)  * sc.epsilon_0
  cap_0 = 100*epsilon/debyeLength
  
  if(abs(potential/1000) < abs(potential_threshold)):
    capacitance = cap_0 * np.cosh(sc.e*abs(counter_ion.charge*potential)/(2000*sc.Boltzmann*temperature))
  
  else:
    capacitance = cap_0/(np.sqrt(2*nu)*np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*potential/(1000*sc.Boltzmann*temperature)-np.log(2/nu)))
  
  if not prop.singleElectrode:
    rightPotential = potential - prop.potentialDifference
    if rightPotential < 0:
      counter_ion = cath 
      co_ion = an
      
    else:
      counter_ion = an
      co_ion = cath
    nu = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
    potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
    epsilon = getEffectiveDielectricConstant(prop.solvent,rightPotential) * sc.epsilon_0
    cap_0 = 100*epsilon/debyeLength
    
    if(abs(rightPotential/1000)<abs(potential_threshold)):
      capacitance2 = cap_0 * np.cosh(sc.e*abs(counter_ion.charge*rightPotential)/(2000*sc.Boltzmann*temperature))
    else:
        capacitance2 = cap_0/(np.sqrt(2*nu)*np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*rightPotential/(1000*sc.Boltzmann*temperature)-np.log(2/nu)))
    total_capacitance = capacitance*capacitance2/(capacitance+capacitance2)
  else:
    capacitance2 = 0
    total_capacitance = 0
  
  return (capacitance, capacitance2, total_capacitance)

def getCompositeDiffuseLayerCapacitanceX(potential):
  
  debyeLength = getDebyeLength(prop.bulkConcentration,prop.electrolyteIons,prop.solvent)
  
  if potential < 0:
    counter_ion = cath 
    co_ion = an
      
  else:
    counter_ion = an
    co_ion = cath
   
  nu = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
  potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
  epsilon = getEffectiveDielectricConstant(prop.solvent,potential)
    
  cap_0 = 100*epsilon/debyeLength
  # ~ print('Left charge density',getChargeDensity(potential), potential)
  
  if(abs(potential/1000) < abs(potential_threshold)):
    capacitance = cap_0 * np.cosh(sc.e*abs(counter_ion.charge*potential)/(2000*sc.Boltzmann*temperature))
  
  else:
    capacitance = cap_0/(np.sqrt(2*nu)*np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*potential/(1000*sc.Boltzmann*temperature)-np.log(2/nu)))
  
  if not prop.singleElectrode:
    rightPotential = potential - prop.potentialDifference
    if rightPotential < 0:
      counter_ion = cath 
      co_ion = an
      
    else:
      counter_ion = an
      co_ion = cath
    nu = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
    potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
    # ~ print('Right charge density',getChargeDensity(rightPotential), rightPotential)
    if(abs(rightPotential/1000)<abs(potential_threshold)):
      capacitance2 = cap_0 * np.cosh(sc.e*abs(counter_ion.charge*rightPotential)/(2000*sc.Boltzmann*temperature))
    else:
        capacitance2 = cap_0/(np.sqrt(2*nu)*np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*rightPotential/(1000*sc.Boltzmann*temperature)-np.log(2/nu)))
    total_capacitance = capacitance*capacitance2/(capacitance+capacitance2)
  else:
    capacitance2 = 0
    total_capacitance = 0
  
  return (capacitance, capacitance2, total_capacitance)

# Ion-size specific Bikerman capacitance for symmetric binary electrolyte taken from equation (25) of https://www.sciencedirect.com/science/article/pii/S0021979717302151 
def getBikermanCapacitance(potential):
  debyeLength = getDebyeLength(prop.bulkConcentration,prop.electrolyteIons,prop.solvent)
  if potential < 0:
    counter_ion = cath 
    co_ion = an
      
  else:
    counter_ion = an
    co_ion = cath
   
  # ~ nu = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
  radius = counter_ion.radiusAng * 1e-10  
  firstNumerator = np.sqrt((counter_ion.charge*sc.e)**2*sc.N_A*epsilon*1000*prop.bulkConcentration/(sc.Boltzmann*temperature))
  # ~ print('firstNumerator',firstNumerator,potential)
  secondNumerator = np.sqrt(8*np.pi*radius**3*sc.N_A*1000*prop.bulkConcentration/3)
  # ~ print('secondNumerator',secondNumerator,potential)
  thirdNumerator = np.sinh(abs(counter_ion.charge*sc.e*potential)/(1000*sc.Boltzmann*temperature))
  # ~ print('thirdNumerator',thirdNumerator,potential)
  firstDenominator = 1 + secondNumerator**2*(np.cosh(counter_ion.charge*sc.e*potential/1000/sc.Boltzmann/temperature)-1)
  # ~ print('firstDenominator',firstDenominator,potential)
  secondDenominator = np.sqrt(np.log(firstDenominator))
  # ~ print('secondDenominator',secondDenominator,potential)
  
  if potential == 0 :
    capacitance = 100*epsilon/debyeLength
  else:
    capacitance = 100*firstNumerator*secondNumerator*thirdNumerator/(firstDenominator*secondDenominator)
  
  if not prop.singleElectrode:
    rightPotential = potential - prop.potentialDifference
    
    if rightPotential < 0:
      counter_ion = cath 
      co_ion = an
      
    else:
      counter_ion = an
      co_ion = cath
    
    radius = counter_ion.radiusAng * 1e-10  
    firstNumerator = np.sqrt((counter_ion.charge*sc.e)**2*sc.N_A*epsilon*1000*prop.bulkConcentration/(sc.Boltzmann*temperature))
    secondNumerator = np.sqrt(8*np.pi*radius**3*sc.N_A*1000*prop.bulkConcentration/3)
    thirdNumerator = np.sinh(abs(counter_ion.charge*sc.e*rightPotential)/(1000*sc.Boltzmann*temperature))
    firstDenominator = 1 + secondNumerator**2*(np.cosh(counter_ion.charge*sc.e*rightPotential/1000/sc.Boltzmann/temperature)-1)
    secondDenominator = np.sqrt(np.log(firstDenominator))
    
    if rightPotential == 0:
      capacitance2 = 100*epsilon/debyeLength
    else:
      capacitance2 = 100*firstNumerator*secondNumerator*thirdNumerator/(firstDenominator*secondDenominator)
    total_capacitance = capacitance*capacitance2/(capacitance+capacitance2)
  else:
    capacitance2 = 0
    total_capacitance = 0
  return (capacitance,capacitance2,total_capacitance)


def getGouyChapmanCapacitance(potential):
  
  debyeLength = getDebyeLength(prop.bulkConcentration,prop.electrolyteIons,prop.solvent)
  
  capacitance = 100*epsilon* np.cosh(abs(an.charge*sc.e*potential/(2000*sc.Boltzmann*temperature)))/debyeLength
  
  if not prop.singleElectrode:
    rightPotential = potential - prop.potentialDifference
    capacitance2 = 100*epsilon* np.cosh(abs(an.charge*sc.e*rightPotential/(2000*sc.Boltzmann*temperature)))/debyeLength
    total_capacitance = capacitance*capacitance2/(capacitance+capacitance2)
  else:
    capacitance2 = 0
    total_capacitance = 0
  return (capacitance, capacitance2, total_capacitance)
  
def getReducedCDLcapacitance(potential):
  debyeLength = getDebyeLength(prop.bulkConcentration,prop.electrolyteIons,prop.solvent)
  
  if potential < 0:
    counter_ion = cath 
    co_ion = an
      
  else:
    counter_ion = an
    co_ion = cath
  
  
  nu = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
  potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
    
  cap_0 = 100*epsilon/debyeLength
  
  reduced_potential = potential/1000 + getChargeDensity(potential)*counter_ion.radiusAng*1e-10/epsilon
  # ~ print('Left charge density',getChargeDensity(potential),potential)#*counter_ion.radiusAng*1e-10/epsilon,reduced_potential,potential)
  # ~ print('Left reduced charge density',getChargeDensity(1000*reduced_potential),reduced_potential)#*counter_ion.radiusAng*1e-10/epsilon,reduced_potential,potential)
  
    
  if(abs(reduced_potential) < abs(potential_threshold)):
    capacitance = cap_0 * np.cosh(sc.e*abs(counter_ion.charge*reduced_potential)/(2*sc.Boltzmann*temperature))
  
  else:
    capacitance = cap_0/(np.sqrt(2*nu)*np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*reduced_potential/(sc.Boltzmann*temperature)-np.log(2/nu)))
  
  if not prop.singleElectrode:
    rightPotential = potential - prop.potentialDifference
    if rightPotential < 0:
      counter_ion = cath 
      co_ion = an
      
    else:
      counter_ion = an
      co_ion = cath
    nu = 2*prop.bulkConcentration/ionStericConcentration(counter_ion.radiusAng)
    potential_threshold = getThresholdPotential(counter_ion,prop.bulkConcentration)/1000  
    reduced_potential = rightPotential/1000 + getChargeDensity(rightPotential)*counter_ion.radiusAng*1e-10/epsilon
    # ~ print('Right charge density',getChargeDensity(rightPotential))
    
    if(abs(reduced_potential)<abs(potential_threshold)):
      capacitance2 = cap_0 * np.cosh(sc.e*abs(counter_ion.charge*reduced_potential)/(2*sc.Boltzmann*temperature))
    else:
        capacitance2 = cap_0/(np.sqrt(2*nu)*np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*reduced_potential/(sc.Boltzmann*temperature)-np.log(2/nu)))
    total_capacitance = capacitance*capacitance2/(capacitance+capacitance2)
  else:
    capacitance2 = 0
    total_capacitance = 0
  
  return (capacitance, capacitance2, total_capacitance)
  

def getIntegralCapacitance(index,V1,V2):
  prop.singleElectrode = False
   
  def cap(potentialDifference):
    potential = getChargeBalancePotential(potentialDifference)
    if index == 1:
      [capacitance,capacitance2,total_capacitance] = getCompositeDiffuseLayerCapacitance(potential)
    elif index == 2:
      [capacitance,capacitance2,total_capacitance] = getBikermanCapacitance(potential)
    elif index == 3:
      [capacitance,capacitance2,total_capacitance] = getReducedCDLcapacitance(potential)
    elif index == 4:
      [capacitance,capacitance2,total_capacitance] = getGouyChapmanCapacitance(potential)
    else:
      raise ValueError("Index should only between 1 and 4. 1 for CDL cap and 2 for Bikerman cap and 3 for reduced CDL cap and 4 for Gouy-Chapman cap")
    
    return [capacitance,capacitance2,total_capacitance]
  N = 10000 + 1
  voltage = np.linspace(V1,V2,N)
  differential_capacitance = [cap(vol)[2] for vol in voltage]
  total_chargeDensity = 0
  for i in range(V1,V2):
    pot = getChargeBalancePotential(i)
    total_chargeDensity += getChargeDensity(pot)
  #integral_cap_charge = total_chargeDensity/(V2/1000-V1/1000)
  integral_capacitance = np.trapz(differential_capacitance,voltage)/(V2-V1)
  
    
  return integral_capacitance #, integral_cap_charge
  




voltageList = np.array(np.linspace(-1000,1000,4000))
    
'''
V = np.linspace(0,1000,500)
ccap = []
bcap = []
F_el = []
F_en = []
F_st = []
F_total = []
for v in V:
  prop.singleElectrode = False
  potential = getChargeBalancePotential(v)
  bcap.append(getBikermanCapacitance(potential)[2])
  ccap.append(getCompositeDiffuseLayerCapacitance(potential)[2])
  F_el.append(getElectrostaticEnergy(potential))
  F_en.append(getEntropicEnergy(potential))
  F_st.append(getStericEnergy(potential))
F_total = [F_el[i] + F_en[i] + F_st[i] for i in range(len(V))]  
print('CDL',getIntegralCapacitance(1,-2700,2700))
print('Bik',getIntegralCapacitance(2,-2700,2700))
# ~ print('GC',getIntegralCapacitance(3,0,1000))
# ~ print('GC',getIntegralCapacitance(6,0,1))
# ~ plt.plot(V,F_el,label='el')
# ~ plt.plot(V,F_en,label='en')
# ~ plt.plot(V,F_st,label='st')
# ~ plt.plot(V,F_total,label='tot')
# ~ plt.legend()
# ~ plt.yscale('log')
# ~ plt.show()
v = 2700
potential = getChargeBalancePotential(v)
print('charge density is ',getChargeDensity(potential)) '''

cdl = []
reduced_cdl = []
bikcap = []
print(getEffectiveDielectricConstant(prop.solvent,500))
print(getEffectiveDielectricConstant(prop.solvent,-500))
for v in voltageList:
  prop.singleElectrode = True
  potential = getChargeBalancePotential(v)
  cdl.append(getCompositeDiffuseLayerCapacitance(potential)[0])
  reduced_cdl.append(getReducedCDLcapacitance(potential)[0])
  bikcap.append(getBikermanCapacitance(potential)[0])
  
plt.plot(voltageList,cdl)#,voltageList,reduced_cdl,voltageList,bikcap)
plt.show()

print('CDL',getIntegralCapacitance(1,0,1000))
# ~ print('Bik',getIntegralCapacitance(2,0,1000))
# ~ print('Reduced CDL',getIntegralCapacitance(3,0,1000))
#print('Energy',getTotalFreeEnergy(1000))

leftChargeDensity = []
rightChargeDensity = []
cdlcap = []
rcdl = []
potentialDifference = np.linspace(0,2700,101)
for potdiff in potentialDifference:
  
  leftChargeDensity.append(getChargeDensity(getChargeBalancePotential(potdiff)))
  rightChargeDensity.append(getChargeDensity(getChargeBalancePotential(potdiff)-potdiff))
  # ~ cdlcap.append(getCompositeDiffuseLayerCapacitance(potdiff)[0])
  rcdl.append(getReducedCDLcapacitance(potdiff)[0])



'''
plt.plot(potentialDifference,rightChargeDensity)
# ~ plt.plot(potentialDifference,cdlcap)
plt.show()
# ~ plt.plot(potentialDifference,cdlcap)
# ~ plt.show()

'''
