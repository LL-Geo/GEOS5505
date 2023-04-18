
import numpy as np

############################################################3

def magcomp(modJind,Iind,Dind,modJrem,Irem,Drem,C):
    """
    Vector addition of magnetic (remnant + induced) components.
    """    
    ## Induced magnetization components
    Jix = modJind*np.cos(Iind)*np.cos(C-Dind)
    Jiy = modJind*np.cos(Iind)*np.sin(C-Dind)
    Jiz = modJind*np.sin(Iind)

    ## Remnant magnetization components
    Jrx = modJrem*np.cos(Irem)*np.cos(C-Drem)
    Jry = modJrem*np.cos(Irem)*np.sin(C-Drem)
    Jrz = modJrem*np.sin(Irem)

    ## Vector addition    
    Jtotx = Jix+Jrx
    Jtoty = Jiy+Jry
    Jtotz = Jiz+Jrz
   
    return Jtotx,Jtoty,Jtotz

##############################################

def convert_H_to_B_nT( H_Am ):
    """
    Convert from the field H (A/m) to B (nT).
    """
    ## permeabilita' del vuoto 
    ## muzero = 4.0 * pi * 10.0^-7
    ## B nanoTesla
    ## B_nT = ( muzero * H_Am ) * 10.0^9
    B_nT =  np.pi * 400.0 * H_Am 
    return B_nT

###############################################

def convert_B_nT_to_H( B_nT ):
    """
    Convert from the field B (nT) to H (A/m).
    """
    H_Am = B_nT / (np.pi * 400.0)
    return H_Am

###############################################
