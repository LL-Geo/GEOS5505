import numpy as np
from .magutils import magcomp,convert_H_to_B_nT
import warnings

#########################################        

def _cotangent(theta):
    return 1.0/np.tan(theta)

def _arccotangent(theta):
    assert theta != 0.0
    return np.arctan2(1,theta)


########################################################################

def tmagpolybodies2D(xzobs,Jinds,Jrems,northxax,bodies):
    """
    Total magnetic field (2D) for a set of polygonal bodies defined by their corners. 
    Takes into account both induced and remnant magnetization.
    Based on Talwani & Heitzler (1964), the default algorithm in Mag2Dpoly package. 
    """
    tmag = np.zeros(xzobs.shape[0])
    for ise in range(bodies.bo.size):
        tmag += tmagpoly2Dgen(xzobs,Jinds[ise],Jrems[ise],northxax,bodies.bo[ise])
    
    return tmag
    

###################################################################################

def tmagpolybodies2Dgen(xzobs,Jinds,Jrems,northxax,bodies,forwardtype):
    """
    Total magnetic field (2D) for a set of polygonal bodies defined by their corners. 
    Takes into account both induced and remnant magnetization.
    Generic version containing four different algorithm formulations ``forwardtype``, passed as a string:
    - "talwani"      --> Talwani & Heitzler (1964)
    - "talwani_red"  --> Talwani & Heitzler (1964) rederived from Kravchinsky et al. 2019
    - "krav"         --> Kravchinsky et al. (2019) rectified by Ghirotto et al. (2020)
    - "wonbev"       --> Won & Bevis (1987)
    """
    tmag = np.zeros(xzobs.shape[0])
    for ise in range(bodies.bo.size):
        tmag += tmagpoly2Dgen(xzobs,Jinds[ise],Jrems[ise],northxax,bodies.bo[ise],forwardtype)
    
    return tmag


###################################################################################

def checkanticlockwiseorder(body):
    """
    Check whether the polygonal body has segments ordered anticlockwise.
    """
    ## https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    ## https://www.element84.com/blog/determining-the-winding-of-a-polygon-given-as-a-set-of-ordered-points
    #
    # Check direction (anti)clockwise for a reference
    #   system like the following:
    #
    #   z
    #  /\ 
    #  |      2
    #  |   1     3
    #  |      4
    #  |
    #  -------------> x
    #
    encarea2=0.0
    for ise in range(body.nsegm):
        x1 = body.ver1[ise,0]
        z1 = body.ver1[ise,1] 
        x2 = body.ver2[ise,0]
        z2 = body.ver2[ise,1]
        encarea2 += (x2-x1)*(z2+z1)

    # anticlockwise -> encarea2 < 0.0
    # clockwise -> encarea2 > 0.0
    if encarea2<0.0:
        anticlockw=True
    else :
        anticlockw=False
    #
    # The reference system for the magnetic anomaly functions
    #   is reversed in z:
    #
    #  -------------> x
    #  |
    #  |      4
    #  |   1     3
    #  |      2
    #  \/
    #  z
    #
    # so, consequently, we flip the direction of
    # clockwise/anticlockwise:   !(anticlockw)   
    return  not(anticlockw)


###############################################################33

def tmagpoly2D(xzobs,Jind,Jrem,northxax,body,forwardtype):
    """
    Total magnetic field (2D) for a polygon defined by its corners. Takes into account both induced and remnant magnetization.
    Based on Talwani & Heitzler (1964), the default algorithm in pyMag2DPoly. 
    """
    aclockw = checkanticlockwiseorder(body)

    if aclockw==False:
        raise ValueError("tmagforwardpoly2D(): vertices *not* ordered anticlockwise. Aborting.")
   
    ##---------------------
    ## Get the angles
   
    ## check modules
    assert Jind.mod >= 0.0
    assert Jrem.mod >= 0.0

    ## `northxax` is the angle between geographic north and the positive x axis
    assert 0.0 <= northxax <= 360.0
    Cnorth = np.deg2rad(northxax)
    
    ## check angles
    assert -90.0 <= Jind.Ideg <= 90.0
    assert -90.0 <= Jrem.Ideg <= 90.0
    
    assert -180.0 <= Jind.Ddeg <= 180.0
    assert -180.0 <= Jrem.Ddeg <= 180.0

    # deg to rad
    Iind = np.deg2rad(Jind.Ideg)
    Dind = np.deg2rad(Jind.Ddeg)
    Irem = np.deg2rad(Jrem.Ideg)
    Drem = np.deg2rad(Jrem.Ddeg)

    # Calculation of Jx and Jz only for case != wonbev
    Jtotx,_,Jtotz = magcomp(Jind.mod,Iind,Dind,Jrem.mod,Irem,Drem,Cnorth)

    ##-------------------------------------
    ## Loop on observation points and segments
    nobs = xzobs.shape[0]
    totfield = np.zeros(nobs)

    ## loop on observation points
    for iob in range(nobs):
        
        xo = xzobs[iob,0]
        zo = xzobs[iob,1]

        ## loop on segments
        tsum = 0.0
        for ise in range(body.nsegm):

            x1 = body.ver1[ise,0]-xo
            z1 = body.ver1[ise,1]-zo
            x2 = body.ver2[ise,0]-xo
            z2 = body.ver2[ise,1]-zo

            tsum += tmagtalwani(x1,z1,x2,z2,Jtotx,Jtotz,Iind,Dind,Cnorth)
                          
        # convert field from A/m to nT
        totfield[iob] = convert_H_to_B_nT( tsum )

    return totfield

#######################################################################################

def tmagtalwani(x1,z1,x2,z2,Jx,Jz,Iind,Dind,C):
    """
    Total magnetic field (2D) for a line segment. Formulas from Talwani & Heitzler (1964).
    """    

    # Quantities for error definitions
    eps = np.finfo(np.float64).eps
    small = 1e4*eps
    anglelim = 0.995*np.pi

    #--------------
    x21 = x2-x1
    z21 = z2-z1
    s = np.sqrt(x21**2+z21**2)

    # Return 0 if two corners are too close
    if s < small :
        return 0.0

    # Get the angles
    theta1 = np.arctan2(z1,x1)
    theta2 = np.arctan2(z2,x2)
    
    # If z21 = 0.0 no contribution    
    if z21 != 0.0 :
        g = -x21/z21
    else :
        return 0.0

    phi = _arccotangent(g)

    thetadiff = theta2-theta1
    # In the case polygon sides cross the x axis
    if thetadiff < -np.pi :
        thetadiff = thetadiff + 2.0*np.pi
    elif thetadiff > np.pi :
        thetadiff = thetadiff - 2.0*np.pi
           
    # Error if a corner is too close to the observation point (calculation continues)
    # and the corner are slightly moved away
    if (x1 < small) and (z1 < small) :
        x1 = small
        z1 = small
        warnings.warn("A corner is too close to an observation point (calculation continues)")
    
    if (x2 < small) and (z2 < small) :
        x2 = small
        z2 = small
        warnings.warn("A corner is too close to an observation point (calculation continues)")

    ########
    r1 = np.sqrt(x1**2+z1**2)
    r2 = np.sqrt(x2**2+z2**2)

    flog = np.log(r2)-np.log(r1)
 
    # Error if the side is too close to the observation point (calculation continues)
    if abs(thetadiff) > anglelim :
        warnings.warn("A polygon side is too close to an observation point (calculation continues)")

    # vertical component
    V = 2.0*np.sin(phi) * (Jx*( (thetadiff)*np.cos(phi) + np.sin(phi)*flog) - \
                      Jz*( (thetadiff)*np.sin(phi) - np.cos(phi)*flog) )

    # horizonatal component
    H = 2.0*np.sin(phi) * (Jx*( (thetadiff)*np.sin(phi) - np.cos(phi)*flog) + \
                      Jz*( (thetadiff)*np.cos(phi) + np.sin(phi)*flog) )

    # Divided by 4π to take into account algorithm formulation in emu units 
    totfield = (1.0/(4.0*np.pi)) * (H*np.cos(Iind)*np.cos(C-Dind) + V*np.sin(Iind))
    
    return totfield 


###########################################################################

def tmagpoly2Dgen(xzobs,Jind,Jrem,northxax,body,forwardtype) :
    """
    Total magnetic field (2D) for a polygon defined by its corners. Takes into account both induced and remnant magnetization.
    Generic version containing four different algorithm formulations ``forwardtype``, passed as a string:
    - "talwani"      --> Talwani & Heitzler (1964)
    - "talwani_red"  --> Talwani & Heitzler (1964) rederived from Kravchinsky et al. 2019
    - "krav"         --> Kravchinsky et al. (2019) rectified by Ghirotto et al. (2020)
    - "wonbev"       --> Won & Bevis (1987)
    """

    ## LOOPING on segments MUST be in ANTI-CLOCKWISE order
    aclockw = checkanticlockwiseorder(body)

    if aclockw==False:
        raise ValueError("tmagforwardpoly2D(): vertices *not* ordered clockwise. Aborting.")
   
    ##---------------------
    ## Get the angles
   
    ## check modules
    assert Jind.mod >= 0.0
    assert Jrem.mod >= 0.0

    ## `northxax` is the angle between geographic north and the positive x axis
    assert 0.0 <= northxax <= 360.0
    Cnorth = np.deg2rad(northxax)
    
    ## check angles
    assert -90.0 <= Jind.Ideg <= 90.0
    assert -90.0 <= Jrem.Ideg <= 90.0
    
    assert -180.0 <= Jind.Ddeg <= 180.0
    assert -180.0 <= Jrem.Ddeg <= 180.0

    # check right forwardtype
    if (forwardtype != "talwani") and (forwardtype != "talwani_red") and (forwardtype != "krav") and (forwardtype != "wonbev"):
        raise ValueError("tmagforwardpoly2D(): [forwardtype] must be 'talwani' or 'talwani_red' or 'krav' or 'wonbev'")
    
    # deg to rad
    Iind = np.deg2rad(Jind.Ideg)
    Dind = np.deg2rad(Jind.Ddeg)
    Irem = np.deg2rad(Jrem.Ideg)
    Drem = np.deg2rad(Jrem.Ddeg)

    # Calculation of Jx and Jz only for case != wonbev
    if forwardtype != "wonbev":
        Jtotx,_,Jtotz = magcomp(Jind.mod,Iind,Dind,Jrem.mod,Irem,Drem,Cnorth)
    
    ##-------------------------------------
    ## Loop on observation points and segments
    nobs = xzobs.shape[0]
    totfield = np.zeros(nobs)

    ## loop on observation points
    for iob in range(nobs):
        
        xo = xzobs[iob,0]
        zo = xzobs[iob,1]

        ## loop on segments
        tsum = 0.0
        for ise in range(body.nsegm):

            x1 = body.ver1[ise,0]-xo
            z1 = body.ver1[ise,1]-zo
            x2 = body.ver2[ise,0]-xo
            z2 = body.ver2[ise,1]-zo

            if forwardtype == "talwani":
                tsum += tmagtalwani(x1,z1,x2,z2,Jtotx,Jtotz,Iind,Dind,Cnorth)

            elif forwardtype == "talwani_red":
                tsum += tmagtalwanired(x1,z1,x2,z2,Jtotx,Jtotz,Iind,Dind,Cnorth)

            elif forwardtype == "krav":
                tsum += tmagkrav(x1,z1,x2,z2,Jtotx,Jtotz,Iind,Dind,Cnorth)

            elif forwardtype == "wonbev":
                tsum += tmagwonbev(x1,z1,x2,z2,Jind.mod,Jrem.mod,Iind,Dind,Irem,Drem,Cnorth)
                          
        # convert field from A/m to nT
        totfield[iob] = convert_H_to_B_nT( tsum )

    return totfield

########################################################################

def tmagkrav(x1,z1,x2,z2,Jtotx,Jtotz,Iind,Dind,Cnorth):
    """
    Total magnetic field (2D) for a line segment. Formulas from Kravchinsky et al (2019) rectified by Ghirotto et al. (2021). 
    """
    
    # Quantities for error definitions
    eps = np.finfo(np.float64).eps
    small = 1e4*eps
    anglelim = 0.995*np.pi

    #--------------
    x21 = x2-x1
    z21 = z2-z1
    tmpgamma = np.sqrt(x21**2+z21**2)

    # Return 0 if two corners are too close
    if tmpgamma < small :
        return 0.0

    # check if den != 0.0
    if tmpgamma!=0.0 :
        gammax = x21 / tmpgamma
        gammaz = z21 / tmpgamma
    else :
        return 0.0    
    
    # if the segment is horizontal it provides no contribution!
    if z21==0.0:
        return 0.0
    
    #------------
    g = x21/z21

    if x1 >= g*z1 :
        delta = 1.0
    elif x1 < g*z1 :
        delta = -1.0

    #--------------------
    # Get the angles
    alpha1 = np.arctan2(delta*(z1+g*x1),(x1-g*z1))
    alpha2 = np.arctan2(delta*(z2+g*x2),(x2-g*z2))
    
    #In the case polygon sides cross the x axis
    alphadiff = alpha2 - alpha1
    if alphadiff < -np.pi :
        alphadiff = alphadiff + 2.0*np.pi
    elif alphadiff > np.pi :
        alphadiff = alphadiff - 2.0*np.pi

    # Error if a corner is too close to the observation point (calculation continues)
    # and the corner are slightly moved away
    if (x1 < small) and (z1 < small) :
        x1 = small
        z1 = small
        warnings.warn("A corner is too close to an observation point (calculation continues)")
    
    if (x2 < small) and (z2 < small) :
        x2 = small
        z2 = small
        warnings.warn("A corner is too close to an observation point (calculation continues)")

        
    r1 = np.sqrt(x1**2+z1**2)
    r2 = np.sqrt(x2**2+z2**2)

    lor21 = np.log(r2)-np.log(r1)

    # Error if the side is too close to the observation point (calculation continues)
    if abs(alphadiff) > anglelim :
        warnings.warn("A polygon side is too close to an observation point (calculation continues)")
  
    #--------------------
    P = gammaz*gammax*lor21 + delta*(gammaz**2)*(alphadiff)
    Q = (gammaz**2)*lor21 - delta*gammax*gammaz*(alphadiff)
    
    ## horizonatl and vertical field components
    H = 1.0/(2.0*np.pi) * (Jtotz*Q + Jtotx*P)
    V = 1.0/(2.0*np.pi) * (Jtotx*Q - Jtotz*P)
    
    ## total field anomaly 
    totfield = V*np.sin(Iind)+H*np.cos(Iind)*np.cos(Cnorth-Dind)

    return totfield

########################################################################################

def tmagtalwanired(x1,z1,x2,z2,Jx,Jz,Iind,Dind,C):
    """ 
    Total magnetic field (2D) for a ribbon. Talwani & Heitzler (1964) modified by Kravchinsky et al. (2019).
    """
    # Quantities for error definitions
    eps = np.finfo(np.float64).eps
    small = 1e4*eps
    anglelim = 0.995*np.pi
    
    #--------------
    x21 = x2-x1
    z21 = z2-z1
    s = np.sqrt(x21**2+z21**2)
    
    # Return 0 if two corners are too close
    if s < small :
        return 0.0

    # if the segment is horizontal it provides no contribution!
    if z21 != 0.0:
        g = -x21/z21
    else:
        return 0.0

    phi = _arccotangent(g)
    
    den1 = x1+z1*_cotangent(phi)
    den2 = x2+z2*_cotangent(phi)
    num1 = z1-x1*_cotangent(phi)
    num2 = z2-x2*_cotangent(phi)

    # Controls on signs of atan argument (abs in den1 and den2)
    #-----------------------
    if den1 < 0.0 : 
        den1 = -den1
        delta = -1.0
        theta1 = np.arctan2(num1,den1)
    else :
        delta = 1.0
        theta1 = np.arctan2(num1,den1)

    if den2 < 0.0 :
        den2 = -den2
        theta2 = np.arctan2(num2,den2)
    else :
        theta2 = np.arctan2(num2,den2)        

    #-----------------------

    # In the case polygon sides cross the x axis
    thetadiff = theta2-theta1
    if thetadiff < -np.pi :
        thetadiff = thetadiff + 2.0*np.pi
    elif thetadiff > np.pi :
        thetadiff = thetadiff - 2.0*np.pi

    # Error if a corner is too close to the observation point (calculation continues)
    # and the corner are slightly moved away
    if (x1 < small) and (z1 < small) :
        x1 = small
        z1 = small
        warnings.warn("A corner is too close to an observation point (calculation continues)")
    
    if (x2 < small) and (z2 < small) :
        x2 = small
        z2 = small
        warnings.warn("A corner is too close to an observation point (calculation continues)")

    ########  
    r1 = np.sqrt(x1**2+z1**2)
    r2 = np.sqrt(x2**2+z2**2)

    flog = np.log(r2)-np.log(r1)
    
    # Error if the side is too close to the observation point (calculation continues)
    if abs(thetadiff) > anglelim :
        warnings.warn("A polygon side is too close to an observation point (calculation continues)")

        
    # vertical component    
    V = 2.0*np.sin(phi) * (Jx * (delta*(thetadiff)*np.cos(phi) + np.sin(phi)*flog)- \
                    Jz * (delta*(thetadiff)*np.sin(phi) - np.cos(phi)*flog) )
 
    # horizontal component
    H = 2.0*np.sin(phi) * (Jx * (delta*(thetadiff)*np.sin(phi) - np.cos(phi)*flog)+ \
                    Jz * (delta*(thetadiff)*np.cos(phi) + np.sin(phi)*flog) )
    
    ## total field anomaly divided by 4π to take into account algorithm formulation in emu units
    totfield = (1.0/(4.0*np.pi)) * (H*np.cos(Iind)*np.cos(C-Dind) + V*np.sin(Iind))
    
    return totfield  


#######################################################################################################################

def tmagwonbev(x1,z1,x2,z2,modJind,modJrem,Iind,Dind,Irem,Drem,C):
    """ 
    Total magnetic field (2D) for a line segment. Formulas from Won & Bevis (1987).
    """

    
    # Quantities for error definitions
    eps = np.finfo(np.float64).eps
    small = 1e4*eps
    anglelim = 0.995*np.pi

    # β is angle among North and profle direction
    betai = Dind - C + np.pi/2
    betar = Drem - C + np.pi/2
    
    #-------------------
    x21 = x2-x1
    z21 = z2-z1

    R  = np.sqrt(x21**2+z21**2)
    # Return 0 if two corners are too close
    if R < small :
        return 0.0

    # Error if a corner is too close to the observation point (calculation continues)
    # and the corner are slightly moved away
    if (x1 < small) and (z1 < small) :
        x1 = small
        z1 = small
        warnings.warn("A corner is too close to an observation point (calculation continues)")
    
    if (x2 < small) and (z2 < small) :
        x2 = small
        z2 = small
        warnings.warn("A corner is too close to an observation point (calculation continues)")

    ###
    r1 = np.sqrt(x1**2+z1**2)
    r2 = np.sqrt(x2**2+z2**2)

    lor21 = np.log(r2) - np.log(r1)

    theta1 = np.arctan2(z1,x1) 
    theta2 = np.arctan2(z2,x2)

    # In the case polygon sides cross the x axis
    if np.sign(z1) != np.sign(z2):
        test = x1*z2 - x2*z1
        if test > 0.0 :
            if z1 >= 0.0 :
                theta2 = theta2 + 2*np.pi
        elif test < 0.0 :
            if z2 >= 0.0 :
                theta1 = theta1 + 2*np.pi
        else :
            return 0.0 

    # Error if the side is too close to the observation point (calculation continues)
    thetadiff = theta1-theta2
    if abs(thetadiff) > anglelim :
        warnings.warn("A polygon side is too close to an observation point (calculation continues)")

    #------------------------
    
    P = (1/R**2)*(x1*z2 - x2*z1)*(((x1*x21 - z1*z21)/(r1**2))- \
                                 ((x2*x21 - z2*z21)/(r2**2)))

    Q = (1/R**2)*(x1*z2 - x2*z1)*(((x1*z21 + z1*x21)/(r1**2))- \
                                 ((x2*z21 + z2*x21)/(r2**2)))
    
    if x21 != 0.0 :
        g = z21/x21
        derZz = ((x21**2)/(R**2))*((theta1 - theta2) + g*lor21) - P
        derZx = -((x21*z21)/(R**2))*((theta1 - theta2) + g*lor21) + Q
        derXz = -((x21**2)/(R**2))*(g*(theta1 - theta2) - lor21) + Q
        derXx = ((x21*z21)/(R**2))*(g*(theta1 - theta2) - lor21) + P
        
    else :

        derZz = -P
        derZx = -((z21**2)/(R**2))*lor21 + Q
        derXz = Q
        derXx = ((z21**2)/(R**2))*(theta1 - theta2) + P  

    # Magnetic strenght components due to induced magnetization
    DELTAHzind = 2.0*modJind*(np.sin(Iind)*derZz + np.sin(betai)*np.cos(Iind)*derZx) 
    DELTAHxind = 2.0*modJind*(np.sin(Iind)*derXz + np.sin(betai)*np.cos(Iind)*derXx) 

    # Magnetic strenght components due to remnant magnetization
    DELTAHzrem = 2.0*modJrem*(np.sin(Irem)*derZz + np.sin(betar)*np.cos(Irem)*derZx) 
    DELTAHxrem = 2.0*modJrem*(np.sin(Irem)*derXz + np.sin(betar)*np.cos(Irem)*derXx) 

    DELTAHztot = DELTAHzind + DELTAHzrem
    DELTAHxtot = DELTAHxind + DELTAHxrem

    ## total field anomaly divided by 4π to take into account algorithm formulation in emu units
    DELTAHtot = -(1.0/(4.0*np.pi))*(DELTAHztot*np.sin(Iind) + DELTAHxtot*np.sin(betai)*np.cos(Iind))
   
    return DELTAHtot

###################################################################################

