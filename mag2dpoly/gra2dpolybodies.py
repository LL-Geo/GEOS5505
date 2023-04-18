import numpy as np
import warnings

#########################################        

def _cotangent(theta):
    return 1.0/np.tan(theta)

def _arccotangent(theta):
    assert theta != 0.0
    return np.arctan2(1,theta)


########################################################################

def tgravpolybodies2D(xzobs,den,bodies):
    """
    Total magnetic field (2D) for a set of polygonal bodies defined by their corners. 
    Takes into account both induced and remnant magnetization.
    Based on Talwani & Heitzler (1964), the default algorithm in Mag2Dpoly package. 
    """
    tmag = np.zeros(xzobs.shape[0])
    for ise in range(bodies.bo.size):
        tmag += tgravpoly2Dgen(xzobs,den[ise],bodies.bo[ise])
    
    return tmag
    

###################################################################################

def tgravpolybodies2Dgen(xzobs,den,bodies,forwardtype):
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
        tmag += tgravpoly2Dgen(xzobs,den[ise],bodies.bo[ise],forwardtype)
    
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




def gravtalwani(x1, z1, x2, z2, rho):
    # Quantities for errors definitions
    small = 1e4 * np.finfo(np.float64).eps
    anglelim = 0.995 * np.pi

    # Definition of Gravitational Constant
    gamma = 6.6743e-11

    # Check if a corner is too close to the observation point (calculation continues)
    # and the corner is slightly moved away
    if abs(x1) < small and abs(z1) < small:
        x1 = np.copysign(small, x1)
        z1 = np.copysign(small, z1)
        print("Warning: A corner is too close to an observation point (calculation continues)")

    if abs(x2) < small and abs(z2) < small:
        x2 = np.copysign(small, x2)
        z2 = np.copysign(small, z2)
        print("Warning: A corner is too close to an observation point (calculation continues)")

    denom = z2 - z1

    # Check on denominator ≠ 0
    if denom == 0.0:
        denom = small

    r1sq = x1**2 + z1**2
    r2sq = x2**2 + z2**2

    theta_diff = np.arctan2(z2, x2) - np.arctan2(z1, x1)

    # In the case polygon sides cross the x axis
    if theta_diff < -np.pi:
        theta_diff = theta_diff + 2.0 * np.pi
    elif theta_diff > np.pi:
        theta_diff = theta_diff - 2.0 * np.pi

    # Error if the side is too close to the observation point (calculation continues)
    if abs(theta_diff) > anglelim:
        print("Warning: A polygon side is too close to an observation point (calculation continues)")

    # Compute terms
    alpha = (x2 - x1) / denom
    beta = (x1 * z2 - x2 * z1) / denom
    term1 = beta / (1.0 + alpha**2)
    term2 = 0.5 * (np.log(r2sq) - np.log(r1sq))

    eq = term1 * (term2 - alpha * theta_diff)

    # Minus sign to take into account opposite looping on polygon segments
    factor = -2.e5 * rho * gamma

    g = factor * eq

    return g


###########################################################################

def tgravpoly2Dgen(xzobs, rho, body, forwardtype):
    # Check if vertices are ordered anticlockwise
    aclockw = checkanticlockwiseorder(body)

    if not aclockw:
        raise ValueError("tgravpoly2D(): vertices *not* ordered anticlockwise. Aborting.")

    nobs = xzobs.shape[0]

    if isinstance(body.ver1, float) and isinstance(rho, float):
        grav = np.zeros(nobs, dtype=float)
    elif not isinstance(body.ver1, float):
        grav = np.zeros(nobs, dtype=type(body.ver1))
    elif not isinstance(rho, float):
        grav = np.zeros(nobs, dtype=type(rho))

    # Check the right forwardtype
    if forwardtype not in ["talwani", "wonbev"]:
        raise ValueError("tgravpoly2Dgen(): [forwardtype] must be 'talwani' or 'wonbev'")

    # Loop on observation points and segments
    nobs = xzobs.shape[0]
    grav = np.zeros(nobs)

    # Loop on observation points
    for iob in range(nobs):
        xo = xzobs[iob, 0]
        zo = xzobs[iob, 1]

        # Loop on segments
        tsum = 0.0
        for ise in range(body.nsegm):
            x1 = body.ver1[ise, 0] - xo
            z1 = body.ver1[ise, 1] - zo
            x2 = body.ver2[ise, 0] - xo
            z2 = body.ver2[ise, 1] - zo

            if forwardtype == "talwani":
                tsum += gravtalwani(x1, z1, x2, z2, rho)
            elif forwardtype == "wonbev":
                tsum += gravwonbev(x1, z1, x2, z2, rho)

        grav[iob] = tsum

    return grav

########################################################################


