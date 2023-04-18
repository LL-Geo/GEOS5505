
import numpy as np
from dataclasses import dataclass

#########################################

class BodySegments2D:
    """
    Class containing the segments of a polygonal body. 
    To create an instance a set of indices have to be passed on.
    """
    def __init__(self,idx1,vertices):
        assert vertices.shape[1]==2
        ## circular shift to get second set of indices
        idx2 = np.roll(idx1,-1)
        ## first set of vertices
        self.ver1 = vertices[idx1,:].view()
        ## second set of vertices
        self.ver2 = vertices[idx2,:].view()
        self.nsegm = self.ver1.shape[0]

#########################################

class MagPolyBodies2D:
    """
    Class containing a set of polygonal bodies described by their segments and all vertices.
    To create an instance, input an array of vectors of indices 
    (of vertices) for each body and the array of all the vertices.
    """
    def __init__(self,bodyindices,allvert):
        assert allvert.shape[1]==2
        ## array of all vertices for all bodies
        self.allvert = allvert
        # bodyindices must be an array of arrays
        assert type(bodyindices[0])==np.ndarray
        N=bodyindices.size
        ## array of bodies defined by their vertices
        self.bo = np.zeros(N,dtype=np.object)
        for i in range(N):
            self.bo[i] = BodySegments2D(bodyindices[i],self.allvert)

#########################################

@dataclass
class MagnetizVector: 
    """
    Class containing the components of a magnetization vector, 
    i.e., module, inclination and declination angles.
    """  
    mod: float
    Ideg: float
    Ddeg: float

## mv = MagnetizVector(2.0,3.2,4.9)
#########################################
