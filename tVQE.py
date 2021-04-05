from __future__ import print_function
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import scipy.sparse
import scipy.sparse.linalg
import sys
import math



    

class tUCCSD():
    """
    Assumes that we have some operator, which will be applied in a specific way G*psi or exp{iG}*psi to create a trial state
    """
    def __init__(self,H, G, ref, params):
        """
        _H      : sparse matrix
        _G_ops  : list of sparse matrices - each corresponds to a variational parameter
        _ref    : reference state vector
        _params : initialized list of parameters
        """

        self.H = H
        self.G = G
        self.ref = cp.deepcopy(ref)
        self.curr_params = params
        self.dim = self.H.shape[0] 
        
        self.iter = 0
        self.energy_per_iteration = []
        self.psi_norm = 1.0
        self.n_procs = 1

    def expm_multiply(self,a,g):
        return  math.cos(a)*scipy.sparse.identity(self.dim,float)+math.sin(a)*g
    
    def energy(self,parameters):
        new_state = self.prepare_state(parameters)
        assert(new_state.transpose().conj().dot(new_state).toarray()[0][0]-1<0.0000001)
        energy = new_state.transpose().conj().dot(self.H.dot(new_state))[0,0]
        assert(np.isclose(energy.imag,0))
        self.curr_energy = energy.real
        return energy.real

    def prepare_state(self,parameters):
        """ 
        Prepare state:
        exp{A1}exp{A2}exp{A3}...exp{An}|ref>
        """
        new_state = self.ref * 1.0
        for k in reversed(range(0, len(parameters))):
            new_state = self.expm_multiply(parameters[k],self.G[k]).dot(new_state)
        return new_state
    
    
    def check_impact(self,pos1,pos2,op):
        self.curr_params.insert(pos1,0.01)
        self.G.insert(pos1,op)
        self.curr_params.insert(pos2,-0.01)
        self.G.insert(pos2,op)
        e1=self.energy(self.curr_params)
        self.curr_params[pos2]=0.01
        self.curr_params[pos1]=-0.01
        e2=self.energy(self.curr_params)
        del self.G[pos2]
        del self.curr_params[pos2]
        del self.G[pos1]
        del self.curr_params[pos1]
        return min(e1,e2)

           
        
        
    def derivative(self,parameters,pos,op):
        #print('parameters!:', parameters)
        derived = self.ref * 1.0
        for k in reversed(range(0, len(parameters))):
            derived = self.expm_multiply(parameters[k],self.G[k]).dot(derived)
            if k==pos:
                derived=np.complex(0,1)*op.dot(derived)
        bra=self.prepare_state(parameters).transpose().conj()
        
        return abs(bra.dot(self.H.dot(derived)).A[0][0]-(derived.transpose().conj().dot(self.H.dot(bra.transpose().conj()))).A[0][0])
        
    
    
    def gradient(self,parameters):
        
        grad = np.zeros(shape=len(parameters),dtype=float)
        ket = self.prepare_state(parameters)
        hbra = ket.transpose().conj().dot(self.H)
        
        for term in range(len(parameters)):
            grad[term]=2*hbra.dot(self.G[term]).dot(ket).toarray()[0][0].real
            hbra=self.expm_multiply(-parameters[term],self.G[term]).dot(hbra.transpose().conj()).transpose().conj()
            ket=self.expm_multiply(-parameters[term],self.G[term]).dot(ket)
        return grad


    
    def callback(self,x):
        try:
            err = np.sqrt(np.vdot(self.der, self.der))
            #print(" Iter:%4i Current Energy = %20.16f Gradient Norm %10.1e Gradient Max %10.1e" %(self.iter,
                #self.curr_energy.real, err, np.max(np.abs(self.der))))
        except:
            #print(" Iter:%4i Current Energy = %20.16f Psi Norm Error %10.1e" %(self.iter,
                #self.curr_energy.real, 1-self.psi_norm))
            pass
        self.iter += 1
        self.energy_per_iteration.append(self.curr_energy)
        sys.stdout.flush()




