import numpy as np

class CO2toCO():
    """Kinetic model of CO2 to CO for acid reaction"""
    def __init__(self, pCO2, pCO, xH2O, cHp, k1, K1, k2, K2, k3, K3):
        self.A = np.zeros((3,3))
        self.b = np.zeros(3)
        self.pCO2 = pCO2
        self.pCO = pCO
        self.xH2O = xH2O
        self.cHp = cHp
        self.k1 = k1
        self.K1 = K1
        self.k2 = k2
        self.K2 = K2
        self.k3 = k3
        self.K3 = K3
        self._update_A()
 
    def _update_A(self):
        pCO2 = self.pCO2
        pCO = self.pCO
        xH2O = self.xH2O
        cHp = self.cHp
        k1 = self.k1
        K1 = self.K1
        k1r = k1 / K1
        k2 = self.k2
        K2 = self.K2
        k2r = k2 / K2
        k3 = self.k3
        K3 = self.K3
        k3r = k3 / K3
        #print 'coverages', thetaCOOH, thetaCO, thetaFree
        if 1:
            A11 = - (k1r + k2 * cHp)
            A12 = k2r * xH2O
            A13 = k1 * pCO2 * cHp
            A21 = k2 * cHp
            A22 = -(k2r * xH2O + k3)
            A23 = k3r * pCO
            A31 = 1. #k1r
            A32 = 1. #k3
            A33 = 1. #-(k1 * pCO2 * cHp + k3r * pCO)
            self.A = np.array([[A11, A12, A13],
                               [A21, A22, A23],
                               [A31, A32, A33]])
            self.b[-1] = 1.

    def solve2(self):
        """ coverages: [COOH, CO, *]
        """
        pCO2 = self.pCO2
        pCO = self.pCO
        xH2O = self.xH2O
        cHp = self.cHp
        k1 = self.k1 * pCO2 * cHp
        K1 = self.K1
        k1r = k1 / K1
        k2 = self.k2 * cHp
        K2 = self.K2
        k2r = k2 / K2 * xH2O
        k3 = self.k3
        K3 = self.K3
        k3r = k3 / K3 * pCO
        #thetas3 = np.array([ 1. - A - A*B, C - A*B, A ])
        if 0: # works
            B = ( k1r + k2 + k1 ) / ( k1r + k2 + k2r )
            C = ( k1r + k2 ) / ( k1r + k2 + k2r )
            A1 = k2 / ( k2 + k2r + k3 )  -  ( k1r + k2 ) / ( k1r + k2 + k2r )
            A2 = ( k2 - k3r ) / ( k2 + k2r + k3 )  - ( k1r + k2 + k1 ) / ( k1r + k2 + k2r )
            A = A1 / A2
            M = np.array([[1., 1., 1.],
                          [0., 1., B],
                          [0., 0., 1.]])
            b = np.array([1.,C,A])
            thetas3 = np.linalg.solve(M, b)
        if 1: # 
            B = ( k1r + k2 + k1 ) / ( k1r + k2 + k2r )
            C = ( k1r + k2 ) / ( k1r + k2 + k2r )
            A1 = ( k2 * k3 + k1r * k2r + k1r * k3 )
            A2 = ( k3r * k1r + k3r * k2 + k3r * k2r                     
                   + k2 * k3 + k1r * k2r + k1r * k3 
                   + k1 * k2 + k1 * k2r + k1 * k3)
            A = A1 / A2
        thetas3 = np.array([ 1. - A - C + A*B, C - A*B, A ])
        rate3 = self._get_rate3(thetas3)
        return thetas3, rate3

    def solve(self):
        thetas3 = np.linalg.solve(self.A, self.b)
        rate3 = self._get_rate3(thetas3)
        return thetas3, rate3

    def update_params(self):
        ### Not Implemented ... just create a new instance
        pass
    
    def _get_rate3(self, thetas, verbose=0):
        pCO2 = self.pCO2
        pCO = self.pCO
        xH2O = self.xH2O
        cHp = self.cHp
        k1 = self.k1
        K1 = self.K1
        k1r = k1 / K1
        k2 = self.k2
        K2 = self.K2
        k2r = k2 / K2
        k3 = self.k3
        K3 = self.K3
        k3r = k3 / K3
        thetaCOOH, thetaCO, thetaFree = thetas
        r1  = k1 * pCO2 * cHp * thetaFree
        r1r = k1r * thetaCOOH
        r2  = k2 * thetaCOOH * cHp
        r2r = k2r * thetaCO * xH2O
        r3  = k3 * thetaCO
        r3r = k3r * pCO * thetaFree
        netrate_1 = r1-r1r
        netrate_2 = r2-r2r
        netrate_3 = r3-r3r
        if verbose:
            print('coverages', thetaCOOH, thetaCO, thetaFree)
            print('rate constants', k1, k2, k3)
            print('rates',  netrate_1, netrate_2, netrate_3)
            print('dCOOHdt', r1 -r1r -r2 +r2r)
            print('dCOdt', r2 +r3r -r2r -r3)
            print('A', self.A)
            print('b', self.b)
            print('A * thetas', np.dot(self.A, thetas))
            print(self.A[0,0]*thetas[0] + self.A[0,1]*thetas[1] + self.A[0,2] * thetas[2])
        return np.array([netrate_1, netrate_2, netrate_3])
