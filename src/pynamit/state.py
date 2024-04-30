class state(object):
    """ State of the ionosphere.

    """
    def __init__(self, sha, shc_TJr_to_shc_PFAC, mu0, RI, shc_VB, shc_TB):
        """ Initialize the state of the ionosphere.
    
        """

        self.shc_TJr_to_shc_PFAC = shc_TJr_to_shc_PFAC
        self.mu0 = mu0
        self.RI = RI
        self.sha = sha

        # Initialize the spherical harmonic coefficients
        self.set_shc(VB = shc_VB)
        self.set_shc(TB = shc_TB)

    
    def set_shc(self, **kwargs):
        """ Set spherical harmonic coefficients.

        Specify a set of spherical harmonic coefficients and update the
        rest so that they are consistent. 

        This function accepts one (and only one) set of spherical harmonic
        coefficients. Valid values for kwargs (only one):

        - 'VB' : Coefficients for magnetic field scalar ``V``.
        - 'TB' : Coefficients for surface current scalar ``T``.
        - 'VJ' : Coefficients for magnetic field scalar ``V``.
        - 'TJ' : Coefficients for surface current scalar ``T``.
        - 'Br' : Coefficients for magnetic field ``Br`` (at ``r = RI``).
        - 'TJr': Coefficients for radial current scalar.

        """ 
        valid_kws = ['VB', 'TB', 'VJ', 'TJ', 'Br', 'TJr']

        if len(kwargs) != 1:
            raise Exception('Expected one and only one keyword argument, you provided {}'.format(len(kwargs)))
        key = list(kwargs.keys())[0]
        if key not in valid_kws:
            raise Exception('Invalid keyword. See documentation')

        if key == 'VB':
            self.shc_VB = kwargs['VB']
            self.shc_VJ = self.RI / self.mu0 * (2 * self.sha.n + 1) / (self.sha.n + 1) * self.shc_VB
            self.shc_Br = 1 / self.sha.n * self.shc_VB
        elif key == 'TB':
            self.shc_TB = kwargs['TB']
            self.shc_TJ = -self.RI / self.mu0 * self.shc_TB
            self.shc_TJr = -self.sha.n * (self.sha.n + 1) / self.RI**2 * self.shc_TJ 
            self.shc_PFAC = self.shc_TJr_to_shc_PFAC.dot(self.shc_TJr) 
        elif key == 'VJ':
            self.shc_VJ = kwargs['VJ']
            self.shc_VB = self.mu0 / self.RI * (self.sha.n + 1) / (2 * self.sha.n + 1) * self.shc_VJ
            self.shc_Br = 1 / self.sha.n * self.shc_VB
        elif key == 'TJ':
            self.shc_TJ = kwargs['TJ']
            self.shc_TB = -self.mu0 / self.RI * self.shc_TJ
            self.shc_TJr = -self.sha.n * (self.sha.n + 1) / self.RI**2 * self.shc_TJ 
            self.shc_PFAC = self.shc_TJr_to_shc_PFAC.dot(self.shc_TJr) 
        elif key == 'Br':
            self.shc_Br = kwargs['Br']
            self.shc_VB = self.shc_Br / self.sha.n
            self.shc_VJ = -self.RI / self.mu0 * (2 * self.sha.n + 1) / (self.sha.n + 1) * self.shc_VB
        elif key == 'TJr':
            self.shc_TJr = kwargs['TJr']
            self.shc_TJ = -1 /(self.sha.n * (self.sha.n + 1)) * self.shc_TJr * self.RI**2
            self.shc_TB = -self.mu0 / self.RI * self.shc_TJ
            self.shc_PFAC = self.shc_TJr_to_shc_PFAC.dot(self.shc_TJr) 
            print('check the factor RI**2!')
        else:
            raise Exception('This should not happen')