# -*- coding: utf-8 -*-
"""
DCE signal simulation 

class of functions used to simulate the DCE signal from given a set of parameters. 

ouri cohen - Tue Jan 28 15:23:49 2020
 
"""

import numpy as np
import math
import matplotlib.pyplot as plt

class DCE():
    def __init__(self,tissue_dict,acq_dict):
        """
        initialize the DCE class with the input parmeters
    
        Returns
        -------
        None.

        """            
        self.parse_dce_params(tissue_dict)
        self.parse_acquisition_params(acq_dict) 
        
        self.pi = math.pi
        
        return 
    
    def parse_dce_params(self,tissue_dict):
        """
        takes in a tissue dictionary and parses it into vectors for use in generating the time series. 
        
        Parameters
        ----------
        tissue_dict : dict
            dictionary of dce parameters used in the reconstruction

        Returns
        -------
        None.

        """
        self.Ktrans = tissue_dict['Ktrans']
        self.ve = tissue_dict['ve']
        self.vp = tissue_dict['vp']
        self.T1 = tissue_dict['T1']         
        self.BAT = tissue_dict['BAT'] 
        self.B1 =  tissue_dict['B1'] 
        
        return
    
    def parse_acquisition_params(self,acq_dict):
        """
        Parse the parameters used in the acquisition 
        """
        self.schedule_length = acq_dict['schedule_length'] # number of images in the time series
        self.delta_t = acq_dict['delta_t'] # temporal resolution of the acquisition, in sec 
        self.r1 = 1e-3 * acq_dict['r1'] # relaxivity of gadolinium converted from 1/(mM * s) to 1/(mM * ms) 
        self.FA = acq_dict['FA'] # flip angle 
        self.TR = acq_dict['TR'] # ms        

        return

    def generate_parker_aif(self):
        """
        generates an AIF curve from parameters of advanced cancer population-based 
        AIF as described in Parker et al "Experimentally‐derived functional form 
        for a population‐averaged high‐temporal‐resolution arterial input function 
        for dynamic contrast‐enhanced MRI", MRM 2006;56:993-1000

        Returns
        -------
        Cp: (array) AIF waveform

        """     
        tsec = self.delta_t * np.arange(0,self.schedule_length,1) # assumes 1 image/sec 
        t_min = tsec/60 # minutes
        A = [0.809, 0.330] # scaling const, mmol*min
        T = [0.17046, 0.365] # centers, min
        sigma = [0.0563, 0.132] # gaussian width, min
        alpha = 1.050 # amplitude constant of exponential, mmol
        beta = 0.1685 # decay constant of exponential, 1/min
        s = 38.078 # sigmoid width, 1/min
        tau = 0.483 # sigmoid center, min
        Hct = 0.42 # hematocrit
        C = math.sqrt(2*self.pi)
                
        # equation [1] in Parket et al                 
        exp_term0 = ( A[0] / (C*sigma[0]) ) * np.exp( -(t_min-T[0]) * (t_min-T[0]) / (2*sigma[0]**2) )
        exp_term1 = ( A[1] / (C*sigma[1]) ) * np.exp( -(t_min-T[1]) * (t_min-T[1]) / (2*sigma[1]**2) )
        sig_term = alpha * np.exp(-beta*t_min) / (1 + np.exp( -s * (t_min-tau) ) )
        
        Cb = exp_term0 + exp_term1 + sig_term                        
        
        # according to Parker et al to get the actual Cp we need to divide by the 
        # hematocrit 
        self.Cp = Cb/(1-Hct)        
              
        return
        
    def apply_bolus_arrival_time_delay(self):
        """
        apply the bolus arrival time delay to the AIF by shifting the 
        AIF (i.e. Cp) curve. The signal is shifted and zero-padded.
        """
        shift = np.round(self.BAT).astype(int) # The BAT needs to be integers so we just round it to the nearest one 
        shifted_Cp = np.zeros((self.Cp.shape[0],self.Ktrans.shape[0])) #pre-allocate N x P array for each of the P tissue combinations        
        for jj in range(len(shift)):            
            curr_indx =  shift[jj].item()
            shifted_Cp[curr_indx:,jj] = self.Cp[:-curr_indx] 
        
        # write back to Cp 
        self.Cp = shifted_Cp

        return

    def expConv(self, A, B):
        # Returns f = A ConvolvedWith exp(-B t)        
        # Adapted from: https://github.com/davidssmith/DCEMRI.jl/blob/master/src/models.jl
        
        f = np.zeros(len(A))
        time = (self.delta_t/60) * np.arange(len(A)) # convert to minutes

        for tt in range(len(A)-1):
            x = B * ( time[tt+1] - time[tt] )
            dA = ( A[tt+1] - A[tt] ) / x
            E = np.exp(-x)
            E0 = 1 - E
            E1 = x - E0
            f[tt+1] = E*f[tt] + A[tt] * E0 + dA * E1
        
        f = f / B
        
        return f 

    def tofts_model(self):
        # Extended Tofts Model 
        # Adapted from: https://github.com/davidssmith/DCEMRI.jl/blob/master/src/models.jl

        Ct = np.zeros((len(self.Ktrans),self.schedule_length))
        
        for ii in range(len(self.Ktrans)):
            kep = self.Ktrans[ii]/self.ve[ii]
            Ct[ii,:] = self.vp[ii]*self.Cp[:,ii] + self.Ktrans[ii] * self.expConv(self.Cp[:,ii], kep)

        self.Ct = Ct

        return Ct


    def convert_concentration_to_signal(self):
        """
        convert the concentration time curve into a signal time curve using the input T1 and Gd relaxivitiy    
        """
        St = np.zeros((self.Ct.shape)) # pre-allocate
        FA = self.B1 * self.FA*np.pi/180 # radians
        sinFA = np.sin(FA)
        cosFA = np.cos(FA)
        A = np.zeros(self.Ct.shape[1])
        
        for ii in range(len(self.Ktrans)):
            for tt in range(self.Ct.shape[1]):
                # note that the units of r1 is 1/(mM * ms) since we converted it in the parse_acquisition_params()
                R1 = 1/self.T1[ii] # 1/ms, the index ii is over the different tissue values 
                TR = self.TR[0] # ms
                A[tt] =  TR * ( R1 + (self.r1 * self.Ct[ii,tt]) )                
                expA = np.exp(-A[tt])
                
                St[ii,tt] = sinFA[ii] * (1 - expA )/ (1- cosFA[ii] * expA)
    
        return St

        
    def run_sequence(self):
        self.generate_parker_aif()
        self.apply_bolus_arrival_time_delay()
        traj = self.tofts_model()        
        traj = self.convert_concentration_to_signal() 

        traj = np.transpose(traj,(1,0)) # transpose so time-series is in the first dimension

        return traj
        

def main():    
    
    tissue_dict = {'Ktrans':np.array([0.1,0.05272]),
                   'vp':np.array([0.01,0.005304]),
                   've':np.array([0.3,0.1577])}
    acq_dict = {'schedule_length': 48,
                'delta_t': 5}
        
    dce_data = DCE(tissue_dict,acq_dict)
    traj = dce_data.run_sequence()
    plt.plot(range(schedule_length),dce_data.Ct,range(schedule_length),dce_data.Cp,'r')
    #plt.plot(range(schedule_length),dce_data.Ct,'b')
    plt.show()    
if __name__ == '__main__':
    main()
        

