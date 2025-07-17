# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:33:27 2024

@author: mmotte
"""
import inspect
import multiprocessing
import sys
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
from numpy.fft import fft2,ifft2,fftshift
from .Detector import Detector

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

        
#%%

class ZWFS:
    def __init__(self, tel, diameter:float = 1.06, phase_shift:float = np.pi/2, transmittance = 1, zpf = 4, phase_shift_unit = 'radian', propagation_method = 'FFT'):
        """ ZWFS
        The ZWFS consist in a 2D mask at the focal plane of the telescope to perform the Fourier Filtering of the EM-Field. 
        By default the ZWFS detector is considered to be noise-free (for calibration purposes). For now, the detector parametres has not yet been implemented. 
        
        Parameters
        ----------
        tel: TYPE
            The telescope object to which the ZWFS is associated. This object carries the phase, flux and pupil information.
        diameter: float 
            The diameter of the ZWFS mask in lambda/D: default value 1.06
        phase_shift: float
            The phase shift induce by the ZWFS in radian or lambda depending on chosen phase shift unit: default value pi/2
        transmittance: float
            The portion of flux going into the mask 
        zpf: int 
            In FFT propagation method, it is the Zero padding factor. In MFT propagation method it correspond to the diameter of the mask in fourier plane (ie the resolution of the fourier transform inside the mask): default value 4.
        phase_shift_unit: sting
            If one want to express the depth in lambda or radian. default : radian
        propagation_method: string
            Select the fourier transform propagation method: default FFT but MFT recommanded. 
            
        ************************** PROPAGATING THE LIGHT TO THE ZWFS OBJECT **************************
        The light can be propagated from a telescope object tel through the ZWFS object wfs using the * operator:
        _ tel*wfs
        This operation will trigger:
            _ propagation of the tel.src light through the PWFS detector (phase and flux)
            _ computation of the ZWFS signals

        If the tel.src object is an asterism of sources with the same wavelength, each source is propagated to the ZWFS.
        The resulting intensities are summed incoherently before being integrated by the ZWFS camera.


        """
        if propagation_method != 'FFT' and propagation_method != 'MFT':
            warnings.warn('Unknown propagation method, using FFT')
            self.propagation_method = 'FFT'
        else:
            self.propagation_method = propagation_method
        
        self.telescope = tel
        self.tag = 'ZWFS' 
        # if nSubap is None:
        #     self.nSubap = self.telescope.resolution
        # else:
        #     self.nSubap = nSubap
        self.nSignal = (np.where(self.telescope.pupil==1)[1]).size#self.telescope.pupil.size
        if propagation_method == 'FFT':
            if zpf <=2:
                warnings.warn('Zero-padding factor should be superior to 2 for an FFT propagation method')
            else:
                self.zpf = zpf
        if propagation_method == 'MFT':
            if zpf <=4:
                warnings.warn('Number of pixel inside the mask should be superior to 4 for a MFT propagation method')
            else:
                self.zpf = zpf/self.telescope.resolution
        self.resolution = np.int64(self.zpf*self.telescope.resolution)
        self.transmittance = transmittance
        
        if diameter <=0:
            warnings.warn(f'{diameter} is not a possible diameter, using d=1.06 instead')
            self.diameter = 1.06
                
        self.radius = self.diameter/2
        if phase_shift_unit == 'lambda':
            self.phase_shift = phase_shift*2*np.pi()
        else:
            self.phase_shift = phase_shift
        self.epsilon = (1-np.exp(1j*self.phase_shift))
        
        self.pixel_size = self.telescope.D/self.resolution
        self.Z_bool =self.ZWFS_bool()
        self.Zmask = self.ZWFS_mask()
        
        self.amplitude_mask = self.Z_bool*np.sqrt(self.transmittance)
        # _, self.Ib = self.propagation(mask = self.amplitude_mask, phase = 0)
        self.Ip = self.telescope.pupilReflectivity**2*self.transmittance
        self.wfs_measure(phase_in=np.zeros(self.telescope.pupil.shape), reconstructor='linear', known_EM=True)
        
        self.ref_signal = self.signal
        self.im_ref = self.img_ZWFS
    def ZWFS_bool(self):
        if self.propagation_method == 'MFT':
            Zradius_pixels = self.resolution/2
            center = (self.resolution/2-0.5,self.resolution/2-0.5)
        else:
            Zradius_pixels = self.zpf*self.radius
            center = (self.resolution/2,self.resolution/2)
        
        Y, X = np.ogrid[:self.resolution, :self.resolution]
        Z_bool = (X - center[0])**2 + (Y - center[1])**2 <= Zradius_pixels**2
        return Z_bool
    def ZWFS_mask(self, phase = None):
        if phase is None:
            phase = self.phase_shift
        Zmask =np.zeros((self.resolution,self.resolution), dtype=np.complex128)*np.sqrt(self.transmittance)
        Zmask= (1-(1-np.exp(1j*phase))*self.Z_bool)*np.sqrt(self.transmittance)
        
        return Zmask 
    def propagation(self, mask = None, phase = None, epsilon = None, computing_Ib = False) : 
        if phase is None:
            phase = self.telescope.src.phase
        
        if self.propagation_method == 'MFT':
            # a = time.time()
            Na = self.telescope.resolution
            Nb = np.int64(self.zpf*Na)
            m = self.diameter
            EM = self.telescope.pupilReflectivity*np.exp(1j*phase)
            if epsilon is None: 
                epsilon = self.epsilon
            if computing_Ib:
                _ , EM_det = self.MFT(Na, Nb, m, EM, mask = self.Z_bool)
                EM_det*=np.sqrt(self.transmittance)
            else:
                _ ,EM_det = self.MFT(Na, Nb, m, EM, mask = self.Z_bool)
                EM_det=(EM-(1-np.exp(1j*self.phase_shift))*EM_det)*np.sqrt(self.transmittance)
            EM_det[self.telescope.pupil==0]=np.nan
            IM_detect = np.abs(EM_det)**2
            # b = time.time()
            # print(b-a)
        else:
            if mask is None:
                mask = self.Zmask
            # a = time.time()
            if self.propagation_method != 'FFT':
                warnings.warn(f'{self.propagation_method} is not a possible propagation method, using FFT instead')
            EM = np.zeros((self.resolution,self.resolution), np.complex128)
            
            EM[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2]\
                =self.telescope.pupilReflectivity*np.exp(1j*phase)
            Psi_turb = fftshift(fft2(fftshift(EM))) # 1er prop
            Psi_FP_turb = mask*Psi_turb # PF multiplication par masque
            # print(np.sum(np.abs(Psi_FP_turb)**2))
            EM_det = fftshift(ifft2(fftshift((Psi_FP_turb))))[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2] # PP detecteur
            EM_det[self.telescope.pupil==0]=np.nan
            IM_detect=np.abs(EM_det)**2
            # b = time.time()
            # print(b-a)
        # print(np.sum(IM_detect))
        return EM_det,IM_detect
    def wfs_measure(self, phase_in = None, reconstructor = None, known_EM = False, reconstructor_iteration:int = 1, estimated_phase = 0):
        if reconstructor_iteration<1:
            warnings.warn('Number of iteration ofr the reconstructor should be superior or equal to 1, Taking 1')
            reconstructor_iteration = 1
        if phase_in is not None:
            self.telescope.src.phase = phase_in
        self.EM_detect, self.img_ZWFS = self.propagation()
        self.signal =  self.img_ZWFS[~np.isnan(self.img_ZWFS)].ravel()#np.reshape(self.img_ZWFS, (self.nSignal,))
        if known_EM:
            Psi_b,self.Ib = self.propagation(mask = self.amplitude_mask, computing_Ib=True)
            self.beta = np.angle(Psi_b)
            self.Ip =self.telescope.pupilReflectivity**2*self.transmittance #self.propagation(mask = np.sqrt(self.transmittance),phase = 0)
        else:
            Psi_b,self.Ib = self.propagation(mask = self.amplitude_mask, phase = estimated_phase, computing_Ib=True)
            self.beta = np.angle(Psi_b)
        self.sin_phase = self.phase_sin()
        # if np.size(np.where(np.abs(self.sin_phase)==1)[1])>0:
        #     self.sin_phase/=np.nanmax(np.abs(self.sin_phase))
        if reconstructor is not None:
            self.retrieved_phase = self.reconstructor(reconstructor, iteration= reconstructor_iteration)
        try:
            self.signal-=self.ref_signal
        except:
            None
        # self.signal =  np.reshape(self.img_ZWFS, (self.nSignal,))
    def phase_sin(self, Ib = None, Ip = None, img_ZWFS = None, phase_shift = None):
        if Ib == None:
            Ib = self.Ib
        if Ip == None:
            Ip = self.Ip
        if img_ZWFS == None:
            img_ZWFS = self.img_ZWFS
        if phase_shift == None:
            phase_shift = self.phase_shift
        sin_phase = (img_ZWFS-Ip-4*Ib*(np.sin(phase_shift/2))**2)/(4*np.sqrt(Ib*Ip)*np.sin(phase_shift/2))
        sin_phase[self.telescope.pupil==0]=0
        return sin_phase
        
    def reconstructor(self, reconstr = 'linear', iteration = 1):
        try:
            for i in range(iteration):
                if reconstr == 'linear':
                    phase = 1/np.cos(self.phase_shift/2+self.beta)*(np.sin(self.phase_shift/2+self.beta)+self.sin_phase)
                    phase[self.telescope.pupil==0]=0
                if reconstr == 'asin':
                    phase = self.phase_shift/2+self.beta+np.arcsin(self.sin_phase)
                    phase[self.telescope.pupil==0]=0
                    if np.size(np.where(np.isnan(phase))[0])>0:
                        # phase[np.where(np.isnan(phase))]= self.phase_shift/2+self.beta[np.where(np.isnan(phase))]+np.arcsin(np.mod(self.sin_phase[np.where(np.isnan(phase))], 1))
                        phase[np.where(np.isnan(phase))]=np.sqrt(np.pi)*(1/np.cos(self.phase_shift/2+self.beta[np.where(np.isnan(phase))])*(np.sin(self.phase_shift/2+self.beta[np.where(np.isnan(phase))])+self.sin_phase[np.where(np.isnan(phase))])) #np.nanstd(phase[self.telescope.pupil!=0][~np.isnan(phase[self.telescope.pupil!=0])])
                if iteration>1:
                    Psi_b,self.Ib = self.propagation(mask = self.amplitude_mask, phase=phase, computing_Ib=True)
                    self.beta = np.angle(Psi_b)
                    self.Ip =self.telescope.pupilReflectivity**2*self.transmittance #self.propagation(mask = np.sqrt(self.transmittance),phase = 0)
                    self.sin_phase = self.phase_sin()
            return phase
        except:
            None
    def MFT(self, Na:int, Nb:int, m, EMF_pupil, mask = 1):
        Nb = np.int64(Nb)
        Na = np.int64(Na)
        x = np.zeros((Na, 1))
        u = np.zeros((Nb, 1))
        x[:,0] = (np.arange(0, Na)-(Na)/2+0.5)*1/Na
        u[:,0] = (np.arange(0, Nb)-(Nb)/2+0.5)*m/Nb
        Ft = (m/(Na*Nb)*np.exp(-2*1j*np.pi*(u@x.T))@EMF_pupil@np.exp(-2*1j*np.pi*(x@u.T)))
        iFt = m/(Na*Nb)*np.exp(2*1j*np.pi*(x@u.T))@(Ft*mask)@np.exp(2*1j*np.pi*(u@x.T))
        return Ft, iFt
    
    
    def __mul__(self,obj): 
        if obj.tag=='detector':
            obj._integrated_time+=self.telescope.samplingTime
            obj.integrate(obj._integrated_time)

        else:
            print('Error light propagated to the wrong type of object')
        return -1
#%%

class ZWFS__:
    def __init__(self, tel, diameter:float = None, phase_shift:float = np.pi/2, transmittance = 1, zpf = 4, phase_shift_unit = 'radian'):
        self.telescope = tel
        self.tag = 'ZWFS' 
        self.telescope = tel
        self.nSignal = (np.where(self.telescope.pupil==1)[1]).size#self.telescope.pupil.size
        self.zpf = zpf
        self.resolution = self.zpf*self.telescope.resolution
        self.transmittance = transmittance
        if diameter is None:
            self.diameter = 1.06
        else:
            self.diameter = diameter
  
        self.radius = self.diameter/2
        if phase_shift_unit == 'lambda':
            self.phase_shift = phase_shift*2*np.pi()
        else:
            self.phase_shift = phase_shift
        self.pixel_size = self.telescope.D/self.resolution
        self.ZWFS_mask()
        self.amplitude_mask = np.zeros((self.resolution,self.resolution), dtype = np.complex128)
        self.amplitude_mask[self.Z_bool] = 1*np.sqrt(self.transmittance)+0*1j
        _, self.Ib = self.propagation(mask = self.amplitude_mask, phase = 0)
        # self.Ip = self.telescope.pupilReflectivity**2*self.transmittance
        self.wfs_measure(phase_in=np.zeros(self.telescope.pupil.shape))
        
        self.ref_signal = self.signal
        self.im_ref = self.img_ZWFS
        
    def ZWFS_mask(self):
        Zradius_pixels = self.zpf*self.radius
        center = (self.resolution/2,self.resolution/2)
        Y, X = np.ogrid[:self.resolution, :self.resolution]
        self.Z_bool = (X - center[0])**2 + (Y - center[1])**2 <= Zradius_pixels**2
        self.Zmask =np.ones((self.resolution,self.resolution), dtype=np.complex128)*np.sqrt(self.transmittance)
        self.Zmask[self.Z_bool] = np.exp(self.phase_shift*1j)*np.sqrt(self.transmittance)
        
    def propagation(self, mask = None, phase = None) : 
        if mask is None:
            mask = self.Zmask
        if phase is None:
            EM = np.zeros((self.resolution,self.resolution), np.complex128)
            
            EM[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2]\
                =self.telescope.pupilReflectivity*np.exp(1j*self.telescope.src.phase)
        else:
            EM = np.zeros((self.resolution,self.resolution), np.complex128)
            
            EM[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2]\
                =self.telescope.pupilReflectivity*np.exp(1j*phase)
        Psi_turb = fftshift(fft2(fftshift(EM))) # 1er prop
        Psi_FP_turb = mask*Psi_turb # PF multiplication par masque
        # print(np.sum(np.abs(Psi_FP_turb)**2))
        EM_det = fftshift(ifft2(fftshift((Psi_FP_turb))))[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2] # PP detecteur
        EM_det[self.telescope.pupil==0]=np.nan
        IM_detect=np.abs(EM_det)**2
        # print(np.sum(IM_detect))
        return EM_det,IM_detect
    def wfs_measure(self, phase_in = None, reconstructor = None):
        if phase_in is not None:
            self.telescope.src.phase = phase_in
        self.EM_detect, self.img_ZWFS = self.propagation()
        self.signal =  self.img_ZWFS[~np.isnan(self.img_ZWFS)].ravel()#np.reshape(self.img_ZWFS, (self.nSignal,))
        Psi_b,self.Ib = self.propagation(mask = self.amplitude_mask)
        self.beta = np.angle(Psi_b)
        self.Ip =self.telescope.pupilReflectivity**2*self.transmittance #self.propagation(mask = np.sqrt(self.transmittance),phase = 0)
        self.sin_phase = (self.img_ZWFS-self.Ip-4*self.Ib*(np.sin(self.phase_shift/2))**2)/(4*np.sqrt(self.Ib*self.Ip)*np.sin(self.phase_shift/2))
        self.sin_phase[self.telescope.pupil==0]=0
        # self.sin_phase[np.abs(self.sin_phase)>1]=1
        if reconstructor is not None:
            self.reconstructor(reconstructor)
        try:
            self.signal-=self.ref_signal
        except:
            None
        # self.signal =  np.reshape(self.img_ZWFS, (self.nSignal,))
    
        
        
    def reconstructor(self, reconstr = 'linear'):
        # amplitude_mask = np.zeros((self.resolution,self.resolution))
        # amplitude_mask[self.Z_bool] = 1
        # self.Ib = self.propagation(mask = amplitude_mask) 
        # EM = np.zeros((self.resolution,self.resolution), np.complex_)
        # EM[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2]\
        #     =self.telescope.pupil
        # _, self.Ip = self.propagation(mask = np.ones((self.resolution,self.resolution)), EM = EM)
        
        if reconstr == 'linear':
            phase = 1/np.cos(self.phase_shift/2+self.beta)*(np.sin(self.phase_shift/2+self.beta)+self.sin_phase)
            phase[self.telescope.pupil==0]=0
        if reconstr == 'non-linear':
            phase = self.phase_shift/2+self.beta+np.arcsin(self.sin_phase)
            phase[self.telescope.pupil==0]=0
            
        self.retrieved_phase = phase
    def __mul__(self,obj): 
        if obj.tag=='detector':
            obj._integrated_time+=self.telescope.samplingTime
            obj.integrate(obj._integrated_time)

        else:
            print('Error light propagated to the wrong type of object')
        return -1
#%%
class ZWFS_:
    def __init__(self, tel, diameter:float = None, phase_shift:float = np.pi/2, transmission = 1, zpf = 4, psfCentering = True):
        self.tag = 'ZWFS' 
        self.telescope = tel
        self.nSignal = self.telescope.pupil.size
        self.zpf = zpf
        self.resolution = self.zpf*self.telescope.resolution
        if diameter is None:
            self.diameter = 1.06
        else:
            self.diameter = diameter
  
        self.radius = self.diameter/2
        self.phase_shift = phase_shift      
        self.transmission = transmission
        self.pixel_size = self.telescope.D/self.resolution
        self.Z_mask()
        self.wfs_measure()
 
    def Z_mask(self):
        Zradius_pixels = self.zpf*self.radius
        center = (self.resolution/2,self.resolution/2)
        Y, X = np.ogrid[:self.resolution, :self.resolution]
        self.Z_bool = (X - center[0])**2 + (Y - center[1])**2 <= Zradius_pixels**2
        self.Zmask = np.ones((self.resolution,self.resolution), dtype=np.complex128)
        self.Zmask[self.Z_bool] = self.transmission*np.exp(self.phase_shift*1j)
        # if psfCentering : 
        #     x               = np.arange(-self.telescope.resolution*self.zpf//2,self.telescope.resolution*self.zpf//2,1)
        #     xx,yy           = np.meshgrid(x,x)
        #     R               = xx**2+yy**2
        #     self.Zmask=np.ones(R.shape,dtype=np.complex128)
        #     self.Z_bool = R<=(self.radius*self.zpf)**2
        #     self.Zmask[self.Z_bool] = self.transmission*np.exp(self.phase_shift*1j)
        
        # else:
        #     size_zwfs=np.int64(self.telescope.resolution*self.zpf)
        #     #X=np.round(np.linspace(-self.zpf*self.telescope.resolution,self.zpf*self.telescope.resolution-1,size_zwfs))
        #     X=np.arange(-self.zpf//2*(self.telescope.resolution)-1,self.zpf//2*(self.telescope.resolution)-1,1)
        
        #     [x,y] = np.meshgrid(X,X)
        #     [r,theta]=cart2pol(x,y)
            
        #     self.Zmask = np.ones((size_zwfs,size_zwfs),dtype = complex)
        #     self.Z_bool = r<=np.int64((self.diameter/2)*self.zpf)
        #     self.Zmask[self.Z_bool] = self.transmission*np.exp(self.phase_shift*1j);
    def wfs_measure(self, phase_in = None, reconstructor = None):
        # self.b = self.telescope.src.phas
        if phase_in is not None:

            self.telescope.src.phase = phase_in
            # b - 
        self.compute_EMF_focale()
        self.propagate()
        Ip = np.float64(self.telescope.pupil)
        EMF_mask = np.zeros((self.resolution,self.resolution), dtype=np.complex128)
        EMF_mask[self.Z_bool] = 1
        EMF_mask = EMF_mask*self.EMF
        image_masked = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(EMF_mask))))**2
        self.Ib = image_masked[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2]
        self.Ib[self.telescope.pupil==0]=0
        self.Ieq =  (self.img_ZWFS-Ip-4*self.Ib*np.sin(self.phase_shift/2)**2)/(4*np.sqrt(Ip*self.Ib)*np.sin(self.phase_shift/2))
        self.Ieq[self.telescope.pupil==0]=0
        if reconstructor == 'linear':
            self.retrieved_phase = 1/np.cos(self.phase_shift/2)*(np.sin(self.phase_shift/2)+ self.Ieq)
            self.retrieved_phase[self.telescope.pupil==0]=0
            self.signal =  np.reshape(self.retrieved_phase, (self.nSignal))
            self.retrieved_OPD = self.retrieved_phase*self.telescope.src.wavelength/(2*np.pi)
        if reconstructor == 'non-linear':
            self.retrieved_phase = self.phase_shift/2+np.arcsin(self.Ieq)
            self.retrieved_phase[self.telescope.pupil==0]=0
            self.signal =  np.reshape(self.retrieved_phase, (self.nSignal))
            self.retrieved_OPD = self.retrieved_phase*self.telescope.src.wavelength/(2*np.pi)
        if reconstructor is None:
            self.signal =  np.reshape(self.img_ZWFS, (self.nSignal,))
            self.sin_phase = self.Ieq
            
    def propagate(self):
      
        self.ZWFS_EMF = self.EMF*self.Zmask
        
        self.pupil_ZWFS = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(self.ZWFS_EMF)))[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2]
        
        self.img_ZWFS = np.abs(self.pupil_ZWFS)**2
        self.img_ZWFS[self.telescope.pupil==0]=0
         
    def compute_EMF_focale(self):

        zeropadding_pupil = np.zeros((self.resolution,self.resolution), np.complex128)
        
        zeropadding_pupil[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2]\
            =self.telescope.pupil*np.exp(1j*self.telescope.src.phase)

        self.EMF = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(zeropadding_pupil)))

        self.EMF_pixel_size = self.telescope.src.wavelength/self.telescope.D/self.zpf*180/np.pi*3600
    
    def __mul__(self,obj): 
        if obj.tag=='detector':
            obj._integrated_time+=self.telescope.samplingTime
            obj.integrate(obj._integrated_time)

        else:
            print('Error light propagated to the wrong type of object')
        return -1