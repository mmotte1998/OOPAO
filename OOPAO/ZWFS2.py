# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:42:34 2024

@author: mmotte
"""

import inspect
import multiprocessing
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
import warnings
from OOPAO.Detector import Detector


import numpy as np
import matplotlib.pyplot as plt
from OOPAO.ZWFS import ZWFS
class ZWFS2:
    def __init__(self, tel, diameter:float = None, phase_shift = [-np.pi/2,np.pi/2], flux_ratio:float = 1/2, transmittance:complex = 1, zpf = 4, phase_shift_unit = 'radian', propagation_method = 'FFT'):
        self.telescope = tel
        if isinstance(phase_shift, float):
            phase_shift = [-phase_shift,phase_shift]
        self.zwfs1 = ZWFS(tel, diameter=diameter, phase_shift = phase_shift[0], transmittance=flux_ratio*transmittance, zpf = zpf, phase_shift_unit=phase_shift_unit, propagation_method=propagation_method)
        self.zwfs2 = ZWFS(tel, diameter=diameter, phase_shift = phase_shift[1], transmittance=(1-flux_ratio)*transmittance, zpf = zpf, phase_shift_unit=phase_shift_unit, propagation_method=propagation_method)
        # self.zwfs1.beta_ref = self.zwfs1.beta
        # self.zwfs1.Ib_ref = self.zwfs1.Ib
        # self.zwfs2.beta_ref = self.zwfs2.beta
        # self.zwfs2.Ib_ref = self.zwfs2.Ib
        self.zpf = zpf
        self.wfs_measure()
        self.tag = 'ZWFS'
    def wfs_measure(self, phase_in = None, reconstructor = None, reconstructor_iteration = 1, known_EM = False):
        if reconstructor_iteration<1:
            warnings.warn('Number of iteration ofr the reconstructor should be superior or equal to 1, Taking 1')
            reconstructor_iteration = 1
        # self.b = self.telescope.src.phas
        self.zwfs1.wfs_measure(phase_in, known_EM=known_EM)
        self.zwfs2.wfs_measure(phase_in, known_EM=known_EM)
        if reconstructor is not None:
            self.retrieved_phase = self.reconstructor(reconstructor, iteration= reconstructor_iteration)
        self.img_ZWFS = np.concatenate((self.zwfs1.img_ZWFS,self.zwfs2.img_ZWFS), axis = 1)#(-self.zwfs1.img_ZWFS+self.zwfs2.img_ZWFS)/(2*self.telescope.pupil**2+1-(self.zwfs1.img_ZWFS+self.zwfs2.img_ZWFS))
        # # self.img_ZWFS[self.telescope.pupil==0]=0
        # self.signal = self.img_ZWFS.reshape(self.img_ZWFS.size)
        self.signal = np.append(self.zwfs1.signal, self.zwfs2.signal)
        self.nSignal = self.signal.size
       

    def reconstructor(self, reconstructor = 'atan', iteration = 1):
        for i in range(iteration):
            self.sin_phase = np.concatenate((self.zwfs1.sin_phase,self.zwfs2.sin_phase), axis = 0)
            M11 = -np.sin(self.zwfs1.beta+self.zwfs1.phase_shift/2)
            M11[np.isnan(M11)]=0
            M12 =  np.cos(self.zwfs1.beta+self.zwfs1.phase_shift/2)
            M12[np.isnan(M12)]=0
            M21 = -np.sin(self.zwfs2.beta+self.zwfs2.phase_shift/2)
            M21[np.isnan(M21)]=0
            M22 = np.cos(self.zwfs2.beta+self.zwfs2.phase_shift/2)
            M22[np.isnan(M22)]=0
            self.det_M = M11*M22-M12*M21
            if np.where(self.det_M[self.telescope.pupil==1]==0)[0]>0:
                warnings.warn('Some pixels has a null determinant')
            self.cos_phi = (M22*self.zwfs1.sin_phase-self.zwfs2.sin_phase*M12)/self.det_M
            self.sin_phi = (-M21*self.zwfs1.sin_phase+self.zwfs2.sin_phase*M11)/self.det_M
            if reconstructor == 'linear':
                retrieved_phase = self.sin_phi/self.cos_phi
            if reconstructor == 'atan':
                retrieved_phase = np.arctan2(self.sin_phi,self.cos_phi)
                retrieved_phase[self.zwfs2.telescope.pupil==0]=0
            if iteration>1:
                Psi_b,self.zwfs1.Ib = self.zwfs1.propagation(mask = self.zwfs1.amplitude_mask, phase=retrieved_phase, computing_Ib=True)
                self.zwfs1.beta = np.angle(Psi_b)
                self.zwfs1.sin_phase = self.zwfs1.phase_sin()
                Psi_b,self.zwfs2.Ib = self.zwfs2.propagation(mask = self.zwfs2.amplitude_mask, phase=retrieved_phase, computing_Ib=True)
                self.zwfs2.beta = np.angle(Psi_b)
                self.zwfs2.sin_phase = self.zwfs2.phase_sin()
        return retrieved_phase