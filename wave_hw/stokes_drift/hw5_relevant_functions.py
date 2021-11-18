#pulling out relevant functions from HW and adding notes from solution

def stokes_drift(self,z=0):
    #define variables
    a=self.initial_amplitude
    w=self.wavefrequency
    k=self.initial_wavenumber
    h=self.initial_depth

    #calculate langrangian stokes drift at given Z value
    drift = (a**2)*w*k*((sp.cosh(2*k*(z+h)))/(2*(sp.sinh(k*h)**2)))
    return drift


def integrated_mass_flux(self,depth):
        #calculated integrated mass flux at different depths
        #this incorporates shoaling
        #todo: add refraction

        #initial properties of wave
        a=self.initial_amplitude
        g=self.grav
        rho=self.rho
        w=self.wavefrequency
        k=self.initial_wavenumber

        #properties updated to shoal depth
        c=self.speed(self.initial_wavenumber,depth)
        a = a * self.shoaling_coefficient(depth)
        k=self.shoal_wavenumber #corrected so using shoal wave number

        ##calculate flux
        flux = 0.5 *((rho*(a**2)*g)/c) #my original expression
        #flux = 0.5 *((rho*(a**2)*g)*k/w) #alternative version from HW solution

        #calculate depth averaged current from mass flux function added based on HW solution
        current=flux/(rho*depth)

        return flux,current



##functions for calculating from spectrum

def stoke_drift_from_spectrum(a,f,depth,sample_depth):
    #get the drift for a specific wave frequency within a spectrum
    wave_period = 1/f
    spec_wave = wave_solver(wave_period,depth,a)
    drift = spec_wave.stokes_drift(z=sample_depth)


    ##############
    ##from solution
        

    ##############
    return drift

def integrated_stoke_drift(a_array,f_array,depth,sample_depth):
    #get a drift spectrum from the sea surface spectrum -- need to rework function names
    drift_spectrum = []
    for i in range(len(a_array)):
        drift_spectrum.append(stoke_drift_from_spectrum(a_array[i],f_array[i],depth,sample_depth))

    return drift_spectrum