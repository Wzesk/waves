import numpy as np


class wave_solver:
    grav = 9.80665  #m^2/s
    waveperiod = 0
    wavefrequency = 0

    #changing values as wave moved initialized for deep water
    initial_amplitude = 0
    initial_depth = 0
    initial_wavelength = 0
    initial_wavenumber = 0
    
    
    def __init__(self,measuredperiod,init_depth, measuredamplitude):
        #set simple properties
        self.initial_amplitude = measuredamplitude
        self.initial_depth = init_depth
        self.waveperiod = measuredperiod
        self.wavefrequency = (2 * np.pi ) / measuredperiod
        
        #calculation other initial properties based on dispersion relationship
        deep_dr = self.dispersion(self.initial_depth)
        self.initial_wavenumber = deep_dr * self.wavefrequency**2 / self.grav
        self.initial_wavelength = (2*np.pi) / self.initial_wavenumber
    
    
    
    def dispersion(self,eff_depth):
        depthratio = self.wavefrequency**2 * eff_depth / self.grav
        assert depthratio > 0
            
        # Guess at dispersion relationship (from https://www.sciencedirect.com/science/article/pii/037838399090032R):
        dr = np.tanh(depthratio ** 0.75) ** (-2.0 / 3.0)
        
        #converge iteratively (strategy based on https://github.com/ChrisBarker-NOAA/wave_utils)
        iter = 0
        f = dr * np.tanh(dr * depthratio) - 1
        while abs(f) > 1e-10:
            qp = dr * depthratio
            fp = qp / (np.cosh(qp) ** 2) + np.tanh(qp)
            dr = dr - f / fp
            f = dr * np.tanh(dr * depthratio) - 1
            iter += 1
            if iter > 200:
                raise RuntimeError("could not converge")
        #set dispersion relationship
        return dr                

    
    def horizontal_velocity(self,sampleheight):
        #horizontal velocity at a given depth -- data is averaged so leaving out final cos term
        u = self.initial_amplitude * self.wavefrequency * (np.cosh(self.initial_wavenumber * (self.initial_depth + sampleheight)) / np.sinh(self.initial_wavenumber * self.initial_depth))
        return u
    
    
    def vertical_velocity(self,sampleheight):
        #vertical velocity at a given depth -- data is averaged so leaving out final cos term
        w = self.initial_amplitude * self.wavefrequency * (np.sinh(self.initial_wavenumber * (self.initial_depth + sampleheight)) / np.sinh(self.initial_wavenumber * self.initial_depth))
        return w
    
    def wave_pressure(self,sampleheight):
        #pressure at a given depth -- data is averaged so leaving out final cos term
        pr = (self.grav * self.initial_depth) + (self.initial_amplitude * self.grav * (np.cosh(self.initial_wavenumber * (self.initial_depth + sampleheight)) / np.cosh(self.initial_wavenumber * self.initial_depth)))
        return pr

    def speed(self,wn,eff_depth):
        #calculates wave phase speed, includes shallow water term
        wave_phase_speed = np.sqrt(self.grav / wn * np.tanh(wn * eff_depth))
        #print("phase speed: ",wave_phase_speed, " at depth: ",eff_depth)
        
        return wave_phase_speed

    def group_speed(self,wn,eff_depth):
        #calculates group speed, including shallow water term
        wave_phase_speed = self.speed(wn,eff_depth)
        wave_group_speed = wave_phase_speed * ( 0.5 + ( ( wn * eff_depth ) / np.sinh( 2 * wn * eff_depth ) ) )
        #print("group speed: ",wave_group_speed," at depth: ", eff_depth)
        
        return wave_group_speed

    def shoaling_coefficient(self, shoal_depth):
        #first updates wave number and carrier wave speed for shoal depth 
        #the calculate shoaling co, as the ratio of wave height at a given shoal depth to an initial deep water wave height.
        
        initial_group_speed = self.group_speed(self.initial_wavenumber, self.initial_depth) #deep group speed

        shoal_dr = self.dispersion(shoal_depth) #dispersion relationship at shoal depth
        shoal_wave_number = shoal_dr * self.wavefrequency**2 / self.grav #wave number at shoal depth
        shoal_group_speed = self.group_speed(shoal_wave_number, shoal_depth) #shoal group speed

        return np.sqrt(initial_group_speed / shoal_group_speed)#wave height ratio
        
    def refraction_coefficient(self,incedent_angle,shoal_depth):
        if incedent_angle == 0:
            return 1

        #angle inputted in degrees, change to radians
        rad_incedent_angle = np.radians(incedent_angle)

        #get initial phase speed and shoal phase speed -- C0 and C
        initial_phase_speed = self.speed(self.initial_wavenumber, self.initial_depth) #deep group speed
        shoal_dr = self.dispersion(shoal_depth) #dispersion relationship at shoal depth
        shoal_wave_number = shoal_dr * self.wavefrequency**2 / self.grav #wave number at shoal depth
        shoal_phase_speed = self.speed(shoal_wave_number, shoal_depth) #shoal group speed
        phase_speed_ratio = shoal_phase_speed/initial_phase_speed # C/C0
       
        if -1 >= (np.sin(rad_incedent_angle) * phase_speed_ratio) >= 1:#checking for acceptable input for arcsin
            print("angle: ",incedent_angle," shoaling: ",phase_speed_ratio," resulted in an arcsin input error")
            return 1
       
        #print("sin of incedent angle: ",np.sin(rad_incedent_angle),"group speed ratio: ",group_speed_ratio)
        
        #calculate new refracted angle   I think this matches --> theta=arcsin(c*sin(theta_deep)/c_deep). 
        rad_refracted_angle = np.arcsin(np.sin(rad_incedent_angle) * phase_speed_ratio)
        
        #new distance between rays
        ray_span_ratio =np.sqrt( np.cos(rad_refracted_angle)/np.cos(rad_incedent_angle))
        
        return ray_span_ratio