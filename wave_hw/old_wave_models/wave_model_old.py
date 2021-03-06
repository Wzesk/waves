import numpy as np
import sympy as sp

class wave_solver:
    rho = 1025 #density
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

    def amp_from_horizontal(self,horizonal_velocity,sampleheight):
        #for spectral analysis, calculate amplitude based on horizontal velocity spectra

        amp = horizonal_velocity / ( self.wavefrequency * (sp.cosh(self.initial_wavenumber * (self.initial_depth + sampleheight)) / sp.sinh(self.initial_wavenumber * self.initial_depth)))
        return amp

    def amp_from_vertical(self,vertical_velocity,sampleheight):
        #for spectral analysis, calculate amplitude based on vertical velocity spectra

        amp = vertical_velocity / (self.wavefrequency * (sp.sinh(self.initial_wavenumber * (self.initial_depth + sampleheight)) / sp.sinh(self.initial_wavenumber * self.initial_depth)))
        return amp

    def horizontal_velocity(self,sampleheight):
        #horizontal velocity at a given depth -- data is averaged so leaving out final cos term
        u = self.initial_amplitude * self.wavefrequency * (sp.cosh(self.initial_wavenumber * (self.initial_depth + sampleheight)) / sp.sinh(self.initial_wavenumber * self.initial_depth))
        return u
    
    
    def vertical_velocity(self,sampleheight):
        #vertical velocity at a given depth -- data is averaged so leaving out final cos term
        w = self.initial_amplitude * self.wavefrequency * (sp.sinh(self.initial_wavenumber * (self.initial_depth + sampleheight)) / sp.sinh(self.initial_wavenumber * self.initial_depth))
        return w
    
    def wave_pressure(self,sampleheight):
        #pressure at a given depth -- data is averaged so leaving out final cos term

        ##initial calculation
        # pr = (self.grav * (self.initial_depth-sampleheight)) + (self.initial_amplitude * self.grav 
        # * (np.cosh(self.initial_wavenumber * 
        # (self.initial_depth + sampleheight)) / np.cosh(self.initial_wavenumber * self.initial_depth)))
        
        #based solution handed back in class
        pr = (self.initial_amplitude * self.grav * self.rho) * (sp.cosh(self.initial_wavenumber * 
        (self.initial_depth + sampleheight)) / sp.cosh(self.initial_wavenumber * self.initial_depth))

        return pr

    def speed(self,wn,eff_depth):
        #calculates wave phase speed, includes shallow water term
        wave_phase_speed = sp.sqrt(self.grav / wn * sp.tanh(wn * eff_depth))
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



def spectrum_horizontal_to_pressure(uu,vv,f,depth):
    grav = 9.80665  #m^2/s
    rho = 1025 #density
    w = f #wave frequency
    dr = spectrum_dispersion(depth,w,grav)
    k = dr * w**2 / grav
    pressure = (uu+vv)*((rho/k*w)**2)
    return pressure

def spectrum_dispersion(eff_depth,wavefrequency,grav):
    depthratio = wavefrequency**2 * eff_depth / grav
    if not depthratio > 0:
        depthratio = 0.01
        
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



def uu_to_pp(uxx,f,depth,sample_depth):
    if f>0 and uxx>0 and f<100:
        wave_period = 1/f
        #print("calculated period: ",wave_period)
        spec_wave = wave_solver(wave_period,depth,1)#starting with dummy amplitude of 1
        spec_wave.initial_amplitude = spec_wave.amp_from_horizontal(uxx,sample_depth) # set a calculated amplitude
        pressure = spec_wave.wave_pressure(sample_depth)
        #print("calculated amp: ",spec_wave.initial_amplitude)
        return pressure
    elif():
        print("f<0")
        wave_period = 1
        spec_wave = wave_solver(wave_period,depth,1)#starting with dummy amplitude of 1
        spec_wave.initial_amplitude = spec_wave.amp_from_horizontal(1,sample_depth) # set a calculated amplitude
        pressure = spec_wave.wave_pressure(sample_depth)
        return pressure

def ww_to_pp(wxx,f,depth,sample_depth):
    if f>0 and wxx>0 and f<100:
        wave_period = 1/f
        #print("calculated period: ",wave_period)
        spec_wave = wave_solver(wave_period,depth,1)#starting with dummy amplitude of 1
        spec_wave.initial_amplitude = spec_wave.amp_from_vertical(wxx,sample_depth) # set a calculated amplitude
        pressure = spec_wave.wave_pressure(sample_depth)
        #print("calculated amp: ",spec_wave.initial_amplitude)
        return pressure
    elif():
        print("f<0")
        wave_period = 1
        spec_wave = wave_solver(wave_period,depth,1)#starting with dummy amplitude of 1
        spec_wave.initial_amplitude = spec_wave.amp_from_vertical(wxx,sample_depth) # set a calculated amplitude
        pressure = spec_wave.wave_pressure(sample_depth)
        return pressure

def up_crossings(measured_data): #using pressure data
    data_mean = np.mean(measured_data)
    up_crossings = []
    for i in range(len(measured_data)-1): # check to see if value crosses mean at each step.  If it goes from neg to pos it is an upcrossing
        prev_sample = measured_data[i]-data_mean #find out if prev sample was above or below mean
        next_sample = measured_data[i+1]-data_mean #find out if next sample was above or below mean
        if(prev_sample<0 and next_sample>0): #it is an upcrossing
            up_crossings.append(i)
    return up_crossings

def height_from_up_crossings(measured_data,up_crossings):
    #this using up crossings to extract periods and heights from pressure data
    data_mean = np.mean(measured_data)
    print("mean depth: ",data_mean) #should match sensor depth
    periods = []
    heights = []
    for i in range(len(up_crossings)-1): # check the time and heights between each up crossing to get hightest that is wave crest
        periods.append((up_crossings[i+1]-up_crossings[i])* 0.03125)#gap between up-crossings converted to seconds
        period_pressures= measured_data[up_crossings[i]:up_crossings[i+1]]
        heights.append((sorted(period_pressures, reverse=True)[0] - data_mean)*2) # get the highest point and subtract the mean
    return periods, heights

def return_significant(wave_data):
    top_third = sorted(wave_data, reverse=True)[0:int((np.round((len(wave_data)/3))))]
    return np.mean(top_third)