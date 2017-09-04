import numpy as np

from traits.api import HasTraits, Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum, Button, on_trait_change, cached_property, Code, List, NO_COMPARE
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group
from enable.api import Component, ComponentEditor

class dip_position(HasTraits):
    """Calculates the Larmor frequencies and the dip position (tau) in spin echo envelope for XY8 decoupling sequence."""
    
    
    nuclei =  Enum('H1', 'H2', 'B10', 'B11', 'C13', 'N15', 'F19', 'P31',
                     label='nuclei',
                     desc='choose the nucleus to calculate larmor frequency',
                     editor=EnumEditor(values={'H1':'1:H1','H2':'2:H2', 'B10':'3:B10', 'B11':'4:B11', 'C13':'5:C13', 'N15':'6:N15','F19':'7:F19', 'P31':'8:P31', 'e':'9:e'},cols=8),)  
    
    calculate_button = Button(label='calculate', desc='calculate')    
    
    magnetic_mode = Bool(True, label='magnetic_mode')
    
    magnetic_field = Float(value = 100., desc='Magnetic field [G]', label='Magnetic field [G]', mode='text', auto_set=False, enter_set=True)
    pi_pulse_duration = Range(low= 0., high=200., value = 30., desc='duration of the pi-pulse [ns]', label='duration of the pi-pulse [ns]', mode='text', auto_set=False, enter_set=True, format_str='%.2f')
   
    MW_frequency = Float(value = 2., desc='MW Frequency [GHz]', label='MW frequency [MHz]', mode='text', auto_set=False, enter_set=True)
    crossing = Bool(False, label='crossing')
    
    larmor_frequency = Float(value=0.0, desc='Larmor frequency [MHz]', label='Larmor frequency [MHz]', format_str='%.2f')
    dip_position = Float(value=0.0, desc='Dip position [ns]', label='Dip position [ns]', format_str='%.2f')
    dip_position_in_XY8 = Float(value=0.0, desc='Dip position in XY8 [ns]', label='Dip position in XY8 [ns]', format_str='%.2f')
    larmor_period=Float(value=0.0, desc='Larmor period [ns]', label='Larmor period [ns]', format_str='%.2f')

    
    
    def _calculate_button_fired(self):
       
       #gyromagnetic ratio of several nuclei, MHz/T 
        g_C13=10.705
        g_H1=42.576
        g_H2=6.536
        g_B10=28.75/(2*np.pi)
        g_B11=85.84/(2*np.pi)
        g_N15=4.316
        g_P31=17.235 
        g_F19=40.052
        g_electron=28024.95164
    

        if self.nuclei == 'H1':
            g=g_H1
            
        elif self.nuclei == 'H2':
            g=g_H2
            
        elif self.nuclei == 'B10':
            g=g_B10
        
        elif self.nuclei == 'B11':
            g=g_B11
            
        elif self.nuclei == 'C13':
            g=g_C13
            
        elif self.nuclei == 'N15':
            g=g_N15
            
        elif self.nuclei == 'P31':
            g=g_P31
            
        elif self.nuclei == 'F19':
            g=g_F19
            
        elif self.nuclei == 'e':
            g=g_electron
            
            
        if self.magnetic_mode:
        
            if self.crossing:
                self.MW_frequency = 2870 + self.magnetic_field * 2.8
            else:
                self.MW_frequency = 2870 - self.magnetic_field * 2.8
                
            self.larmor_frequency=g*self.magnetic_field/10000 #Larmor frequency in MHz
            
            self.larmor_period=1e+3/self.larmor_frequency
            
            self.dip_position=(1/(2*self.larmor_frequency))*1e+3 #in ns
            
            self.dip_position_in_XY8=(self.dip_position-self.pi_pulse_duration)/2
        else:
            if self.crossing:
                self.magnetic_field = (-2870 + self.MW_frequency) / 2.8
            else:
                self.magnetic_field = (2870 - self.MW_frequency) / 2.8
        
            self.larmor_frequency=g*self.magnetic_field/10000 #Larmor frequency in MHz
            self.larmor_period=1e+3/self.larmor_frequency
            
            self.dip_position=(1/(2*self.larmor_frequency))*1e+3 #in ns
            
            self.dip_position_in_XY8=(self.dip_position-self.pi_pulse_duration)/2
        #return self.larmor_frequency, self.dip_position, self.dip_position_in_XY8
        

    traits_view = View(VGroup(HGroup(Item('nuclei', style='custom', show_label=False),
                                    ),
                              HGroup(Item('crossing'),
                                     Item('magnetic_mode'),
                                    ),
                              HGroup(Item('pi_pulse_duration', width= -80),
                                    ),
                             
                              Group( HGroup(Item('magnetic_field', width= -60),
                                            Item('MW_frequency', style='readonly'),
                                            label='Magnetic field',                                               
                                           ),
                                     HGroup(Item('MW_frequency', width= -60),
                                            Item('magnetic_field', style='readonly'),
                                            label='MW frequency',
                                            ),
                                     layout='tabbed',
                                   ),  
                               HGroup(Item('calculate_button', show_label=False),
                                     ),
                               HGroup(Item('larmor_frequency', style='readonly'),
                                      Item('larmor_period', style='readonly'),
                                     ),
                               HGroup(Item('dip_position', style='readonly'),
                                     ),
                               HGroup(Item('dip_position_in_XY8', style='readonly'),  
                                     ),         
                              ),                              
                       title='Calculator', width=500, height=400, buttons=[], resizable=True,
                       )
  