"""
Example custom startup script. To provide custom startup behavior,
rename this file to 'pi3diamond_custom.py' and adapt it to your needs.
"""

if __name__ == '__main__':
    # start confocal including auto_focus tool
    from measurements.confocal import Confocal
    confocal = Confocal()
    confocal.edit_traits()
    from measurements.auto_focus import AutoFocus
    auto_focus = AutoFocus(confocal)
    auto_focus.edit_traits()

    # some imports providing short names for the main modules
    import numpy as np
    import hardware.api as ha
    import measurements as me
    import analysis as an
        
