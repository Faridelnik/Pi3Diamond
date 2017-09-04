import visa
import numpy as np
import time

instr = visa.instrument("GPIB0::30")

print instr.ask("*IDN?")

instr.write("POW -10")
#instr.write("OUTP ON")

instr.write("FREQ:STAR 1e9")
instr.write("FREQ:STOP 2e9")
instr.write("FREQ:STEP 100e6")
print instr.ask("FREQ:STAR?;STOP?;STEP?")

instr.write("FREQ:MODE SWE")
instr.write("SWE:DWEL 0.001")
instr.write("SWE:DIR UP")
instr.write("SWE:COUNT 2")

#instr.write("TRIG:SOUR EXT; SLOP POS")
#instr.write("TRIG:SOUR IMM")


#instr.write("INIT:CONT ON")
#instr.write("INIT")
time.sleep(1)
#instr.write("TRIG")

