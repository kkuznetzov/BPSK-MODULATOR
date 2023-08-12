# BPSK-MODULATOR
BPSK modulator and demodulator
Code that implements modulation and demodulation using BPSK.

The input of the modulator is a text file, the output is a wav file.
bpsk_modulator.py - simple BPSK modulator.
bpsk_modulator_cosine.py - added filtering of the signal envelope, similar to PSK31.

The demodulator input is a wav file, the output is a text file.  
bpsk_costas_demodulator.py - PLL demodulator based on the Costas loop. Bit synchronization is implemented by finding the maximum output of the average filter.  
bpsk_costas_demodulator_early_late.py - PLL demodulator based on the Costas loop. Bit synchronization is implemented by the Early-Late algorithm.

Additional code:  
add_noise.py - adding noise to the signal.  
add_signal.py - adding a sine wave to the signal.

BPSK signal and BPSK signal with cosine filter.
![BPSK and BPSK with filter](https://github.com/kkuznetzov/BPSK-MODULATOR/blob/master/IMG/BPSK.png)

BPSK signal spectrum.

![BPSK signal spectrum](https://github.com/kkuznetzov/BPSK-MODULATOR/blob/master/IMG/BPSK_SPECTRUM.png)


BPSK signal with cosine filter spectrum

![BPSK signal with cosine filter spectrum](https://github.com/kkuznetzov/BPSK-MODULATOR/blob/master/IMG/BPSK_FILTER_SPECTRUM.png)
