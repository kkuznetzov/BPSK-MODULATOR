#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 20 18:36:53 2023

@author: kkuznetzov
"""

from os.path import dirname, join as pjoin
from scipy.io import wavfile
import pdb
import scipy.io
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os

# Input wav file
# Имя входного wav файла
wav_file_name = 'wav\\bpsk_out.wav'
wav_file_name = os.path.join(os.path.dirname(__file__), wav_file_name)

# Name of the output data file
# Имя выходного файла с данными
data_file_out_name = 'received_data.txt'
data_file_out_name = os.path.join(os.path.dirname(__file__), data_file_out_name)

# Sampling rate of wav file, other values don't work
# Частота дискретизации файла wav
wav_samplerate = 44100

# Carrier frequency value for transmission
# Значение частоты несущей частоты для передачи
carrier_frequency = 2940

# Data transfer rate, bits per second
# Скорость передачи данных
transmit_bitrate = 588

# Carrier frequency duration in bit durations, transmitted before the preamble
# Used for PLL operation and signal detection
# Размер несущей в битах, до передачи преамбулы
carrier_bit_size = 24

# Carrier phase value (0 = -1, 1 = +1)
# Значение частотаы преамбула, задано значением бита
carrier_bit_value = 0

# Preamble (alternating +1 and -1) duration, in bit durations
# Used for bit synchronization
# Размер преамбулы, бит
preamble_bit_size = 32

# Postamble duration, only needed for Windows player, loses end of wav file
# Размер постбулы, бит
postamble_bit_size = 24

# Synchronization word size, bits. Used for byte synchronization
# Размер слова синхронизации, бит
synchronization_word_bit_size = 8

# Sync word bits value
# Значение бит слова синхронизации
synchronization_word_bit_value = 0

# Number of wav sampling samples per data bit
# Число отсчётов частоты дискретщизации на каждый бит
bit_samplerate_period_per_bit = wav_samplerate / transmit_bitrate

# Open wav file for reading
# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_name)
input_signal_length = input_signal_data.shape[0]

# Signal duration, bits and bytes
# Длительность посылки бит и байт
input_signal_bit_length = input_signal_length / bit_samplerate_period_per_bit
input_signal_byte_length = input_signal_bit_length / 8

# Output parameters to the console
# Вывод параметров
print("Частота дискретизации wav =", wav_samplerate)
print("Значение частоты несущей частоты =", carrier_frequency)
print("Число периодов дискрертизации на бит =", bit_samplerate_period_per_bit)
print("Число отсчётов входного файла =", input_signal_length)
print("Длительность входных данных секунд =", input_signal_length/input_signal_samplerate)
print("Размер несущей, бит =", carrier_bit_size)
print("Значение несущей несущей, бит =", carrier_bit_value)
print("Размер преамбулы, бит =", preamble_bit_size)
print("Размер слова синхронизации, бит =", synchronization_word_bit_size)
print("Значение бит слова синхронизации =", synchronization_word_bit_value)
print("Длина посылки в битах =", input_signal_bit_length)
print("Длина посылки в байтах =", input_signal_byte_length)

# Calculate an array with sin (Q) and cos samples (I)
# Формируем отсчёты синусоиды и косинусоиды несущей частоты
carrier_sin_samples_index = np.arange(bit_samplerate_period_per_bit)
carrier_sin_samples = np.sin(2 * np.pi * (carrier_frequency / transmit_bitrate) * carrier_sin_samples_index / bit_samplerate_period_per_bit)
carrier_cos_samples = np.cos(2 * np.pi * (carrier_frequency / transmit_bitrate) * carrier_sin_samples_index / bit_samplerate_period_per_bit)

# plt.plot(carrier_sin_samples_index, carrier_sin_samples, "-g", carrier_sin_samples_index, carrier_cos_samples, "-b")
# plt.show()

# Scale the samples of the input signal so that they are in the range from -1 to 1
# Масштабируем входной сигнал, что бы максимум был 1 или -1
input_signal_maximum_amplitude = max(abs(input_signal_data))
input_signal_data = input_signal_data / input_signal_maximum_amplitude

# The result of multiplying the input signal by the reference signals
# In-phase (i) and quadrature (q)
# Результат перемножения сигнала на опорные сигналы
costas_loop_i_signal_reference_multiplication_value = 0
costas_loop_q_signal_reference_multiplication_value = 0

# The result of multiplying the in-phase and quadrature error values
# For PI filter
# Результат перемножения синфазной и квадратурной ошибок
costas_loop_iq_error_multiplication_value = 0

# Current value of reference signals
# Текущее значение опорного сигнала для ФАПЧ типа 2
costas_loop_reference_cos_sample_value = 0
costas_loop_reference_sin_sample_value = 0

# Phase counter Costas Loop
# Счётчик фазы для ФАПЧ типа 2
costas_loop_phase_cnt = 0

# Calculation of PI filter coefficients
# Вычисление коэффициентов ПИ регулятора
costas_loop_bandwidth = transmit_bitrate
costas_loop_samplerate = wav_samplerate
costas_loop_damping_factor = math.sqrt(2) / 2
costas_loop_phase_error_detector_gain = 0.5 # Kp
costas_loop_oscillator_gain = 1 # Ko

# Filter coefficients
# Коэффициенты фильтра типа 2
costas_loop_coefficient_h = costas_loop_bandwidth / costas_loop_samplerate
costas_loop_coefficient_h = costas_loop_coefficient_h / (costas_loop_damping_factor + 1 / (4 * costas_loop_damping_factor))
costas_loop_coefficient_k1 = 4 * costas_loop_damping_factor / (costas_loop_phase_error_detector_gain * costas_loop_oscillator_gain)
costas_loop_coefficient_k1 = costas_loop_coefficient_k1 * costas_loop_coefficient_h
costas_loop_coefficient_k2 = 4 * (costas_loop_coefficient_h ** 2) / (costas_loop_phase_error_detector_gain * costas_loop_oscillator_gain)

# Current PI filter values
# Значение выхода фильтра ФАПЧ типа 2
'''costas_loop_i_loop_value_k1 = 0
costas_loop_q_loop_value_k1 = 0
costas_loop_i_loop_value_k2 = 0
costas_loop_q_loop_value_k2 = 0
costas_loop_i_filter_output = 0
costas_loop_q_filter_output = 0'''
costas_loop_iq_loop_value_k1 = 0
costas_loop_iq_loop_value_k2 = 0
costas_loop_iq_filter_output = 0

# To implement average filters
# Буферы плавающего среднего, счётчик буфера и размер буфера
costas_loop_average_buffer_size = bit_samplerate_period_per_bit
costas_loop_average_buffer_counter = 0
costas_loop_i_average_buffer = np.linspace(0, 0, int(costas_loop_average_buffer_size))
costas_loop_q_average_buffer = np.linspace(0, 0, int(costas_loop_average_buffer_size))
costas_loop_bit_value_average_buffer = np.linspace(0, 0, int(costas_loop_average_buffer_size))
costas_loop_i_average_value = 0
costas_loop_q_average_value = 0

# Carrier capture flag
# Флаг захвата несущей
costas_loop_carrier_lock_flag = 0

# Carrier sign to further define bit values
# Знак несущей для дальнейшего определения значений битов
costas_loop_carrier_lock_sing = 0

# Carrier capture counter and counter threshold
# Счётчик для определения захвата несущей и порог счётчика
costas_loop_carrier_lock_counter = 0
costas_loop_carrier_lock_counter_threshold = carrier_bit_size // 2

# Carrier lock amplitude threshold
# Порог определения захвата несущей
costas_loop_carrier_lock_threshold = 0.1

# Signal threshold for bit selection/detection
# Порог для выделения бит
bit_signal_level_threshold = 0.05

# The average value of the signal and the previous value. To search for a preamble
# Значение сигнала и предыдущее значение
bit_signal_average_value = 0
bit_signal_average_previous = 0

# Signs of received bits
# Знаки принимаемых бит
bit_0_signal_sign = 0
bit_1_signal_sign = 0

# To implement the Early-Late algorithm
# Для алгоритма Early-Late
bit_signal_el_value_hold = 0  # t
bit_signal_el_value_early = 0 # t + delta t
bit_signal_el_value_late = 0  # t - delta t
bit_signal_el_value_error = 0
bit_signal_el_time_hold = round(bit_samplerate_period_per_bit / 2)
bit_signal_el_time_early = round(bit_signal_el_time_hold - bit_samplerate_period_per_bit / 4)
bit_signal_el_time_late = round(bit_signal_el_time_hold + bit_samplerate_period_per_bit / 4)
bit_signal_el_error_threshold = 0.05

# To search for a preamble
# Preamble capture flag, amplitude rise (maximum) flag, phase (bit sign) change counter
# Для поиска преамбулы
preamble_lock_flag = 0
bit_signal_average_value_rising_flag = 0
bit_value_change_counter = 0
bit_value_change_lock_threshold = preamble_bit_size // 2

# The value of the received bit and the previous bit value
# Значение принятого бита и предыдущего значения бита
bit_digital_value = 0
bit_digital_previous_value = 0

# The value of the received bit and the previous bit value
# Значение счётчика фильтра в которой захвачена преамбула
bit_digital_average_buffer_counter_value = 0

# To search for the word sync
# Для поиска слова синхронизации
synchronization_word_bit_counter = 0
synchronization_word_lock_flag = 0

# Byte counter and bit counter
# Счётчик байт и счётчик бит
output_byte_count = 0
output_bit_count = 0

# Output data, bytes
# Выходные данные, байты
byte_value = 0
output_stream_bytes = []

# Debug
# Для отладки
costas_loop_i_filter_output_debug = []
costas_loop_q_filter_output_debug = []
costas_loop_iq_filter_output_debug = []
costas_loop_carrier_lock_sample_debug = []
costas_loop_carrier_lock_value_debug = []
bit_signal_average_value_debug = []
bit_signal_el_error_value_debug = []
bit_digital_value_debug_x = []
bit_digital_value_debug_y = []
bit_digital_value_debug = []
preamble_lock_sample_debug = []
preamble_lock_value_debug = []
synchro_lock_sample_debug = []
synchro_lock_value_debug = []
byte_digital_sample_x_debug = []
byte_digital_sample_y_debug = []
byte_digital_value_debug = []

# Loop through input samples
# Проход по входным отсчётам
for i in range(int(input_signal_length)):
    # PLL implementation for carrier lock
    # Реализация ФАПЧ для захвата несущей

    # Current values of reference signals
    # Текущее значение опорного сигнала
    costas_loop_reference_sin_sample_value = carrier_sin_samples[int(costas_loop_phase_cnt)]
    costas_loop_reference_cos_sample_value = carrier_cos_samples[int(costas_loop_phase_cnt)]

    # Multiply the input signal by the reference signals
    # Перемножаем входной сигнал на опорные сигналы
    costas_loop_i_signal_reference_multiplication_value = costas_loop_reference_sin_sample_value * input_signal_data[i]
    costas_loop_q_signal_reference_multiplication_value = costas_loop_reference_cos_sample_value * input_signal_data[i]

    # Not used
    # Считаем значения фильтра ФАПЧ, синфазный
    '''costas_loop_i_loop_value_k1 = costas_loop_i_signal_reference_multiplication_value * costas_loop_coefficient_k1
    costas_loop_i_filter_output = costas_loop_i_loop_value_k1 + costas_loop_i_loop_value_k2
    costas_loop_i_loop_value_k2 = costas_loop_i_signal_reference_multiplication_value * costas_loop_coefficient_k2

    # Считаем значения фильтра ФАПЧ, квадратурный
    costas_loop_q_loop_value_k1 = costas_loop_q_signal_reference_multiplication_value * costas_loop_coefficient_k1
    costas_loop_q_filter_output = costas_loop_q_loop_value_k1 + costas_loop_q_loop_value_k2
    costas_loop_q_loop_value_k2 = costas_loop_q_signal_reference_multiplication_value * costas_loop_coefficient_k2'''

    # Floating average filter for I and Q components
    # Фильтр плавающего среднего для компонентов I и Q
    costas_loop_i_average_buffer[costas_loop_average_buffer_counter] = costas_loop_i_signal_reference_multiplication_value
    costas_loop_q_average_buffer[costas_loop_average_buffer_counter] = costas_loop_q_signal_reference_multiplication_value
    costas_loop_i_average_value = np.mean(costas_loop_i_average_buffer)
    costas_loop_q_average_value = np.mean(costas_loop_q_average_buffer)

    # Multiply I and Q components
    # Перемножаем I и Q части
    costas_loop_iq_error_multiplication_value = costas_loop_i_average_value * costas_loop_q_average_value
    # costas_loop_iq_error_multiplication_value = costas_loop_i_filter_output * costas_loop_q_filter_output

    # Computing PI filter values
    # Значение фильтра
    costas_loop_iq_loop_value_k1 = costas_loop_iq_error_multiplication_value * costas_loop_coefficient_k1
    costas_loop_iq_filter_output = costas_loop_iq_loop_value_k1 + costas_loop_iq_loop_value_k2
    costas_loop_iq_loop_value_k2 = costas_loop_iq_error_multiplication_value * costas_loop_coefficient_k2

    # Debug
    costas_loop_i_filter_output_debug.append(costas_loop_i_average_value)
    costas_loop_q_filter_output_debug.append(costas_loop_q_average_value)
    costas_loop_iq_filter_output_debug.append(costas_loop_iq_filter_output)
    if costas_loop_carrier_lock_flag == 0:
        bit_signal_average_value_debug.append(0)
        bit_signal_el_error_value_debug.append(0)

    # Calculation of the phase counter value for reference signals
    # costas_loop_iq_filter_output - phase error
    # Инкремент счётчика фазы, частоты бита 0
    costas_loop_phase_cnt = costas_loop_phase_cnt + 1 + costas_loop_iq_filter_output
    if costas_loop_phase_cnt >= bit_samplerate_period_per_bit:
        costas_loop_phase_cnt = costas_loop_phase_cnt - bit_samplerate_period_per_bit
    if costas_loop_phase_cnt < 0:
        costas_loop_phase_cnt = costas_loop_phase_cnt + bit_samplerate_period_per_bit

    # Floating average buffer counter increment
    # Инкремент счётчика буфера плавающего среднего
    costas_loop_average_buffer_counter += 1
    if costas_loop_average_buffer_counter >= costas_loop_average_buffer_size:
        costas_loop_average_buffer_counter = 0

    # Carrier frequency search
    # Контроль захвата несущей
    if (costas_loop_carrier_lock_flag == 0) and (costas_loop_average_buffer_counter == 0):
        # Counter increment when value is greater than threshold, otherwise reset counter
        # Инкремент счётчика когда значение больше порога, иначе сброс счётчика
        if abs(costas_loop_i_average_value) > costas_loop_carrier_lock_threshold:
            costas_loop_carrier_lock_counter += 1
        else:
            costas_loop_carrier_lock_counter = 0

        # Checking the counter value
        # Проверка значения счётчика
        if costas_loop_carrier_lock_counter > costas_loop_carrier_lock_counter_threshold:
            # Has a carrier lock
            # Есть захват несущей
            costas_loop_carrier_lock_flag = 1
            costas_loop_carrier_lock_sample_debug.append(i)
            costas_loop_carrier_lock_value_debug.append(costas_loop_i_average_value)

            # Remember the sign (phase) of the carrier
            # Запомним знак несущей
            if costas_loop_i_average_value > 0:
                costas_loop_carrier_lock_sing = +1
            if costas_loop_i_average_value < 0:
                costas_loop_carrier_lock_sing = -1

            # Store values for each of the bits depending on the value of the carrier bit
            # Запомним значения для каждого из бит в зависимоти от значения бита несущей
            if carrier_bit_value == 0:
                bit_0_signal_sign = costas_loop_carrier_lock_sing
                bit_1_signal_sign = -costas_loop_carrier_lock_sing
            else:
                bit_0_signal_sign = -costas_loop_carrier_lock_sing
                bit_1_signal_sign = costas_loop_carrier_lock_sing

    # If the carrier is locked
    # Если захвачена несущая
    if costas_loop_carrier_lock_flag == 1:
        # Second average buffer for in-phase component
        # Второй буфер для синфазной составляющей
        costas_loop_bit_value_average_buffer[costas_loop_average_buffer_counter] = costas_loop_i_average_value

        # Average signal level after filter
        # Уровень сигнала равен синфазной составляющей
        bit_signal_average_previous = bit_signal_average_value
        bit_signal_average_value = np.mean(costas_loop_bit_value_average_buffer)

        # Debug
        bit_signal_average_value_debug.append(bit_signal_average_value)
        # costas_loop_i_filter_output_debug[i] = bit_signal_average_value
        # costas_loop_q_filter_output_debug[i] = costas_loop_i_average_value

    # Implementation of the Early-Late algorithm if the carrier is captured
    # Алгоритм Early-Late, после того как захвачена несущая
    if costas_loop_carrier_lock_flag == 1:
        # Shift values
        # early - earlier envelope value
        # hold - the middle of the envelope
        # late - late envelope value
        # Продвигаем значения
        # early - ранее значение огибающей
        # hold - середина огибающей
        # late - позднее значение огибающей
        if round(bit_signal_el_time_late) == costas_loop_average_buffer_counter:
            bit_signal_el_value_late = abs(bit_signal_average_value)
        if round(bit_signal_el_time_early) == costas_loop_average_buffer_counter:
            bit_signal_el_value_early = abs(bit_signal_average_value)
        if round(bit_signal_el_time_hold) == costas_loop_average_buffer_counter:
            bit_signal_el_value_hold = abs(bit_signal_average_value)

            # Error signal
            # Сигнал ошибки
            '''if bit_signal_mm_value_hold != 0:
                bit_signal_mm_value_error = (bit_signal_mm_value_early - bit_signal_mm_value_late) / bit_signal_mm_value_hold
            else:
                bit_signal_mm_value_error = bit_signal_mm_value_early - bit_signal_mm_value_late'''
            bit_signal_el_value_error = (bit_signal_el_value_early - bit_signal_el_value_late) / 2
            # bit_signal_average_value_debug.append(bit_signal_mm_value_error)

            # Update the time value of the middle bit
            # Обновляем время середины бита
            bit_signal_el_time_hold -= bit_signal_el_value_error
            if bit_signal_el_time_hold < 0:
                bit_signal_el_time_hold += bit_samplerate_period_per_bit
            if bit_signal_el_time_hold > bit_samplerate_period_per_bit:
                bit_signal_el_time_hold -= bit_samplerate_period_per_bit

            # Update the time value of the late
            # Обновляем значение времени late
            bit_signal_el_time_late = bit_signal_el_time_hold + (bit_samplerate_period_per_bit / 4)
            if bit_signal_el_time_late < 0:
                bit_signal_el_time_late += bit_samplerate_period_per_bit
            if bit_signal_el_time_late > bit_samplerate_period_per_bit:
                bit_signal_el_time_late -= bit_samplerate_period_per_bit

            # Update the time value of the early
            # Обновляем значение времени early
            bit_signal_el_time_early = bit_signal_el_time_hold - (bit_samplerate_period_per_bit / 4)
            if bit_signal_el_time_early < 0:
                bit_signal_el_time_early += bit_samplerate_period_per_bit
            if bit_signal_el_time_early > bit_samplerate_period_per_bit:
                bit_signal_el_time_early -= bit_samplerate_period_per_bit

            # Continuously update the mid-bit time value
            # Постоянно обновляем значение середины бита
            if abs(bit_signal_el_value_error) <= bit_signal_el_error_threshold:
                bit_digital_average_buffer_counter_value = round(bit_signal_el_time_hold)

        # Debug
        if preamble_lock_flag == 0:
            bit_signal_el_error_value_debug.append(bit_signal_el_value_error)
        else:
            # bit_signal_mm_error_value_debug.append(0)
            bit_signal_el_error_value_debug.append(bit_signal_el_value_error)

    # If the carrier is captured, then we are waiting for the preamble
    # Если несущая захвачена, то ждём преамбулу
    if (costas_loop_carrier_lock_flag == 1) and (preamble_lock_flag == 0):
        # We look at the value of the bit at the time of hold, with a threshold check
        # Смотрим значение бита в момент hold, с учётом порога
        if (abs(bit_signal_average_value) > bit_signal_level_threshold) and (round(bit_signal_el_time_hold) == costas_loop_average_buffer_counter):
            # Debug
            bit_digital_value_debug_x.append(i)
            bit_digital_value_debug_y.append(bit_signal_average_value)

            # The value of a bit depends on the sign of bit 0 and bit 1
            # Значение бита зависит от знака бита 0 и бита 1
            if bit_signal_average_value < 0:
                if bit_0_signal_sign == -1:
                    bit_digital_value = 0
                    bit_digital_value_debug.append('0')
                else:
                    bit_digital_value = 1
                    bit_digital_value_debug.append('1')

            if bit_signal_average_value > 0:
                if bit_0_signal_sign == 1:
                    bit_digital_value = 0
                    bit_digital_value_debug.append('0')
                else:
                    bit_digital_value = 1
                    bit_digital_value_debug.append('1')

            # Checking for bit alternation, this is the preamble
            # Проверка на чередование бит
            if bit_digital_previous_value != bit_digital_value:
                bit_value_change_counter += 1
                bit_digital_previous_value = bit_digital_value
                if bit_value_change_counter > bit_value_change_lock_threshold:
                    preamble_lock_flag = 1

                    # Debug
                    preamble_lock_sample_debug.append(i)
                    preamble_lock_value_debug.append(bit_signal_average_value)

                    # Save the value of the filter counter for detecting bits
                    # Запомним значение счётчика для середины бита
                    bit_digital_average_buffer_counter_value = round(bit_signal_el_time_hold)

    # If the carrier is captured and the preamble, then we wait for the sync word, and then the data
    # Если несущая захвачена и преабула, то ждём синхрослово, а потом данные
    if (costas_loop_carrier_lock_flag == 1) and (preamble_lock_flag == 1):
        if synchronization_word_lock_flag == 0:
            # Determine the value of the received bit
            # Определяем значение принятого бита в середине бита
            if costas_loop_average_buffer_counter == bit_digital_average_buffer_counter_value:

                # Debug
                bit_digital_value_debug_x.append(i)
                bit_digital_value_debug_y.append(bit_signal_average_value)

                # The value of a bit depends on the sign of bit 0 and bit 1
                # Значение бита зависит от знака бита 0 и бита 1
                if bit_signal_average_value < 0:
                    if bit_0_signal_sign == -1:
                        bit_digital_value = 0
                        bit_digital_value_debug.append('0')
                    else:
                        bit_digital_value = 1
                        bit_digital_value_debug.append('1')

                if bit_signal_average_value > 0:
                    if bit_0_signal_sign == 1:
                        bit_digital_value = 0
                        bit_digital_value_debug.append('0')
                    else:
                        bit_digital_value = 1
                        bit_digital_value_debug.append('1')

                # Synchronization word bit counter
                # Счётчик бит слова синхронизации
                if bit_digital_value == synchronization_word_bit_value:
                    synchronization_word_bit_counter += 1
                else:
                    synchronization_word_bit_counter = 0

                # Comparing counter value with threshold
                # Here we fix the capture of the synchronization word
                # Сравнение счётчика с порогом
                if synchronization_word_bit_counter == synchronization_word_bit_size:
                    synchronization_word_lock_flag = 1
                    synchro_lock_sample_debug.append(i)
                    synchro_lock_value_debug.append(bit_signal_average_value)

        # If carrier, preamble and sync word are captured
        # Если захвачены несущаяя, преамбула и синхрослово
        else:
            # Determine the value of the received bit
            # Определяем значение принятого бита в середине бита
            if costas_loop_average_buffer_counter == bit_digital_average_buffer_counter_value:

                # Debug
                bit_digital_value_debug_x.append(i)
                bit_digital_value_debug_y.append(bit_signal_average_value)

                # The value of a bit depends on the sign of bit 0 and bit 1
                # Значение бита зависит от знака бита 0 и бита 1
                if bit_signal_average_value < 0:
                    if bit_0_signal_sign == -1:
                        bit_digital_value = 0
                        bit_digital_value_debug.append('0')
                    else:
                        bit_digital_value = 1
                        bit_digital_value_debug.append('1')

                if bit_signal_average_value > 0:
                    if bit_0_signal_sign == 1:
                        bit_digital_value = 0
                        bit_digital_value_debug.append('0')
                    else:
                        bit_digital_value = 1
                        bit_digital_value_debug.append('1')

                # Putting a bit into a byte
                # Помещаем бит в байт
                byte_value = byte_value | (bit_digital_value << output_bit_count)

                # Bit and byte counters, store byte, reset byte value
                # Счётчики бит и байт, сохраняем байт, сброс значения байта
                output_bit_count += 1
                if output_bit_count == 8:
                    output_stream_bytes.append(byte_value)
                    output_bit_count = 0
                    output_byte_count += 1
                    byte_digital_sample_x_debug.append(i)
                    byte_digital_sample_y_debug.append(0.55 + (output_byte_count % 2) / 30)
                    byte_digital_value_debug.append(byte_value)
                    byte_value = 0

# For debug
# Для отладки
plt.figure("Time Во времени")
# plt.plot(costas_loop_i_filter_output_debug, "-b", costas_loop_q_filter_output_debug, "-g", costas_loop_iq_filter_output_debug, "-r", bit_signal_average_value_debug, "-y", bit_signal_mm_error_value_debug, "-m")
plt.plot(costas_loop_i_filter_output_debug, "-b", costas_loop_q_filter_output_debug, "-g", bit_signal_average_value_debug, "-y", bit_signal_el_error_value_debug, "-m")
plt.title('Receive, rate (Приём BPSK сигнала со скоростью) {0} bit/sec (бит/сек), carrier (частота несущей) {1} Hz (Гц)'.format(transmit_bitrate, carrier_frequency))
plt.xlabel('Sample Номер отсчёта', color='gray')
plt.ylabel('Filter output Выход фильтра для значения захвата сигнала', color='gray')

plt.plot(costas_loop_carrier_lock_sample_debug, costas_loop_carrier_lock_value_debug, 'rs')
for i in range(len(costas_loop_carrier_lock_sample_debug)):
    x = costas_loop_carrier_lock_sample_debug[i]
    y = costas_loop_carrier_lock_value_debug[i]
    if y > 0:
        yt = y + 0.1
    else:
        yt = y - 0.1
    plt.annotate('pll lock', xy = (x, y), xytext = (x, yt), arrowprops = dict(facecolor ='green', shrink = 0.05))

plt.plot(bit_digital_value_debug_x, bit_digital_value_debug_y, 'ro')
for i in range(len(bit_digital_value_debug_x)):
    plt.annotate(bit_digital_value_debug[i], (bit_digital_value_debug_x[i], bit_digital_value_debug_y[i]), ha='center')
    # plt.axvline(bit_digital_value_debug_x[i], color='y', linestyle='dotted', label='axvline - full height')

plt.plot(preamble_lock_sample_debug, preamble_lock_value_debug, 'rs')
for i in range(len(preamble_lock_sample_debug)):
    x = preamble_lock_sample_debug[i]
    y = preamble_lock_value_debug[i]
    if y > 0:
        yt = y + 0.1
    else:
        yt = y - 0.1
    plt.annotate('preamble lock', xy = (x, y), xytext = (x, yt), arrowprops = dict(facecolor ='green', shrink = 0.05))

plt.plot(synchro_lock_sample_debug, synchro_lock_value_debug, 'rs')
for i in range(len(synchro_lock_sample_debug)):
    x = synchro_lock_sample_debug[i]
    y = synchro_lock_value_debug[i]
    if y > 0:
        yt = y + 0.1
    else:
        yt = y - 0.1
    plt.annotate('sync lock', xy = (x, y), xytext = (x, yt), arrowprops = dict(facecolor ='green', shrink = 0.05))

plt.plot(byte_digital_sample_x_debug, byte_digital_sample_y_debug, 'rs')
for i in range(len(byte_digital_sample_x_debug)):
    plt.annotate("0x{0:x}".format(byte_digital_value_debug[i]), (byte_digital_sample_x_debug[i], byte_digital_sample_y_debug[i]), ha='center')
    # plt.axvline(byte_digital_sample_x_debug[i], color='y', linestyle='dotted', label='axvline - full height')

plt.figure("I/Q plot")
plt.plot(bit_digital_value_debug_y, np.zeros(len(bit_digital_value_debug_y)), '.')

plt.grid()
plt.show()

# Convert to uint8
# Преобразуем в uint8
output_stream_bytes_uint8 = np.uint8(output_stream_bytes)

# Write the data file
# Записываем файл с данными
file = open(data_file_out_name, "wb")
file.write(output_stream_bytes_uint8)
file.close()