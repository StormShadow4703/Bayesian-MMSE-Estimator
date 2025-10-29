import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc

# --------------------------
# Parameters
# --------------------------
num_bits = 4000
data_per_frame = 1000
num_trials = 10
a, c, m = 1664525, 1013904223, 2**32
seed = 42
snr_db_range = np.arange(0, 22, 1)
np.random.seed(0)

# Pilot sequence
pilot_symbols = np.array([
    1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,
    1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,
    1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,
    1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1
], dtype=np.complex128)
Lp = len(pilot_symbols)

# --------------------------
# LCG Bitstream
# --------------------------
lcg = [seed]
for _ in range(num_bits - 1):
    lcg.append((a * lcg[-1] + c) % m)
lcg_normalized = np.array(lcg, dtype=np.float64) / m
bitstream = (lcg_normalized > 0.5).astype(int)

# --------------------------
# 16-QAM Mapping
# --------------------------
symbols_bits = [bitstream[i:i+4] for i in range(0, len(bitstream), 4)]
qam_symbols = []
valid_symbol_bits = []
for b in symbols_bits:
    if len(b) < 4:
        continue
    b0, b1, b2, b3 = b
    I = (1 - 2*b0) * (2 - (1 - 2*b2))
    Q = (1 - 2*b1) * (2 - (1 - 2*b3))
    q = (1 / np.sqrt(10)) * (I + 1j * Q)
    qam_symbols.append(q)
    valid_symbol_bits.append([int(b0), int(b1), int(b2), int(b3)])
qam_symbols = np.array(qam_symbols, dtype=np.complex128)
valid_symbol_bits = np.array(valid_symbol_bits, dtype=int)
num_symbols = len(qam_symbols)
signal_power = np.mean(np.abs(qam_symbols)**2)

# --------------------------
# Demodulation
# --------------------------
def demodulate(symbol):
    I = np.real(symbol) * np.sqrt(10)
    Q = np.imag(symbol) * np.sqrt(10)
    b0 = 0 if I > 0 else 1
    b2 = 0 if abs(I) < 2 else 1
    b1 = 0 if Q > 0 else 1
    b3 = 0 if abs(Q) < 2 else 1
    return [b0, b1, b2, b3]

# --------------------------
# Frame Setup
# --------------------------
num_frames = int(np.ceil(num_symbols / data_per_frame))
sigma_h2 = 1.0

# --------------------------
# Storage
# --------------------------
ber_uneq, ber_eq = [], []
ser_uneq, ser_eq = [], []
norm_mmse_list = []

# --------------------------
# SNR Loop
# --------------------------
for snr_db in snr_db_range:
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    sigma_n2 = noise_power
    noise_std = np.sqrt(noise_power / 2)

    ber_uneq_trials, ber_eq_trials = [], []
    ser_uneq_trials, ser_eq_trials = [], []
    mmse_trials = []

    for _ in range(num_trials):
        bit_errors_uneq = 0
        bit_errors_eq = 0
        sym_errors_uneq = 0
        sym_errors_eq = 0
        total_data_symbols = 0
        total_mse = 0.0
        symbol_idx = 0

        for frame in range(num_frames):
            Nd = min(data_per_frame, num_symbols - symbol_idx)
            if Nd <= 0:
                break
            s_frame = qam_symbols[symbol_idx : symbol_idx + Nd]
            bits_frame = valid_symbol_bits[symbol_idx : symbol_idx + Nd]

            h = (np.random.randn() + 1j*np.random.randn()) * np.sqrt(sigma_h2/2)

            n_p = noise_std * (np.random.randn(Lp) + 1j*np.random.randn(Lp))
            y_p = h * pilot_symbols + n_p

            p = pilot_symbols
            denom = sigma_h2 * np.vdot(p, p) + sigma_n2
            numer = sigma_h2 * np.vdot(p, y_p)
            h_hat = numer / denom

            mse_frame = np.abs(h - h_hat)**2
            total_mse += mse_frame
            total_data_symbols += Nd

            n_data = noise_std * (np.random.randn(Nd) + 1j*np.random.randn(Nd))
            y_data = h * s_frame + n_data

            est_bits_uneq = []
            sym_err_uneq = 0
            for i in range(Nd):
                est_b = demodulate(y_data[i])
                est_bits_uneq.extend(est_b)
                if est_b != list(bits_frame[i]):
                    sym_err_uneq += 1

            est_bits_eq = []
            sym_err_eq = 0
            s_hat_eq = y_data / h_hat
            for i in range(Nd):
                est_b = demodulate(s_hat_eq[i])
                est_bits_eq.extend(est_b)
                if est_b != list(bits_frame[i]):
                    sym_err_eq += 1

            bits_frame_flat = bits_frame.flatten().tolist()
            est_bits_uneq = np.array(est_bits_uneq, dtype=int)
            est_bits_eq = np.array(est_bits_eq, dtype=int)
            bits_frame_arr = np.array(bits_frame_flat, dtype=int)

            bit_errors_uneq += np.sum(bits_frame_arr != est_bits_uneq)
            bit_errors_eq += np.sum(bits_frame_arr != est_bits_eq)
            sym_errors_uneq += sym_err_uneq
            sym_errors_eq += sym_err_eq
            symbol_idx += Nd

        ber_uneq_trials.append(bit_errors_uneq / (4 * num_symbols))
        ber_eq_trials.append(bit_errors_eq / (4 * num_symbols))
        ser_uneq_trials.append(sym_errors_uneq / num_symbols)
        ser_eq_trials.append(sym_errors_eq / num_symbols)
        mmse_trials.append(total_mse / (sigma_h2 * num_frames))

    ber_uneq.append(np.mean(ber_uneq_trials))
    ber_eq.append(np.mean(ber_eq_trials))
    ser_uneq.append(np.mean(ser_uneq_trials))
    ser_eq.append(np.mean(ser_eq_trials))
    norm_mmse_list.append(np.mean(mmse_trials))

# --------------------------
# Theoretical AWGN Curves (16-QAM)
# --------------------------
snr_linear = 10**(snr_db_range / 10)
M = 16
k = np.log2(M)
EsN0 = snr_linear
EbN0 = EsN0 / k

# Theoretical Symbol Error Rate for square 16-QAM in AWGN
ser_theoretical = 3/2 * erfc(np.sqrt(0.1 * snr_linear))
ser_theoretical = np.minimum(ser_theoretical, 1)

# Theoretical BER (approximate)
ber_theoretical = ser_theoretical / k

# --------------------------
# Plot Results
# --------------------------
plt.style.use('ggplot')
fig = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# BER Plot
ax1.semilogy(snr_db_range, ber_uneq, 'r-o', label='Unequalized BER')
ax1.semilogy(snr_db_range, ber_eq, 'b-s', label='Equalized BER')
ax1.semilogy(snr_db_range, ber_theoretical, 'k--', label='AWGN Theoretical BER')
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('Bit Error Rate')
ax1.set_title('BER vs SNR (16-QAM)')
ax1.grid(True, which='both', linestyle=':')
ax1.legend()

# SER Plot
ax2.semilogy(snr_db_range, ser_uneq, 'r-o', label='Unequalized SER')
ax2.semilogy(snr_db_range, ser_eq, 'b-s', label='Equalized SER')
ax2.semilogy(snr_db_range, ser_theoretical, 'k--', label='AWGN Theoretical SER')
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('Symbol Error Rate')
ax2.set_title('SER vs SNR (16-QAM)')
ax2.grid(True, which='both', linestyle=':')
ax2.legend()

# MSE Plot
ax3.semilogy(snr_db_range, norm_mmse_list, 'm-o', label='Normalized MMSE (E[|h-h_hat|^2]/E[|h|^2])')
ax3.set_xlabel('SNR (dB)')
ax3.set_ylabel('Normalized MMSE')
ax3.set_title('MSE vs SNR (MMSE Estimation)')
ax3.grid(True, which='both', linestyle=':')
ax3.legend()

plt.tight_layout()
plt.show()
