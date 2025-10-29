**Performance analysis of 16QAM modulation over Rayleigh channel (Non Line-of-Sight) using Minimum Mean Square Error Estimator to estimate the channel.**


The system model considered for this project is shown in Flowchart.png 
It consists of 1 transmit antenna and 1 receive antenna. 

We send random data that has equal probability of 0’s and 1’s generated through Linear Congruential Generator. 
Data is modulated with 16-Quadrature Amplitude Modulation(QAM) with each bit having unit energy. 
These modulated symbols are arranged into packets along with pilots and are transmitted over a Rayleigh Channel (By taking 2 normally distributed samples on complex plane). 
MMSE estimator is used to estimate the channel “h” with the help of IEEE Standard 802.11n pilots. 
Then received symbols are compensated with "ℎ" using Zero Forcing equalization technique. 

**QAM demodulation is performed and then BER vs SNR, SER vs SNR and MSE vs SNR is plotted.**
