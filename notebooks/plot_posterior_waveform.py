# figure 1 left panel: comparison NRSUR and XO4a maximum-likelihood waveform (and 90% confidence interval)

import bilby
import matplotlib.pyplot as plt
import glob
import numpy as np
import h5py
import pandas as pd
from gwpy.timeseries import TimeSeries
import lal
import tqdm
#from pesummary.gw.conversions.evolve import evolve_angles_backwards
from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
#from scipy.spatial.distance import jensenshannon
from scipy import stats
from scipy.stats import gaussian_kde
from pesummary.utils.bounded_2d_kde import Bounded_2d_kde
from pesummary.utils.bounded_1d_kde import bounded_1d_kde
import seaborn as sns
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.ticker as ticker
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
matplotlib.rcParams.update({'font.size': 17})
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 20
import matplotlib.collections as mcoll
import matplotlib.lines as mlines
import config
import warnings
warnings.filterwarnings('ignore')

#set up the waveform generator for NRSUR
waveform_generator_NRSUR= bilby.gw.WaveformGenerator(
    duration=8.0, sampling_frequency=1024.0,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    start_time=1384782888.634277 - 6.,
    waveform_arguments={
        "waveform_approximant": "NRSur7dq4",
        "minimum_frequency": 10,
        "maximum_frequency": 448.0,
        "reference_frequency": 10,
    }
)
#set up waveform generator for XO4a
waveform_generator_XO4a= bilby.gw.WaveformGenerator(
    duration=8.0, sampling_frequency=1024.0,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    start_time=1384782888.634277 - 6.,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomXO4a",
        "minimum_frequency": 10,
        "maximum_frequency": 448.0,
        "reference_frequency": 10,
    }
)


#path to the PSD (samne as LVK paper)
psd_files = {
    "H1": "/home/pe.o4/GWTC4/working/S231123cg/generate-psd/trigtime_1384782888.634277105_0.0_0.0_0/post/clean/glitch_median_PSD_forLI_H1.dat",
    "L1": "/home/pe.o4/GWTC4/working/S231123cg/generate-psd/trigtime_1384782888.634277105_0.0_0.0_0/post/clean/glitch_median_PSD_forLI_L1.dat",
    }

channel_dict = {
        "H1": "H1:GDS-CALIB_STRAIN_CLEAN_BAYESWAVE_S00",
        "L1": "L1:GDS-CALIB_STRAIN_CLEAN_AR" 
    }

#defyining IFOs
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
for ifo in ifos:
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        psd_file=psd_files[ifo.name]
    )
    ifo.maximum_frequency = 448.0
    ifo.minimum_frequency = 20.0
    ifo.duration = 8.0
    ifo.sampling_frequency = 1024
    print('here')
    _data = TimeSeries.get(channel=channel_dict[ifo.name],start=1384782882.634277, end=1384782890.634277, verbose=False, allow_tape=True)
    _data = _data.crop(start=1384782888.634277 - 6., end=1384782888.634277 +2)
    print('here 2')
    lal_timeseries = _data.to_lal()
    lal.ResampleREAL8TimeSeries(
        lal_timeseries, float(1 / ifo.sampling_frequency)
    )
    _data = TimeSeries(
        lal_timeseries.data.data,
        epoch=lal_timeseries.epoch,
        dt=lal_timeseries.deltaT,
    )
    ifo.set_strain_data_from_gwpy_timeseries(_data)
    ifo.calibration_model = bilby.gw.detector.calibration.CubicSpline(
        prefix=f"recalib_{ifo.name}_",
        minimum_frequency=ifo.minimum_frequency,
        maximum_frequency=ifo.maximum_frequency,
        n_points=10,
    )

#Opening PE samples
result_NRSUR = bilby.core.result.read_in_result(config.gw231123_BF_NRSUR)
result_XO4a = bilby.core.result.read_in_result(config.gw231123_BF_XO4a)

posterior_NRSUR = result_NRSUR.posterior
posterior_XO4a = result_XO4a.posterior

#find max likelihood
maxL_ind_NRSUR = np.argmax(posterior_NRSUR["log_likelihood"])
maxL_params_NRSUR = posterior_NRSUR.iloc[maxL_ind_NRSUR].to_dict()

maxL_ind_XO4a = np.argmax(posterior_XO4a["log_likelihood"])
maxL_params_XO4a = posterior_XO4a.iloc[maxL_ind_XO4a].to_dict()

print('NRSUR max L:', maxL_params_NRSUR)
print('XO4a max L:', maxL_params_XO4a)

#generate whitened waveform (NRSUR)
pols_NRSUR = waveform_generator_NRSUR.frequency_domain_strain(parameters=maxL_params_NRSUR)
h_NRSUR = {}
for ifo in ifos:
    h_NRSUR[ifo.name] = ifo.get_detector_response(pols_NRSUR, maxL_params_NRSUR)

h_white_maxL_NRSUR = {}
for ifo in ifos:
    frequency_window_factor = (
        np.sum(ifo.frequency_mask)
        / len(ifo.frequency_mask)
    )
    ht_NRSUR = h_NRSUR[ifo.name] / (ifo.amplitude_spectral_density_array * np.sqrt(ifo.duration / 4))
    h_white_maxL_NRSUR[ifo.name] = (
        np.fft.irfft(ht_NRSUR)
        * np.sqrt(np.sum(ifo.frequency_mask)) / frequency_window_factor
    )

#add 90% uncertainty
inds = np.random.choice(np.arange(len(posterior_NRSUR["chirp_mass"])), size=1000)
h_white_NRSUR = {"H1": [], "L1": []}
for ii in tqdm.tqdm(inds):
    params = posterior_NRSUR.iloc[ii].to_dict()
    pols_NRSUR = waveform_generator_NRSUR.frequency_domain_strain(parameters=params)
    for ifo in ifos:
        h_NRSUR = ifo.get_detector_response(pols_NRSUR, params)
        frequency_window_factor = (
            np.sum(ifo.frequency_mask)
            / len(ifo.frequency_mask)
        )
        ht_NRSUR = h_NRSUR / (ifo.amplitude_spectral_density_array * np.sqrt(ifo.duration / 4))
        h_white_NRSUR[ifo.name].append(
            np.fft.irfft(ht_NRSUR)
            * np.sqrt(np.sum(ifo.frequency_mask)) / frequency_window_factor
        )

# same for XO4a
pols_XO4a = waveform_generator_XO4a.frequency_domain_strain(parameters=maxL_params_XO4a)
h_XO4a = {}
for ifo in ifos:
    h_XO4a[ifo.name] = ifo.get_detector_response(pols_XO4a, maxL_params_XO4a)

h_white_maxL_XO4a = {}
for ifo in ifos:
    frequency_window_factor = (
        np.sum(ifo.frequency_mask)
        / len(ifo.frequency_mask)
    )
    ht_XO4a = h_XO4a[ifo.name] / (ifo.amplitude_spectral_density_array * np.sqrt(ifo.duration / 4))
    h_white_maxL_XO4a[ifo.name] = (
        np.fft.irfft(ht_XO4a)
        * np.sqrt(np.sum(ifo.frequency_mask)) / frequency_window_factor
    )

#add 90% uncertainty
inds = np.random.choice(np.arange(len(posterior_XO4a["chirp_mass"])), size=1000)
h_white_XO4a = {"H1": [], "L1": []}
for ii in tqdm.tqdm(inds):
    params = posterior_XO4a.iloc[ii].to_dict()
    pols_XO4a = waveform_generator_XO4a.frequency_domain_strain(parameters=params)
    for ifo in ifos:
        h_XO4a = ifo.get_detector_response(pols_XO4a, params)
        frequency_window_factor = (
            np.sum(ifo.frequency_mask)
            / len(ifo.frequency_mask)
        )
        ht_XO4a = h_XO4a / (ifo.amplitude_spectral_density_array * np.sqrt(ifo.duration / 4))
        h_white_XO4a[ifo.name].append(
            np.fft.irfft(ht_XO4a)
            * np.sqrt(np.sum(ifo.frequency_mask)) / frequency_window_factor
        )

#Plot
nrsur_color= 'darkviolet'
xo4a_color='green'

fig, axs = plt.subplots(figsize=(10, 8), nrows=4, sharex=True, gridspec_kw={'height_ratios': [4, 1, 4, 1]})

#Hanford
axs[0].plot(waveform_generator_NRSUR.time_array, h_white_maxL_NRSUR["H1"], label='NRSur', color=nrsur_color)
axs[0].fill_between(waveform_generator_NRSUR.time_array, np.percentile(h_white_NRSUR["H1"], 5, axis=0), np.percentile(h_white_NRSUR["H1"], 95, axis=0),color=nrsur_color, alpha=0.2)

axs[0].plot(waveform_generator_XO4a.time_array, h_white_maxL_XO4a["H1"], label='XO4a', color=xo4a_color)
axs[0].fill_between(waveform_generator_XO4a.time_array, np.percentile(h_white_XO4a["H1"], 5, axis=0), np.percentile(h_white_XO4a["H1"], 95, axis=0),color=xo4a_color, alpha=0.2)

axs[0].set_title("LIGO Hanford", fontsize=24)
axs[0].legend(loc='upper left', fontsize=25, framealpha=0.7)
axs[0].set_xlim([maxL_params_NRSUR["geocent_time"] - 0.15, maxL_params_NRSUR["geocent_time"] + 0.13])
axs[0].set_ylim(-4.5, 4.5)
axs[0].set_ylabel(r'$\sigma$ noise', fontsize =22)

axs[1].plot(waveform_generator_NRSUR.time_array, h_white_maxL_NRSUR["H1"]-h_white_maxL_XO4a["H1"], color='black')
axs[1].set_ylabel(r'Difference', fontsize =22)
axs[1].set_ylim(-1, 1)

#Livingston
axs[2].plot(waveform_generator_NRSUR.time_array, h_white_maxL_NRSUR["L1"], label='NRSur', color=nrsur_color)
axs[2].fill_between(waveform_generator_NRSUR.time_array, np.percentile(h_white_NRSUR["L1"], 5, axis=0), np.percentile(h_white_NRSUR["L1"], 95, axis=0),color=nrsur_color, alpha=0.2)

axs[2].plot(waveform_generator_XO4a.time_array, h_white_maxL_XO4a["L1"], label='XO4a', color=xo4a_color)
axs[2].fill_between(waveform_generator_XO4a.time_array, np.percentile(h_white_XO4a["L1"], 5, axis=0), np.percentile(h_white_XO4a["L1"], 95, axis=0),color=xo4a_color, alpha=0.2)

axs[2].set_title("LIGO Livingston", fontsize=24)
axs[2].set_ylim(-4.5, 4.5)
axs[2].set_ylabel(r'$\sigma$ noise', fontsize =22)

axs[3].plot(waveform_generator_NRSUR.time_array, h_white_maxL_NRSUR["L1"]-h_white_maxL_XO4a["L1"], color='black')
axs[3].set_xlabel('Time [s]', fontsize =24)
axs[3].set_ylabel(r'Difference', fontsize =22)
axs[3].set_ylim(-1,1)

fig.align_ylabels(axs)
plt.tight_layout()
