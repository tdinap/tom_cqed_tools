import numpy as np
from laboneq.simple import pulse_library
from scipy.special import erf

#reworking this to make sb, bs, and whatever else into easily usable arrays

# a collection of qubit control and readout parameters as a python dictionary
def single_qubit_parameters():
    return {
        # qb drive settings
        "qb_freq": 4222510746.2798233,
        "qb_pi_len": 62e-9,
        "qb_pi_amp": 0.4963648431759063,
        "qb_ef_freq": 4104967450.626856,
        "qb_ef_len": 50e-9,
        "qb_ef_amp": 0.4942679447349856,
        "qb_drive_dBm_range": 10,
        "qb_drive_resolved_dBm_range": -5,  
        'qb_resolved_freq': 4221728215.756359,
        "qb_resolved_pi_len": 300e-9,
        "qb_resolved_pi_amp": 0.5732342040693759,
        "qb_resolved_pi_ramp_len": 100e-9,
        "qb_resolved_pi_muted_amp": 1,  
        "qb_ge_pulse_type": "gaussian",  # "gaussian" or "const"
        "qb_ef_pulse_type": "gaussian",  # "gaussian" or "const"
        "qb_resolved_pulse_type": "gaussian",  # "gaussian" or "const"
        # measurement settings
        "reset_delay": 0.25e-3,  # 300e-6,  # delay time after each measurement for qubit reset in [s]
        "cavity_reset_delay": 3e-3,
        "ro_len": 2e-6,  # 2e-6,
        "ro_freq": 7683273333.333334,
        "ro_amp": 0.9333333333333333,  #
        "ro_delay": 0e-9,
        "ro_int_delay": 0e-9,  # time of flight
        "ro_drive_dBm_range": 10,
        "ro_acq_dBm_range": 0,
        "acquire_delay": 0.25e-6,  # 0.9e-6,
        # sideband pulse settings (now arrays; 8 entries f0g1..f7g8)
        "sb_alice_freqs": [
            4015973185.5574927,
            100e6,
            100e6 - 2 * 0.6e6,
            100e6 - 3 * 0.6e6,
            100e6 - 4 * 0.6e6,
            100e6 - 5 * 0.6e6,
            100e6 - 6 * 0.6e6,
            100e6 - 7 * 0.6e6,
        ],
        "sb_alice_flat_lens": [
            156e-9, 2e-6, 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6
        ],
        "sb_alice_amps": [
            0.9964890077248618, 1, 1, 1, 1, 1, 1, 1
        ],
        "sb_alice_ramp_lens": [
            20e-9, 20e-9, 20e-9, 160e-9, 160e-9, 160e-9, 160e-9, 160e-9
        ],
        "sb_alice_dBm_range": 10,

        "sb_bob_freqs": [
            3684008774.3045006,
            100e6,
            100e6 - 2 * 0.6e6,
            100e6 - 3 * 0.6e6,
            100e6 - 4 * 0.6e6,
            100e6 - 5 * 0.6e6,
            100e6 - 6 * 0.6e6,
            100e6 - 7 * 0.6e6,
        ],
        "sb_bob_flat_lens": [
            6.35e-07,
            10e-6,
            2.5e-6,
            2.5e-6,
            2.5e-6,
            2.5e-6,
            2.5e-6,
            2.5e-6,
        ],
        "sb_bob_amps": [
            0.9910330896752024, 1, 1, 1, 1, 1, 1, 1
        ],
        "sb_bob_ramp_lens": [
            50e-9, 500e-9, 500e-9, 500e-9, 500e-9, 500e-9, 500e-9, 500e-9
        ],
        "sb_bob_dBm_range": 10,

        # Alice beamsplitter pulse settings (now arrays)
        "bs_alice_freqs": [1000000000.0,
1341861935.44266,
1594537194.4266164,
1855316756.8108354,
2109751301.363879,
2384237038.4939303,
2641349512.194346,
2900362070.334666]
,
        "bs_alice_fwhms": [1000000000.0,
331699.10895569646,
326858.5190947526,
338362.0602459558,
336334.9816809524,
212512.4866205752,
341845.6668327075,
335166.14698925195]
,
        "bs_alice_flat_lens": [
            2.9684131396834543e-07,
            2e-6,
            2e-6,
            2e-06,
            2e-06,
            2e-6,
            2e-6,
            2e-6,
        ],
        "bs_alice_dBm_ranges": [10, 10, 10, 10, 10, 10, 10, 10],
        "bs_alice_amps": [0.5,
0.40842461557777027,
0.2550554722331442,
0.2996891027374975,
0.19926032771655244,
0.2788420452888216,
0.20321380662725252,
0.19219287119822695],
        "bs_alice_ramp_lens": [1.5e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9],
        # Bob beamsplitter pulse settings (now arrays)
        "bs_bob_freqs": [1000000000.0,
1016366522.1309078,
1269336040.2510448,
1529991967.0353997,
1784440544.6761785,
2050039001.0967374,
2322171766.7909393,
2575056856.0720797]
,
        "bs_bob_fwhms": [1000000000.0,
324138.9671759798,
330410.2709393852,
333558.4624060139,
339708.8840517704,
340568.77033350477,
356622.9620983741,
325129.9580249167]
,
        "bs_bob_flat_lens": [
            3e-07,
            2e-06,
            2e-6,
            2e-06,
            2e-06,
            2.5e-6,
            2e-6,
            2e-6,
        ],
        "bs_bob_dBm_ranges": [10, 10, 10, 10, 10, 10, 10, 10],
        "bs_bob_amps": [0.5,
0.4840190421566057,
0.37637700403955243,
0.2860420988034554,
0.33206545130154813,
0.2247520495030237,
0.057056256656539774,
0.2996744183710269],
        "bs_bob_ramp_lens": [10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9],

        ###### alice
        "cav_alice_freq": 4.320895543334294e9,
        "cav_alice_len":  10.054260733652825e-6,#4.29687616482541e-6,
        "cav_alice_amp": 0.4,  #
        "cav_alice_dBm_range": -30,
        "cav_alice_chi": -10.00116116e6,
        "parity_time": 2e-6,

        ###### bob
        "cav_bob_freq": 4.6470891102e9,#4.642074442552949e9,
        "cav_bob_len": 6.896462509807999e-6,
        "cav_bob_amp": 1,
        "cav_bob_dBm_range": -24,
        "cav_bob_chi": -0.53467169e6,
        "parity_time": 2e-6,

    }


def shfqc_lo_settings(serial_num):  # Need to be in multiples of 200 MHz
    return {
        serial_num: {
            # SHFQA LO Frequency
            "QA0_LO": 7.6e9,  #
            # SHFSG LO Frequencies, one center frequency per two channels on SHFQC
            "SG0_LO": 4.0e9,  #
            "SG2_LO": 4.6e9,  #
            "SG4_LO": 1e9,#2.4e9,
        }
    }


def create_lo_settings(serial_num):
    return {"q0": shfqc_lo_settings(serial_num)}


def create_qubit_parameters():
    return {"q0": single_qubit_parameters()}


qubit_parameters = create_qubit_parameters()

########################################################################################
## Pulse definitions
########################################################################################

# def gaussian_second_half(x, **pulse_params):
#     sigma=1 / 3
#     return np.exp(-((x+1.)**2 / (2 * sigma**2)))
# def gaussian_first_half(x, **pulse_params):
#     sigma=1 / 3
#     return np.exp(-((x-1.)**2 / (2 * sigma**2)))


def gaussian_square_custom(
    x, sigma=1/3, ramp=10e-9, zero_boundaries=False, length=100e-9, **_
):
    """Create a gaussian square waveform with a Gaussian shaped ramp up/down portion
    of length ``ramp`` and a flat top.

    Arguments:
        **_ (Any):
            All pulses accept the following keyword arguments:
            - uid ([str][]): Unique identifier of the pulse
            - length ([float][]): Length of the pulse in seconds
            - amplitude ([float][]): Amplitude of the pulse
        ramp (float):
            Gaussian rise/fall length in seconds
        sigma (float):
            Std. deviation of the Gaussian rise/fall portion of the pulse
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Returns:
        pulse (Pulse): Gaussian square pulse.
    """
    num_samples = len(x)
    risefall_in_samples = int(np.round(ramp * num_samples / length))
    flat_in_samples = num_samples - 2 * risefall_in_samples
    
    # Handle very short pulses where ramp > pulse_length
    if flat_in_samples <= 0:
        # For very short pulses, just use a gaussian without the flat part
        gauss_x = np.linspace(-1.0, 1.0, num_samples)
        gauss_part = np.exp(-(gauss_x**2) / (2 * sigma**2))
        return gauss_part
    
    gauss_x = np.linspace(-1.0, 1.0, 2 * risefall_in_samples)
    gauss_part = np.exp(-(gauss_x**2) / (2 * sigma**2))
    gauss_sq = np.concatenate(
        (
            gauss_part[:risefall_in_samples],
            np.ones(flat_in_samples),
            gauss_part[risefall_in_samples:],
        )
    )
    
    if zero_boundaries:
        t_left = gauss_x[0] - (gauss_x[1] - gauss_x[0])
        delta = np.exp(-(t_left**2) / (2 * sigma**2))
        gauss_sq -= delta
        gauss_sq /= 1 - delta
    return gauss_sq
custom_gaussian_square = pulse_library.register_pulse_functional(
    sampler=gaussian_square_custom, name="gaussian_square_custom"
)

# # Basic pulse definitions
readout_pulse = pulse_library.const(
    uid="ro_pulse",
    length=qubit_parameters["q0"]["ro_len"],
    amplitude=qubit_parameters["q0"]["ro_amp"],
)
acquire_kernel = pulse_library.const(
    uid="acquire_kernel",
    length=qubit_parameters["q0"]["ro_len"] - qubit_parameters["q0"]["acquire_delay"], 
    # amplitude=qubit_parameters["q0"]["ro_amp"],
)

if qubit_parameters["q0"]["qb_ge_pulse_type"] == "const":
    ge_X180 = pulse_library.const(
        uid="ge_X180_pulse",
        length=qubit_parameters["q0"]["qb_pi_len"],
        amplitude=qubit_parameters["q0"]["qb_pi_amp"],
        can_compress=True,
    )
    ge_X90 = pulse_library.const(
        uid="ge_X90_pulse",
        length=qubit_parameters["q0"]["qb_pi_len"],
        amplitude=0.5 * qubit_parameters["q0"]["qb_pi_amp"],
        can_compress=True,
    )

elif qubit_parameters["q0"]["qb_ge_pulse_type"] == "gaussian":
    ge_X180 = pulse_library.gaussian(
        uid="ge_X180_pulse",
        length=qubit_parameters["q0"]["qb_pi_len"],
        amplitude=qubit_parameters["q0"]["qb_pi_amp"],
    )
    ge_X90 = pulse_library.gaussian(
        uid="ge_X90_pulse",
        length=0.5 * qubit_parameters["q0"]["qb_pi_len"],
        amplitude=qubit_parameters["q0"]["qb_pi_amp"],
    )

if qubit_parameters["q0"]["qb_ef_pulse_type"] == "const":
    ef_X180 = pulse_library.const(
        uid="ef_X180_pulse",
        length=qubit_parameters["q0"]["qb_ef_len"],
        amplitude=qubit_parameters["q0"]["qb_ef_amp"],
        can_compress=True,
    )
elif qubit_parameters["q0"]["qb_ef_pulse_type"] == "gaussian":
    ef_X180 = pulse_library.gaussian(
        uid="ef_X180_pulse",
        length=qubit_parameters["q0"]["qb_ef_len"],
        amplitude=qubit_parameters["q0"]["qb_ef_amp"],
    )


if qubit_parameters["q0"]["qb_resolved_pulse_type"] == "const":
    resolved_X180 = pulse_library.const(
        uid="resolved_X180_pulse",
        length=qubit_parameters["q0"]["qb_resolved_pi_len"],
        amplitude=qubit_parameters["q0"]["qb_resolved_pi_amp"],
        can_compress=True,
    )
elif qubit_parameters["q0"]["qb_resolved_pulse_type"] == "gaussian":
    resolved_X180 = pulse_library.gaussian(
        uid="resolved_X180_pulse",
        length=qubit_parameters["q0"]["qb_resolved_pi_len"],
        amplitude=qubit_parameters["q0"]["qb_resolved_pi_amp"],
    )
elif qubit_parameters["q0"]["qb_resolved_pulse_type"] == "flat_top_gaussian":
    resolved_X180 = custom_gaussian_square(uid="resolved_X180_pulse", 
                                                  length = qubit_parameters["q0"]["qb_resolved_pi_len"] + qubit_parameters["q0"]["qb_resolved_pi_ramp_len"], 
                                                  ramp = qubit_parameters["q0"]["qb_resolved_pi_ramp_len"],
                                                  amplitude = qubit_parameters["q0"]["qb_resolved_pi_amp"], 
                                                  can_compress = True)
    # resolved_X180 = pulse_library.gaussian(uid="resolved_X180_pulse", length = qubit_parameters["q0"]["qb_resolved_pi_len"], amplitude = qubit_parameters["q0"]["qb_resolved_pi_amp"])

resolved_X180_muted = pulse_library.gaussian(
    uid="resolved_X180_pulse_muted",
    length=qubit_parameters["q0"]["qb_resolved_pi_len"],
    amplitude=qubit_parameters["q0"]["qb_resolved_pi_muted_amp"],
)

sb_pulses = {"alice": {}, "bob": {}}
for name in sb_pulses.keys():
    if name == "alice":
        sb_flat = qubit_parameters["q0"]["sb_alice_flat_lens"]
        sb_ramp = qubit_parameters["q0"]["sb_alice_ramp_lens"]
        sb_amp = qubit_parameters["q0"]["sb_alice_amps"]
    else:
        sb_flat = qubit_parameters["q0"]["sb_bob_flat_lens"]
        sb_ramp = qubit_parameters["q0"]["sb_bob_ramp_lens"]
        sb_amp = qubit_parameters["q0"]["sb_bob_amps"]

    for st in range(8): # indexing sideband transitions f0g1 to f7g8
        sb_pulses[name][f"f{st}g{st+1}"] = custom_gaussian_square(
            uid=f"sb_f{st}g{st+1}_{name}_pulse",
            length=sb_flat[st]+2*sb_ramp[st],
            ramp=sb_ramp[st],
            amplitude=sb_amp[st],
            can_compress=True,
        )
    # Build beamsplitter pulses from array settings
    if name == "alice":
        bs_flat = qubit_parameters["q0"]["bs_alice_flat_lens"]
        bs_ramp = qubit_parameters["q0"]["bs_alice_ramp_lens"]
        bs_amp = qubit_parameters["q0"]["bs_alice_amps"]
    else:
        bs_flat = qubit_parameters["q0"]["bs_bob_flat_lens"]
        bs_ramp = qubit_parameters["q0"]["bs_bob_ramp_lens"]
        bs_amp = qubit_parameters["q0"]["bs_bob_amps"]

    for bs_idx in range(len(bs_flat)):
        sb_pulses[name][f"bs{bs_idx}"] = custom_gaussian_square(
            uid=f"bs{bs_idx}_{name}_pulse",
            length=bs_flat[bs_idx]+2*bs_ramp[bs_idx],
            ramp=bs_ramp[bs_idx],
            amplitude=bs_amp[bs_idx],
            can_compress=True,
        )

# Cavity pulse definitions
cav_alice = pulse_library.const(
    uid="cav_alice_pulse",
    length=qubit_parameters["q0"]["cav_alice_len"],
    amplitude=qubit_parameters["q0"]["cav_alice_amp"],
    can_compress=True,
)
cav_bob = pulse_library.const(
    uid="cav_bob_pulse",
    length=qubit_parameters["q0"]["cav_bob_len"],
    amplitude=qubit_parameters["q0"]["cav_bob_amp"],
    can_compress=True,
)
# cav_bob = pulse_library.gaussian(uid = "cav_bob_pulse", length = qubit_parameters["q0"]["cav_bob_len"], amplitude = qubit_parameters["q0"]["cav_bob_amp"])
