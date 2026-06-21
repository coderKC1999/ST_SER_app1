import numpy as np

from scipy.signal import get_window

import matplotlib.pyplot as plt


def stockwell_spectrogram(

    x,

    Fs,

    window_length=255,

    overlap=0.5,

    fmin=0,

    fmax=4000

):

    N = len(x)

    hop_length = int(

        window_length*(1-overlap)

    )

    t_start = (

        window_length//2

    )/Fs

    t_end = (

        N-window_length//2

    )/Fs

    t = np.arange(

        t_start,

        t_end,

        hop_length/Fs

    )

    f = np.arange(

        fmin,

        fmax+1,

        10

    )

    S = np.zeros(

        (

            len(f),

            len(t)

        )

    )

    for i in range(len(t)):

        center = t[i]

        start = max(

            0,

            int(

                round(

                    center*Fs-window_length/2

                )

            )

        )

        end = min(

            N,

            int(

                round(

                    center*Fs+window_length/2

                )

            )

        )

        xw = x[start:end]

        win = get_window(

            "hamming",

            len(xw)

        )

        xw = xw*win

        for j,current_f in enumerate(f):

            if current_f > 0:

                n = np.arange(

                    len(xw)

                )

                gauss = np.exp(

                    -(n**2)/(2*(current_f*3)**2)

                )

                exp_term = np.exp(

                    -1j*2*np.pi*

                    current_f*n/Fs

                )

                st = np.sum(

                    xw*

                    exp_term*

                    gauss

                )

                S[j,i] = np.abs(st)

    return S,f,t


def save_spectrogram(S, f, t, filename):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(4,4))

    plt.imshow(
        S,
        extent=[t[0], t[-1], f[0], f[-1]],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )

    plt.axis('off')

    plt.tight_layout()

    plt.savefig(
        filename,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close()
