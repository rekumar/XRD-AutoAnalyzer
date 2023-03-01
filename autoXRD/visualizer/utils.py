import numpy as np
import math


def XRDtoPDF(xrd, min_angle, max_angle):

    thetas = np.linspace(min_angle / 2.0, max_angle / 2.0, 4501)
    Q = np.array(
        [4 * math.pi * math.sin(math.radians(theta)) / 1.5406 for theta in thetas]
    )
    S = np.array(xrd).flatten()

    pdf = []
    R = np.linspace(1, 40, 1000)  # Only 1000 used to reduce compute time
    integrand = Q * S * np.sin(Q * R[:, np.newaxis])

    pdf = 2 * np.trapz(integrand, Q) / math.pi

    return R, pdf
