import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks


def detect_seasonal_period(
    series,
    max_lag=None,
    confidence=0.95,
    plot=True
):
    """
    Detect seasonal period using autocorrelation.

    Parameters
    ----------
    series : array-like or pandas Series
        Time series values
    max_lag : int, optional
        Maximum lag to compute ACF (default: len(series)//2)
    confidence : float
        Confidence level for significance (default 0.95)
    plot : bool
        Whether to plot ACF

    Returns
    -------
    period : int or None
        Estimated seasonal period
    """

    series = pd.Series(series).dropna()
    n = len(series)

    if max_lag is None:
        max_lag = n // 2

    # Compute autocorrelation
    acf_values = acf(series, nlags=max_lag, fft=True)

    # Confidence interval threshold
    conf_threshold = 1.96 / np.sqrt(n)

    # Find significant peaks (ignore lag 0)
    peaks, properties = find_peaks(
        acf_values[1:],
        height=conf_threshold
    )

    if len(peaks) == 0:
        print("No significant seasonality detected.")
        return None

    # Convert to actual lag numbers
    seasonal_lags = peaks + 1

    # First significant peak is usually the seasonal period
    period = seasonal_lags[0]

    if plot:
        plt.figure(figsize=(10, 4))
        plt.stem(range(len(acf_values)), acf_values, use_line_collection=True)
        plt.axhline(conf_threshold, color='red', linestyle='--', alpha=0.7)
        plt.axhline(-conf_threshold, color='red', linestyle='--', alpha=0.7)
        plt.axvline(period, color='green', linestyle='--', label=f"Detected period = {period}")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title("Autocorrelation Function (ACF)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return period


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Create synthetic seasonal data (period = 12)
    np.random.seed(0)
    t = np.arange(120)
    series = np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.3, size=len(t))

    period = detect_seasonal_period(series)
    print(f"Detected seasonal period: {period}")
