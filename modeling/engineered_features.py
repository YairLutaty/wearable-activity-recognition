"""Deterministic 222-feature extraction for paired watch sequences."""

from __future__ import annotations

import numpy as np
from scipy.signal import periodogram


SIGNAL_NAMES = (
    "x",
    "y",
    "z",
    "magnitude",
)

BASELINE_STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "rms",
)

DISTRIBUTION_STATISTICS = (
    "p10",
    "p25",
    "p75",
    "p90",
    "skewness",
    "excess_kurtosis",
)

AXIS_PAIRS = (
    (0, 1, "xy"),
    (0, 2, "xz"),
    (1, 2, "yz"),
)

TEMPORAL_STATISTICS = (
    "mean_abs",
    "std",
    "rms",
    "max_abs",
)

FREQUENCY_STATISTICS = (
    "total_power",
    "dominant_frequency_hz",
    "dominant_power_ratio",
    "spectral_centroid_hz",
    "spectral_entropy",
    "relative_power_0p2_1hz",
    "relative_power_1_3hz",
    "relative_power_3_5hz",
    "relative_power_5_10hz",
)

FREQUENCY_BANDS = (
    (
        "relative_power_0p2_1hz",
        0.2,
        1.0,
        False,
    ),
    (
        "relative_power_1_3hz",
        1.0,
        3.0,
        False,
    ),
    (
        "relative_power_3_5hz",
        3.0,
        5.0,
        False,
    ),
    (
        "relative_power_5_10hz",
        5.0,
        10.0,
        True,
    ),
)

FEATURE_GROUP_SIZES = {
    "baseline": 56,
    "distribution_correlation": 54,
    "temporal_dynamics": 40,
    "frequency_structure": 72,
}

TOTAL_FEATURES = sum(
    FEATURE_GROUP_SIZES.values()
)


def _add_magnitude(sequences: np.ndarray) -> np.ndarray:
    xyz = np.asarray(
        sequences,
        dtype=np.float64,
    )

    magnitude = np.sqrt(
        np.sum(
            np.square(xyz),
            axis=2,
            keepdims=True,
        )
    )

    return np.concatenate(
        [xyz, magnitude],
        axis=2,
    )


def _extract_baseline_features(
    signals: np.ndarray,
    sensor_prefix: str,
) -> tuple[np.ndarray, list[str]]:
    signals = signals.astype(
        np.float64,
        copy=False,
    )

    first_quartile, third_quartile = np.quantile(
        signals,
        [0.25, 0.75],
        axis=1,
        method="linear",
    )

    statistic_values = {
        "mean": signals.mean(axis=1),
        "std": signals.std(
            axis=1,
            ddof=1,
        ),
        "min": signals.min(axis=1),
        "max": signals.max(axis=1),
        "median": np.median(
            signals,
            axis=1,
        ),
        "iqr": (
            third_quartile
            - first_quartile
        ),
        "rms": np.sqrt(
            np.mean(
                np.square(signals),
                axis=1,
            )
        ),
    }

    feature_names = [
        (
            f"{sensor_prefix}_"
            f"{signal_name}_"
            f"{statistic_name}"
        )
        for signal_name in SIGNAL_NAMES
        for statistic_name in BASELINE_STATISTICS
    ]

    feature_matrix = np.column_stack([
        statistic_values[
            statistic_name
        ][:, signal_index]
        for signal_index, _
        in enumerate(SIGNAL_NAMES)
        for statistic_name in BASELINE_STATISTICS
    ])

    return feature_matrix, feature_names


def _extract_distribution_features(
    signals: np.ndarray,
    sensor_prefix: str,
) -> tuple[np.ndarray, list[str]]:
    signals = signals.astype(
        np.float64,
        copy=False,
    )

    (
        percentile_10,
        percentile_25,
        percentile_75,
        percentile_90,
    ) = np.quantile(
        signals,
        [0.10, 0.25, 0.75, 0.90],
        axis=1,
        method="linear",
    )

    centered = (
        signals
        - signals.mean(
            axis=1,
            keepdims=True,
        )
    )

    second_moment = np.mean(
        centered**2,
        axis=1,
    )
    third_moment = np.mean(
        centered**3,
        axis=1,
    )
    fourth_moment = np.mean(
        centered**4,
        axis=1,
    )

    nonconstant = (
        second_moment
        > np.finfo(np.float64).eps
    )

    skewness = np.zeros_like(
        second_moment
    )
    excess_kurtosis = np.zeros_like(
        second_moment
    )

    np.divide(
        third_moment,
        second_moment**1.5,
        out=skewness,
        where=nonconstant,
    )

    np.divide(
        fourth_moment,
        second_moment**2,
        out=excess_kurtosis,
        where=nonconstant,
    )

    excess_kurtosis[
        nonconstant
    ] -= 3.0

    statistic_values = {
        "p10": percentile_10,
        "p25": percentile_25,
        "p75": percentile_75,
        "p90": percentile_90,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
    }

    distribution_names = [
        (
            f"{sensor_prefix}_"
            f"{signal_name}_"
            f"{statistic_name}"
        )
        for signal_name in SIGNAL_NAMES
        for statistic_name
        in DISTRIBUTION_STATISTICS
    ]

    distribution_matrix = np.column_stack([
        statistic_values[
            statistic_name
        ][:, signal_index]
        for signal_index, _
        in enumerate(SIGNAL_NAMES)
        for statistic_name
        in DISTRIBUTION_STATISTICS
    ])

    xyz = signals[:, :, :3]

    centered_xyz = (
        xyz
        - xyz.mean(
            axis=1,
            keepdims=True,
        )
    )

    correlation_columns = []
    correlation_names = []

    for (
        first_axis,
        second_axis,
        pair_name,
    ) in AXIS_PAIRS:
        first_values = centered_xyz[
            :, :, first_axis
        ]
        second_values = centered_xyz[
            :, :, second_axis
        ]

        numerator = np.sum(
            first_values * second_values,
            axis=1,
        )

        denominator = np.sqrt(
            np.sum(
                first_values**2,
                axis=1,
            )
            * np.sum(
                second_values**2,
                axis=1,
            )
        )

        correlation = np.zeros(
            len(signals),
            dtype=np.float64,
        )

        np.divide(
            numerator,
            denominator,
            out=correlation,
            where=(
                denominator
                > np.finfo(
                    np.float64
                ).eps
            ),
        )

        correlation_columns.append(
            correlation
        )
        correlation_names.append(
            (
                f"{sensor_prefix}_"
                f"{pair_name}_correlation"
            )
        )

    feature_matrix = np.column_stack([
        distribution_matrix,
        *correlation_columns,
    ])

    feature_names = (
        distribution_names
        + correlation_names
    )

    return feature_matrix, feature_names


def _extract_temporal_features(
    signals: np.ndarray,
    sensor_prefix: str,
    step_s: float,
) -> tuple[np.ndarray, list[str]]:
    signals = signals.astype(
        np.float64,
        copy=False,
    )

    rates = (
        np.diff(
            signals,
            axis=1,
        )
        / step_s
    )

    absolute_rates = np.abs(rates)

    statistic_values = {
        "mean_abs": absolute_rates.mean(
            axis=1
        ),
        "std": rates.std(
            axis=1,
            ddof=1,
        ),
        "rms": np.sqrt(
            np.mean(
                rates**2,
                axis=1,
            )
        ),
        "max_abs": absolute_rates.max(
            axis=1
        ),
    }

    signal_feature_names = [
        (
            f"{sensor_prefix}_"
            f"{signal_name}_rate_"
            f"{statistic_name}"
        )
        for signal_name in SIGNAL_NAMES
        for statistic_name
        in TEMPORAL_STATISTICS
    ]

    signal_feature_matrix = np.column_stack([
        statistic_values[
            statistic_name
        ][:, signal_index]
        for signal_index, _
        in enumerate(SIGNAL_NAMES)
        for statistic_name
        in TEMPORAL_STATISTICS
    ])

    vector_rate_magnitude = np.sqrt(
        np.sum(
            rates[:, :, :3] ** 2,
            axis=2,
        )
    )

    vector_feature_matrix = np.column_stack([
        vector_rate_magnitude.mean(
            axis=1
        ),
        vector_rate_magnitude.std(
            axis=1,
            ddof=1,
        ),
        np.sqrt(
            np.mean(
                vector_rate_magnitude**2,
                axis=1,
            )
        ),
        vector_rate_magnitude.max(
            axis=1
        ),
    ])

    vector_feature_names = [
        (
            f"{sensor_prefix}_"
            "vector_rate_magnitude_mean"
        ),
        (
            f"{sensor_prefix}_"
            "vector_rate_magnitude_std"
        ),
        (
            f"{sensor_prefix}_"
            "vector_rate_magnitude_rms"
        ),
        (
            f"{sensor_prefix}_"
            "vector_rate_magnitude_max"
        ),
    ]

    feature_matrix = np.column_stack([
        signal_feature_matrix,
        vector_feature_matrix,
    ])

    feature_names = (
        signal_feature_names
        + vector_feature_names
    )

    return feature_matrix, feature_names


def _extract_frequency_features(
    signals: np.ndarray,
    sensor_prefix: str,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, list[str]]:
    signals = signals.astype(
        np.float64,
        copy=False,
    )

    frequencies, power_density = periodogram(
        signals,
        fs=sampling_rate_hz,
        window="hann",
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=1,
    )

    if not np.isclose(
        frequencies[1] - frequencies[0],
        0.2,
    ):
        raise ValueError(
            "Expected the canonical 0.2 Hz frequency resolution."
        )

    if not np.isclose(
        frequencies[-1],
        sampling_rate_hz / 2,
    ):
        raise ValueError(
            "Unexpected final one-sided frequency bin."
        )

    positive_frequencies = frequencies[1:]
    positive_power = power_density[:, 1:, :]

    frequency_step_hz = float(
        positive_frequencies[1]
        - positive_frequencies[0]
    )

    power_sum = positive_power.sum(
        axis=1
    )

    valid_power = (
        power_sum
        > np.finfo(np.float64).eps
    )

    safe_power_sum = np.where(
        valid_power,
        power_sum,
        1.0,
    )

    total_power = (
        power_sum * frequency_step_hz
    )

    dominant_indices = np.argmax(
        positive_power,
        axis=1,
    )

    dominant_frequency = (
        positive_frequencies[
            dominant_indices
        ]
    ).astype(np.float64)

    dominant_power = np.take_along_axis(
        positive_power,
        dominant_indices[:, None, :],
        axis=1,
    ).squeeze(axis=1)

    dominant_power_ratio = (
        dominant_power
        / safe_power_sum
    )

    spectral_centroid = (
        (
            positive_power
            * positive_frequencies[
                None, :, None
            ]
        ).sum(axis=1)
        / safe_power_sum
    )

    normalized_power = (
        positive_power
        / safe_power_sum[:, None, :]
    )

    log_normalized_power = np.zeros_like(
        normalized_power
    )

    np.log(
        normalized_power,
        out=log_normalized_power,
        where=normalized_power > 0,
    )

    spectral_entropy = (
        -(
            normalized_power
            * log_normalized_power
        ).sum(axis=1)
        / np.log(
            len(positive_frequencies)
        )
    )

    dominant_frequency[
        ~valid_power
    ] = 0.0
    dominant_power_ratio[
        ~valid_power
    ] = 0.0
    spectral_centroid[
        ~valid_power
    ] = 0.0
    spectral_entropy[
        ~valid_power
    ] = 0.0

    statistic_values = {
        "total_power": total_power,
        "dominant_frequency_hz": dominant_frequency,
        "dominant_power_ratio": dominant_power_ratio,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_entropy": spectral_entropy,
    }

    for (
        feature_name,
        lower_bound,
        upper_bound,
        include_upper,
    ) in FREQUENCY_BANDS:
        if include_upper:
            band_mask = (
                (positive_frequencies >= lower_bound)
                & (positive_frequencies <= upper_bound)
            )
        else:
            band_mask = (
                (positive_frequencies >= lower_bound)
                & (positive_frequencies < upper_bound)
            )

        relative_band_power = (
            positive_power[
                :, band_mask, :
            ].sum(axis=1)
            / safe_power_sum
        )

        relative_band_power[
            ~valid_power
        ] = 0.0

        statistic_values[
            feature_name
        ] = relative_band_power

    feature_names = [
        (
            f"{sensor_prefix}_"
            f"{signal_name}_"
            f"{statistic_name}"
        )
        for signal_name in SIGNAL_NAMES
        for statistic_name
        in FREQUENCY_STATISTICS
    ]

    feature_matrix = np.column_stack([
        statistic_values[
            statistic_name
        ][:, signal_index]
        for signal_index, _
        in enumerate(SIGNAL_NAMES)
        for statistic_name
        in FREQUENCY_STATISTICS
    ])

    return feature_matrix, feature_names


def _validate_sequences(
    accel_sequences: np.ndarray,
    gyro_sequences: np.ndarray,
) -> None:
    if accel_sequences.shape != gyro_sequences.shape:
        raise ValueError(
            "Accelerometer and gyroscope arrays must have equal shapes."
        )

    if (
        accel_sequences.ndim != 3
        or accel_sequences.shape[1:] != (100, 3)
    ):
        raise ValueError(
            "Expected paired sequence arrays with shape (n, 100, 3)."
        )

    if not (
        np.isfinite(accel_sequences).all()
        and np.isfinite(gyro_sequences).all()
    ):
        raise ValueError(
            "Sequence arrays must contain only finite values."
        )


def extract_engineered_features(
    accel_sequences: np.ndarray,
    gyro_sequences: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, list[str]]:
    """Return the tested 222-feature matrix and ordered feature names."""
    _validate_sequences(
        accel_sequences,
        gyro_sequences,
    )

    sampling_rate_hz = float(
        sampling_rate_hz
    )

    if not np.isclose(
        sampling_rate_hz,
        20.0,
    ):
        raise ValueError(
            "The frozen feature contract expects 20 Hz sequences."
        )

    accel_signals = _add_magnitude(
        accel_sequences
    )
    gyro_signals = _add_magnitude(
        gyro_sequences
    )

    feature_groups = []
    name_groups = []

    for extractor, arguments in (
        (
            _extract_baseline_features,
            (),
        ),
        (
            _extract_distribution_features,
            (),
        ),
        (
            _extract_temporal_features,
            (1.0 / sampling_rate_hz,),
        ),
        (
            _extract_frequency_features,
            (sampling_rate_hz,),
        ),
    ):
        accel_features, accel_names = extractor(
            accel_signals,
            "accel",
            *arguments,
        )
        gyro_features, gyro_names = extractor(
            gyro_signals,
            "gyro",
            *arguments,
        )

        feature_groups.append(
            np.concatenate(
                [
                    accel_features,
                    gyro_features,
                ],
                axis=1,
            )
        )
        name_groups.extend(
            accel_names + gyro_names
        )

    feature_matrix = np.concatenate(
        feature_groups,
        axis=1,
    )

    if feature_matrix.shape != (
        len(accel_sequences),
        TOTAL_FEATURES,
    ):
        raise RuntimeError(
            "Unexpected engineered-feature matrix shape."
        )

    if (
        len(name_groups) != TOTAL_FEATURES
        or len(set(name_groups))
        != TOTAL_FEATURES
    ):
        raise RuntimeError(
            "Unexpected engineered-feature name contract."
        )

    if not np.isfinite(
        feature_matrix
    ).all():
        raise RuntimeError(
            "Engineered features contain non-finite values."
        )

    return feature_matrix, name_groups
