// src/utils/constants.js
export const API_URL = 'https://apnea-alert.onrender.com';

export const FEATURE_DESCRIPTIONS = {
  mean: 'Average ECG signal amplitude',
  std: 'Standard deviation of ECG signal',
  min: 'Minimum ECG value',
  max: 'Maximum ECG value',
  range: 'Difference between max and min',
  rms: 'Root mean square of ECG',
  skewness: 'Signal distribution asymmetry',
  kurtosis: 'Signal distribution tailedness',
  hr_mean: 'Average heart rate (BPM)',
  hr_std: 'Heart rate variability',
  rr_mean: 'Average RR interval (seconds)',
  rr_std: 'RR interval variability',
  lf_power: 'Low frequency power (sympathetic activity)',
  hf_power: 'High frequency power (parasympathetic)',
  lf_hf_ratio: 'Sympathovagal balance'
};

export const FEATURE_RANGES = {
  mean: { min: -0.01, max: 0.01, step: 0.0001 },
  std: { min: 0, max: 1, step: 0.01 },
  min: { min: -2, max: 0, step: 0.1 },
  max: { min: 0, max: 3, step: 0.1 },
  range: { min: 0, max: 5, step: 0.1 },
  rms: { min: 0, max: 1, step: 0.01 },
  skewness: { min: -5, max: 10, step: 0.1 },
  kurtosis: { min: 0, max: 30, step: 0.5 },
  hr_mean: { min: 40, max: 120, step: 1 },
  hr_std: { min: 0, max: 20, step: 0.5 },
  rr_mean: { min: 0.5, max: 1.5, step: 0.01 },
  rr_std: { min: 0, max: 0.5, step: 0.01 },
  lf_power: { min: 0, max: 0.001, step: 0.00001 },
  hf_power: { min: 0, max: 0.001, step: 0.00001 },
  lf_hf_ratio: { min: 0, max: 5, step: 0.1 }
};

export const DEFAULT_FEATURES = {
  mean: -0.0003,
  std: 0.35,
  min: -1.2,
  max: 2.1,
  range: 3.3,
  rms: 0.32,
  skewness: 3.5,
  kurtosis: 18.5,
  hr_mean: 68.5,
  hr_std: 8.2,
  rr_mean: 0.88,
  rr_std: 0.12,
  lf_power: 0.00015,
  hf_power: 0.00008,
  lf_hf_ratio: 1.87
};