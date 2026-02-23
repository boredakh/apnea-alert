// src/api/apneaService.js - UPDATED
const API_URL = 'https://apnea-alert.onrender.com';

// ADD THIS SECTION - Feature descriptions for UI
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

// Rename exampleFeatures to EXAMPLE_FEATURES (uppercase for consistency)
export const EXAMPLE_FEATURES = {
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

export const predictApnea = async (features) => {
  try {
    console.log('📡 Sending prediction request...', features);
    
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(features),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const result = await response.json();
    console.log('✅ Prediction result:', result);
    return result;
  } catch (error) {
    console.error('❌ Prediction error:', error);
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_URL}/health`);
    return await response.json();
  } catch (error) {
    console.error('Health check error:', error);
    return { status: 'unreachable', model_loaded: false };
  }
};

export const getModelInfo = async () => {
  try {
    const response = await fetch(`${API_URL}/model/info`);
    return await response.json();
  } catch (error) {
    console.error('Model info error:', error);
    return null;
  }
};

// Keep backward compatibility
export const exampleFeatures = EXAMPLE_FEATURES;