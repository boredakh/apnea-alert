// src/api/fitbitService.js - Complete with HRV support and corrected HR calculations

import * as AuthSession from 'expo-auth-session';
import * as WebBrowser from 'expo-web-browser';
import { makeRedirectUri } from 'expo-auth-session';
import * as Crypto from 'expo-crypto';
import { Platform } from 'react-native';
import Constants from 'expo-constants';

// Get credentials from app.json extra
const FITBIT_CLIENT_ID = Constants.expoConfig?.extra?.fitbitClientId || 'YOUR_FITBIT_CLIENT_ID';
const FITBIT_CLIENT_SECRET = Constants.expoConfig?.extra?.fitbitClientSecret || 'YOUR_FITBIT_CLIENT_SECRET';

// Fitbit API endpoints
const FITBIT_AUTH_ENDPOINT = 'https://www.fitbit.com/oauth2/authorize';
const FITBIT_TOKEN_ENDPOINT = 'https://api.fitbit.com/oauth2/token';
const FITBIT_API_BASE = 'https://api.fitbit.com/1.2';

// Create redirect URI
const REDIRECT_URI = makeRedirectUri({
  scheme: 'apneaalert',
  path: 'fitbit-callback'
});

console.log('🔐 Redirect URI:', REDIRECT_URI);

// Generate PKCE challenge CORRECTLY
async function generatePKCE() {
  // 1. Generate a secure random code verifier (43-128 chars)
  const randomBytes = await Crypto.getRandomBytesAsync(32);
  const codeVerifier = base64URLEncode(randomBytes);
  
  // 2. Create SHA-256 hash of the verifier
  const hashBytes = await Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    codeVerifier,
    { encoding: Crypto.CryptoEncoding.BASE64 }
  );

  // 3. Convert base64 hash to base64url WITHOUT padding
  const codeChallenge = hashBytes
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');

  console.log('✅ PKCE Generated');
  
  return { codeVerifier, codeChallenge };
}

// Helper function to convert to base64url
function base64URLEncode(str) {
  let base64;
  if (typeof str === 'string') {
    base64 = str;
  } else {
    const uint8Array = new Uint8Array(str);
    let binary = '';
    for (let i = 0; i < uint8Array.length; i++) {
      binary += String.fromCharCode(uint8Array[i]);
    }
    base64 = btoa(binary);
  }
  
  return base64
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
}

// Store PKCE verifier globally for this session
let currentCodeVerifier = null;

// Discovery document for Fitbit OAuth
const discovery = {
  authorizationEndpoint: FITBIT_AUTH_ENDPOINT,
  tokenEndpoint: FITBIT_TOKEN_ENDPOINT,
};

export const fitbitConfig = {
  clientId: FITBIT_CLIENT_ID,
  clientSecret: FITBIT_CLIENT_SECRET,
  redirectUri: REDIRECT_URI,
  scopes: ['sleep', 'heartrate', 'profile', 'activity'],
};

// Store tokens
let accessToken = null;
let refreshToken = null;
let tokenExpiry = null;

// Authentication function with PKCE
export const authenticateWithFitbit = async () => {
  try {
    const { codeVerifier, codeChallenge } = await generatePKCE();
    currentCodeVerifier = codeVerifier;
    
    console.log('🔑 Starting auth with PKCE...');
    
    const authRequest = new AuthSession.AuthRequest({
      clientId: fitbitConfig.clientId,
      scopes: fitbitConfig.scopes,
      redirectUri: fitbitConfig.redirectUri,
      responseType: AuthSession.ResponseType.Code,
      extraParams: {
        code_challenge: codeChallenge,
        code_challenge_method: 'S256',
        expires_in: '86400',
      },
    });

    const result = await authRequest.promptAsync(discovery, {
      windowFeatures: { width: 600, height: 800 }
    });
    
    if (result.type === 'success') {
      console.log('✅ Auth successful, exchanging code...');
      const tokenResult = await exchangeCodeForTokens(result.params.code, currentCodeVerifier);
      return { success: true, ...tokenResult };
    }
    
    return { success: false, error: 'Authentication cancelled or failed' };
  } catch (error) {
    console.error('Fitbit auth error:', error);
    return { success: false, error: error.message };
  }
};

// Exchange authorization code for tokens
const exchangeCodeForTokens = async (code, verifier) => {
  try {
    const credentials = btoa(`${fitbitConfig.clientId}:${fitbitConfig.clientSecret}`);
    
    const response = await fetch(FITBIT_TOKEN_ENDPOINT, {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${credentials}`,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        code: code,
        grant_type: 'authorization_code',
        client_id: fitbitConfig.clientId,
        redirect_uri: fitbitConfig.redirectUri,
        code_verifier: verifier,
      }).toString(),
    });

    const tokens = await response.json();
    
    if (response.ok) {
      accessToken = tokens.access_token;
      refreshToken = tokens.refresh_token;
      tokenExpiry = Date.now() + (tokens.expires_in * 1000);
      
      return { 
        success: true, 
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token,
        expiresIn: tokens.expires_in,
        userId: tokens.user_id
      };
    } else {
      throw new Error(tokens.errors?.[0]?.message || 'Token exchange failed');
    }
  } catch (error) {
    console.error('Token exchange error:', error);
    throw error;
  }
};

// Refresh access token
export const refreshAccessToken = async () => {
  if (!refreshToken) {
    throw new Error('No refresh token available');
  }

  try {
    const credentials = btoa(`${fitbitConfig.clientId}:${fitbitConfig.clientSecret}`);
    
    const response = await fetch(FITBIT_TOKEN_ENDPOINT, {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${credentials}`,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: fitbitConfig.clientId,
      }).toString(),
    });

    const tokens = await response.json();
    
    if (response.ok) {
      accessToken = tokens.access_token;
      refreshToken = tokens.refresh_token;
      tokenExpiry = Date.now() + (tokens.expires_in * 1000);
      
      return { 
        success: true, 
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token,
        expiresIn: tokens.expires_in
      };
    } else {
      throw new Error(tokens.errors?.[0]?.message || 'Token refresh failed');
    }
  } catch (error) {
    console.error('Token refresh error:', error);
    throw error;
  }
};

// Ensure valid token before API calls
const ensureValidToken = async () => {
  if (!accessToken) {
    throw new Error('Not authenticated with Fitbit');
  }
  
  if (tokenExpiry && Date.now() > tokenExpiry - 300000) {
    await refreshAccessToken();
  }
  
  return accessToken;
};

// Fetch sleep data from Fitbit
export const fetchFitbitSleepData = async (date = 'today') => {
  try {
    const token = await ensureValidToken();
    
    const response = await fetch(`${FITBIT_API_BASE}/user/-/sleep/date/${date}.json`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.errors?.[0]?.message || 'Failed to fetch sleep data');
    }

    return { success: true, data: data.sleep };
  } catch (error) {
    console.error('Fetch sleep data error:', error);
    return { success: false, error: error.message };
  }
};

// Fetch heart rate data
export const fetchFitbitHeartRateData = async (date = 'today') => {
  try {
    const token = await ensureValidToken();
    
    const response = await fetch(`${FITBIT_API_BASE}/user/-/activities/heart/date/${date}/1d.json`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.errors?.[0]?.message || 'Failed to fetch heart rate data');
    }

    return { success: true, data: data['activities-heart'] };
  } catch (error) {
    console.error('Fetch heart rate error:', error);
    return { success: false, error: error.message };
  }
};

// Fetch HRV data from Fitbit - DEBUG VERSION with raw response
export const fetchFitbitHRVData = async (date = 'today') => {
  try {
    const token = await ensureValidToken();
    
    // Use API version 1 for HRV
    const HRV_API_BASE = 'https://api.fitbit.com/1';
    const url = `${HRV_API_BASE}/user/-/hrv/date/${date}/all.json`;
    
    console.log(`🔍 Fetching HRV from: ${url}`);
    
    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/json'
      },
    });

    // First, get the raw text to see what's actually coming back
    const rawText = await response.text();
    console.log('📄 Raw response (first 200 chars):', rawText.substring(0, 200));
    console.log('📊 Response status:', response.status);
    console.log('📋 Response headers:', Object.fromEntries(response.headers.entries()));

    // Try to parse as JSON
    try {
      const data = JSON.parse(rawText);
      console.log('✅ Successfully parsed JSON');
      
      if (response.ok) {
        console.log('✅ HRV data received!');
        
        if (data.hrv && data.hrv.length > 0) {
          const firstEntry = data.hrv[0];
          console.log('📈 Sample HRV entry:', JSON.stringify(firstEntry, null, 2));
          
          if (firstEntry.minutes && firstEntry.minutes.length > 0) {
            const sampleMinute = firstEntry.minutes[0];
            console.log('⏱️ Sample minute data:', {
              rmssd: sampleMinute.value?.rmssd,
              lf: sampleMinute.value?.lf,
              hf: sampleMinute.value?.hf
            });
          }
        }
        
        return { success: true, data: data, type: 'intraday' };
      } else {
        console.log('❌ API error:', data);
        return { success: false, data: null, error: data };
      }
    } catch (parseError) {
      console.error('❌ JSON parse error. Raw response starts with:', rawText.substring(0, 50));
      return { success: false, data: null, rawResponse: rawText.substring(0, 200) };
    }
    
  } catch (error) {
    console.error('Fetch HRV error:', error);
    return { success: false, data: null, error: error.message };
  }
};

// Main transformation function - ASYNC
export const transformFitbitDataToModelInputs = async (sleepData, heartRateData) => {
  if (!sleepData || sleepData.length === 0) {
    return null;
  }

  // Get today's date for HRV fetch
  const today = new Date().toISOString().split('T')[0];
  const hrvResult = await fetchFitbitHRVData(today);
  const hrvData = hrvResult.success ? hrvResult.data : null;

  // Log HRV data if available
  if (hrvData?.hrv?.length > 0) {
    console.log('📊 HRV data details:', {
      rmssd: hrvData.hrv[0]?.value?.rmssd,
      hasData: !!hrvData.hrv[0]?.value
    });
  }

  const mainSleep = sleepData.reduce((longest, current) => {
    return (current.duration > longest.duration) ? current : longest;
  }, sleepData[0]);

  console.log('🔄 Transforming sleep data for model...');
  console.log('📊 HRV data available:', hrvData ? 'Yes' : 'No');
  
  const features = {
    mean: calculateMeanHeartRate(heartRateData),
    std: calculateHeartRateStd(heartRateData, hrvData),
    min: calculateMinHeartRate(heartRateData),
    max: calculateMaxHeartRate(heartRateData),
    range: calculateHeartRateRange(heartRateData),
    rms: approximateRMS(heartRateData, hrvData),
    skewness: approximateSkewness(heartRateData),
    kurtosis: approximateKurtosis(heartRateData),
    hr_mean: calculateMeanHeartRate(heartRateData),
    hr_std: calculateHeartRateStd(heartRateData, hrvData),
    rr_mean: calculateRRInterval(mainSleep),
    rr_std: calculateRRVariability(mainSleep, hrvData),
    lf_power: calculateLFPower(mainSleep, hrvData),
    hf_power: calculateHFPower(mainSleep, hrvData),
    lf_hf_ratio: calculateLFHFRatio(mainSleep, hrvData),
  };

  console.log('✅ Features calculated - HR range:', features.min, '-', features.max);
  return features;
};

// ============ HELPER CALCULATION FUNCTIONS ============

const calculateMeanHeartRate = (heartRateData) => {
  if (!heartRateData?.[0]?.value?.heartRateZones) {
    return 68.5;
  }
  
  const zones = heartRateData[0].value.heartRateZones;
  const totalMinutes = zones.reduce((sum, zone) => sum + (zone.minutes || 0), 0);
  const weightedHeartRate = zones.reduce((sum, zone) => {
    const avgZoneHR = (zone.min + zone.max) / 2;
    return sum + (avgZoneHR * (zone.minutes || 0));
  }, 0);
  
  return totalMinutes > 0 ? weightedHeartRate / totalMinutes : 68.5;
};

const calculateHeartRateStd = (heartRateData, hrvData) => {
  // Use HRV data if available
  if (hrvData?.hrv?.length > 0) {
    // Check for intraday data first
    if (hrvData.hrv[0]?.minutes?.length > 0) {
      const minuteValues = hrvData.hrv[0].minutes
        .map(m => m.value?.rmssd)
        .filter(v => v != null);
      
      if (minuteValues.length > 0) {
        const avgHRV = minuteValues.reduce((a, b) => a + b, 0) / minuteValues.length;
        console.log('📊 Using intraday HRV for std:', avgHRV, 'ms');
        return avgHRV * 0.158;
      }
    }
    
    // Fallback to daily summary
    const dailyHRV = hrvData.hrv[0];
    if (dailyHRV?.value?.rmssd) {
      const hrv = dailyHRV.value.rmssd;
      console.log('📊 Using daily HRV summary for std:', hrv, 'ms');
      return hrv * 0.158;
    }
  }
  
  // Fallback to heart rate based approximation
  const hr = calculateMeanHeartRate(heartRateData);
  return hr * 0.12;
};

const calculateMinHeartRate = (heartRateData) => {
  if (!heartRateData?.[0]?.value?.heartRateZones) {
    return 40;
  }
  
  // Use resting heart rate if available
  const restingHR = heartRateData[0]?.value?.restingHeartRate;
  if (restingHR) {
    return restingHR;
  }
  
  const restingZone = heartRateData[0].value.heartRateZones.find(z => z.name === 'Out of Range');
  return restingZone?.min || 40;
};

const calculateMaxHeartRate = (heartRateData) => {
  if (!heartRateData?.[0]?.value?.heartRateZones) {
    return 100;
  }
  
  // Use resting heart rate to calculate a reasonable max
  const restingHR = heartRateData[0]?.value?.restingHeartRate;
  if (restingHR) {
    // During sleep, max shouldn't exceed resting by more than 50%
    return Math.min(restingHR * 1.5, 120);
  }
  
  // Fallback to a reasonable sleep-time max
  return 100;
};

const calculateHeartRateRange = (heartRateData) => {
  const max = calculateMaxHeartRate(heartRateData);
  const min = calculateMinHeartRate(heartRateData);
  return max - min;
};

const approximateRMS = (heartRateData, hrvData) => {
  // Use HRV for RMS approximation
  if (hrvData?.hrv?.length > 0) {
    if (hrvData.hrv[0]?.minutes?.length > 0) {
      const minuteValues = hrvData.hrv[0].minutes
        .map(m => m.value?.rmssd)
        .filter(v => v != null);
      
      if (minuteValues.length > 0) {
        const avgHRV = minuteValues.reduce((a, b) => a + b, 0) / minuteValues.length;
        return avgHRV * 0.00615;
      }
    }
    
    const dailyHRV = hrvData.hrv[0];
    if (dailyHRV?.value?.rmssd) {
      const hrv = dailyHRV.value.rmssd;
      return hrv * 0.00615;
    }
  }
  return 0.32;
};

const approximateSkewness = (heartRateData) => {
  return 3.5;
};

const approximateKurtosis = (heartRateData) => {
  return 18.5;
};

const calculateRRInterval = (sleepData) => {
  const hr = sleepData?.minHeartRate || 68.5;
  return 60 / hr;
};

const calculateRRVariability = (sleepData, hrvData) => {
  // Use HRV if available
  if (hrvData?.hrv?.length > 0) {
    if (hrvData.hrv[0]?.minutes?.length > 0) {
      const minuteValues = hrvData.hrv[0].minutes
        .map(m => m.value?.rmssd)
        .filter(v => v != null);
      
      if (minuteValues.length > 0) {
        const avgHRV = minuteValues.reduce((a, b) => a + b, 0) / minuteValues.length;
        return (avgHRV / 1000) * 2.3;
      }
    }
    
    const dailyHRV = hrvData.hrv[0];
    if (dailyHRV?.value?.rmssd) {
      const hrv = dailyHRV.value.rmssd;
      return (hrv / 1000) * 2.3;
    }
  }
  return 0.12;
};

const calculateLFPower = (sleepData, hrvData) => {
  // If we have real LF data from HRV intraday, use it!
  if (hrvData?.hrv?.length > 0 && hrvData.hrv[0]?.minutes?.length > 0) {
    // Average LF across all minutes of sleep
    const lfValues = hrvData.hrv[0].minutes
      .map(m => m.value?.lf)
      .filter(v => v != null);
    
    if (lfValues.length > 0) {
      const avgLF = lfValues.reduce((a, b) => a + b, 0) / lfValues.length;
      console.log('📊 Using real LF data:', avgLF);
      return avgLF / 10000; // Scale to match model expectations
    }
  }
  
  // Fallback to sleep stage approximation
  const deepSleepMinutes = sleepData?.levels?.summary?.deep?.minutes || 60;
  const totalSleepMinutes = (sleepData?.duration || 28800000) / 60000;
  const deepSleepRatio = deepSleepMinutes / totalSleepMinutes;
  return 0.00015 * (1 + deepSleepRatio);
};

const calculateHFPower = (sleepData, hrvData) => {
  // If we have real HF data from HRV intraday, use it!
  if (hrvData?.hrv?.length > 0 && hrvData.hrv[0]?.minutes?.length > 0) {
    const hfValues = hrvData.hrv[0].minutes
      .map(m => m.value?.hf)
      .filter(v => v != null);
    
    if (hfValues.length > 0) {
      const avgHF = hfValues.reduce((a, b) => a + b, 0) / hfValues.length;
      console.log('📊 Using real HF data:', avgHF);
      return avgHF / 10000; // Scale to match model expectations
    }
  }
  
  // Fallback to sleep stage approximation
  const remSleepMinutes = sleepData?.levels?.summary?.rem?.minutes || 90;
  const totalSleepMinutes = (sleepData?.duration || 28800000) / 60000;
  const remSleepRatio = remSleepMinutes / totalSleepMinutes;
  return 0.00008 * (1 + remSleepRatio);
};

const calculateLFHFRatio = (sleepData, hrvData) => {
  const lf = calculateLFPower(sleepData, hrvData);
  const hf = calculateHFPower(sleepData, hrvData);
  return lf / hf;
};