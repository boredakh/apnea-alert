// src/api/fitbitService.js
import * as Crypto from 'expo-crypto';
import { Platform, Linking, AppState } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../services/supabase';

const FITBIT_CLIENT_ID = '23TZXG';
const FITBIT_CLIENT_SECRET = '5ba1f05cd65c3d6d82e3fabfc7f2834f';

const FITBIT_AUTH_ENDPOINT = 'https://www.fitbit.com/oauth2/authorize';
const FITBIT_TOKEN_ENDPOINT = 'https://api.fitbit.com/oauth2/token';
const FITBIT_API_BASE = 'https://api.fitbit.com/1.2';
const HRV_API_BASE = 'https://api.fitbit.com/1';

// Get redirect URI based on platform
const getRedirectUri = () => {
  if (Platform.OS === 'web') {
    return 'https://endymion-seven.vercel.app/fitbit-callback.html';
  }
  return 'https://endymion-seven.vercel.app/fitbit-callback.html';
};

const REDIRECT_URI = getRedirectUri();

console.log('🔐 Redirect URI:', REDIRECT_URI);

// ============ PKCE HELPER FUNCTIONS ============

async function generatePKCE() {
  // Generate 32 random bytes (256 bits) - this ensures proper length
  const randomBytes = await Crypto.getRandomBytesAsync(32);
  
  // Convert to base64 URL-safe string for code_verifier
  let binaryString = '';
  for (let i = 0; i < randomBytes.length; i++) {
    binaryString += String.fromCharCode(randomBytes[i]);
  }
  let base64 = btoa(binaryString);
  const codeVerifier = base64
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
  
  // Create SHA256 hash of the verifier
  const hashBytes = await Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    codeVerifier,
    { encoding: Crypto.CryptoEncoding.BASE64 }
  );

  // Make URL-safe
  const codeChallenge = hashBytes
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');

  console.log('Code Verifier length (should be 43-128):', codeVerifier.length);
  console.log('Code Challenge length (should be 43-128):', codeChallenge.length);

  return { codeVerifier, codeChallenge };
}

// ============ TOKEN STORAGE ============

let currentCodeVerifier = null;
let accessToken = null;
let refreshToken = null;
let tokenExpiry = null;

export const setAccessTokenFromProfile = (token) => {
  if (token && token.access_token) {
    accessToken = token.access_token;
    refreshToken = token.refresh_token;
    tokenExpiry = Date.now() + ((token.expires_in || 28800) * 1000);
    console.log('✅ Token loaded from profile');
    return true;
  }
  console.log('❌ No token to load');
  return false;
};

// ============ FITBIT PROXY (CORS Bypass) ============

const callFitbitProxy = async (endpoint, date = null) => {
  console.log(`📡 Calling fitbit-proxy for: ${endpoint}, date: ${date}`);
  
  const { data: { user } } = await supabase.auth.getUser();
  
  if (!user) {
    throw new Error('Not authenticated');
  }

  const response = await supabase.functions.invoke('fitbit-proxy', {
    body: {
      userId: user.id,
      endpoint: endpoint,
      date: date
    }
  });

  if (response.error) {
    console.error('Proxy error:', response.error);
    throw new Error(response.error.message);
  }

  return response.data;
};

// ============ FITBIT API CALLS (via Proxy) ============

export const getFitbitProfile = async () => {
  try {
    const result = await callFitbitProxy('profile');
    if (result.success) {
      return { success: true, data: result.data.user };
    }
    return { success: false, error: result.error };
  } catch (error) {
    console.error('Fetch profile error:', error);
    return { success: false, error: error.message };
  }
};
export const predictApnea = async (features) => {
  try {
    console.log('📡 Sending prediction request to ML model...');
    
    const response = await fetch('https://apnea-alert.onrender.com/predict', {
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
export const fetchFitbitSleepData = async (date = 'today') => {
  try {
    const result = await callFitbitProxy('sleep', date);
    if (result.success) {
      return { success: true, data: result.data.sleep };
    }
    return { success: false, error: result.error };
  } catch (error) {
    console.error('Fetch sleep data error:', error);
    return { success: false, error: error.message };
  }
};

export const fetchFitbitHeartRateData = async (date = 'today') => {
  try {
    const result = await callFitbitProxy('heartrate', date);
    if (result.success) {
      return { success: true, data: result.data['activities-heart'] };
    }
    return { success: false, error: result.error };
  } catch (error) {
    console.error('Fetch heart rate error:', error);
    return { success: false, error: error.message };
  }
};

export const fetchFitbitHRVData = async (date = 'today') => {
  try {
    const result = await callFitbitProxy('hrv', date);
    if (result.success) {
      return { success: true, data: result.data };
    }
    return { success: false, error: result.error };
  } catch (error) {
    console.error('Fetch HRV error:', error);
    return { success: false, error: error.message };
  }
};

// ============ OAuth AUTHENTICATION ============

export const fitbitConfig = {
  clientId: FITBIT_CLIENT_ID,
  clientSecret: FITBIT_CLIENT_SECRET,
  redirectUri: REDIRECT_URI,
  scopes: ['sleep', 'heartrate', 'profile', 'activity'],
};

export const exchangeCodeForTokens = async (code, verifier, isIOS = false) => {
  try {
    console.log('🔄 Exchanging code for tokens...');
    
    const credentials = btoa(`${fitbitConfig.clientId}:${fitbitConfig.clientSecret}`);
    
    // Use different redirect_uri for iOS vs web
    const redirectUri = isIOS 
      ? 'https://bqbjnqxxauohjdenyxma.supabase.co/functions/v1/fitbit-ios-callback'
      : fitbitConfig.redirectUri;
    
    console.log('📱 Using redirect URI for token exchange:', redirectUri);
    
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
        redirect_uri: redirectUri,
        code_verifier: verifier,
      }).toString(),
    });

    const tokens = await response.json();
    
    if (response.ok) {
      accessToken = tokens.access_token;
      refreshToken = tokens.refresh_token;
      tokenExpiry = Date.now() + (tokens.expires_in * 1000);
      
      console.log('✅ Token exchange successful');
      
      return { 
        success: true, 
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token,
        expiresIn: tokens.expires_in,
        userId: tokens.user_id
      };
    } else {
      console.error('❌ Token exchange failed:', tokens);
      throw new Error(tokens.errors?.[0]?.message || 'Token exchange failed');
    }
  } catch (error) {
    console.error('Token exchange error:', error);
    throw error;
  }
};

export const authenticateWithFitbit = async () => {
  try {
    const { codeVerifier, codeChallenge } = await generatePKCE();
    currentCodeVerifier = codeVerifier;
    
    console.log('🔑 Starting auth with PKCE...');
    console.log('📱 Using redirect URI:', fitbitConfig.redirectUri);
    
    // Web platform - use popup
    if (Platform.OS === 'web') {
      const authUrl = `${FITBIT_AUTH_ENDPOINT}?` + new URLSearchParams({
        client_id: fitbitConfig.clientId,
        response_type: 'code',
        redirect_uri: fitbitConfig.redirectUri,
        scope: fitbitConfig.scopes.join(' '),
        code_challenge: codeChallenge,
        code_challenge_method: 'S256',
      }).toString();
      
      console.log('🔗 Auth URL:', authUrl);
      
      localStorage.setItem('fitbit_code_verifier', codeVerifier);
      
      const width = 600;
      const height = 700;
      const left = window.screen.width / 2 - width / 2;
      const top = window.screen.height / 2 - height / 2;
      
      const popup = window.open(
        authUrl,
        'fitbit-auth',
        `width=${width},height=${height},left=${left},top=${top}`
      );
      
      if (!popup) {
        return { success: false, error: 'Popup blocked' };
      }
      
      return new Promise((resolve) => {
        const interval = setInterval(() => {
          const code = localStorage.getItem('fitbit_auth_code');
          
          if (code) {
            clearInterval(interval);
            localStorage.removeItem('fitbit_auth_code');
            const verifier = localStorage.getItem('fitbit_code_verifier');
            localStorage.removeItem('fitbit_code_verifier');
            exchangeCodeForTokens(code, verifier, false).then(resolve);
          }
          
          if (popup.closed) {
            clearInterval(interval);
            resolve({ success: false, error: 'Cancelled' });
          }
        }, 500);
      });
    }
    // iOS - use Edge Function callback
    else if (Platform.OS === 'ios') {
      console.log('📱 Opening Safari for iOS...');
      
      // Generate a unique session ID
      const randomBytes = await Crypto.getRandomBytesAsync(16);
      const sessionId = Array.from(randomBytes)
        .map(b => b.toString(16).padStart(2, '0'))
        .join('');
      
      console.log('📱 Generated Session ID:', sessionId);
      
      await AsyncStorage.setItem('fitbit_session_id', sessionId);
      await AsyncStorage.setItem('fitbit_code_verifier', codeVerifier);
      
      // Use the Edge Function as callback
      const iosRedirectUri = 'https://bqbjnqxxauohjdenyxma.supabase.co/functions/v1/fitbit-ios-callback';
      
      const iosAuthUrl = `${FITBIT_AUTH_ENDPOINT}?` + new URLSearchParams({
        client_id: FITBIT_CLIENT_ID,
        response_type: 'code',
        redirect_uri: iosRedirectUri,
        scope: fitbitConfig.scopes.join(' '),
        code_challenge: codeChallenge,
        code_challenge_method: 'S256',
        state: sessionId,
      }).toString();
      
      console.log('📱 iOS Auth URL:', iosAuthUrl);
      
      // Open Safari
      await Linking.openURL(iosAuthUrl);
      
      // Return promise that resolves when app detects the stored code
      return new Promise((resolve) => {
        let checkCount = 0;
        const subscription = AppState.addEventListener('change', async (nextAppState) => {
          if (nextAppState === 'active') {
            console.log('📱 App became active, checking for code...');
            checkCount++;
            
            const storedSessionId = await AsyncStorage.getItem('fitbit_session_id');
            console.log('📱 Stored Session ID:', storedSessionId);
            
            if (storedSessionId) {
              const { data, error } = await supabase
                .from('pending_fitbit_codes')
                .select('code')
                .eq('session_id', storedSessionId)
                .is('claimed_at', null)
                .maybeSingle();
              
              console.log('📱 Database query result:', { hasData: !!data, error });
              
              if (data && data.code && !error) {
                await supabase
                  .from('pending_fitbit_codes')
                  .update({ claimed_at: new Date().toISOString() })
                  .eq('session_id', storedSessionId);
                
                subscription.remove();
                const verifier = await AsyncStorage.getItem('fitbit_code_verifier');
                await AsyncStorage.multiRemove(['fitbit_session_id', 'fitbit_code_verifier']);
                
                console.log('📩 Got code from database:', data.code);
                // Pass true for iOS
                const tokenResult = await exchangeCodeForTokens(data.code, verifier, true);
                console.log('📩 Token exchange result:', tokenResult.success);
                resolve({ success: true, ...tokenResult });
                return;
              }
            }
            
            if (checkCount > 60) {
              subscription.remove();
              resolve({ success: false, error: 'Timeout waiting for code' });
            }
          }
        });
        
        setTimeout(() => {
          subscription.remove();
          resolve({ success: false, error: 'Timeout' });
        }, 120000);
      });
    }
    // Android - use WebBrowser
    else {
      const authUrl = `${FITBIT_AUTH_ENDPOINT}?` + new URLSearchParams({
        client_id: fitbitConfig.clientId,
        response_type: 'code',
        redirect_uri: fitbitConfig.redirectUri,
        scope: fitbitConfig.scopes.join(' '),
        code_challenge: codeChallenge,
        code_challenge_method: 'S256',
      }).toString();
      
      console.log('🔗 Auth URL:', authUrl);
      
      const WebBrowser = await import('expo-web-browser');
      
      const result = await WebBrowser.openAuthSessionAsync(authUrl, fitbitConfig.redirectUri);
      
      console.log('📱 WebBrowser result:', result.type);
      
      if (result.type === 'success') {
        const url = new URL(result.url);
        const code = url.searchParams.get('code');
        
        if (code) {
          console.log('✅ Got authorization code');
          const tokenResult = await exchangeCodeForTokens(code, currentCodeVerifier, false);
          return { success: true, ...tokenResult };
        }
      }
      
      return { success: false, error: 'Authentication failed' };
    }
  } catch (error) {
    console.error('Fitbit auth error:', error);
    return { success: false, error: error.message };
  }
};

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

export const revokeToken = async (token) => {
  try {
    const response = await fetch('https://api.fitbit.com/oauth2/revoke', {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${btoa(`${fitbitConfig.clientId}:${fitbitConfig.clientSecret}`)}`,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        token: token,
      }).toString(),
    });

    return { success: response.ok };
  } catch (error) {
    console.error('Revoke token error:', error);
    return { success: false, error: error.message };
  }
};

// ============ ML MODEL TRANSFORMATION FUNCTIONS ============

export const transformFitbitDataToModelInputs = async (sleepData, heartRateData) => {
  if (!sleepData || sleepData.length === 0) {
    return null;
  }

  const today = new Date().toISOString().split('T')[0];
  const hrvResult = await fetchFitbitHRVData(today);
  const hrvData = hrvResult.success ? hrvResult.data : null;

  const mainSleep = sleepData.reduce((longest, current) => {
    return (current.duration > longest.duration) ? current : longest;
  }, sleepData[0]);

  console.log('🔄 Transforming sleep data for model...');
  
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
  if (hrvData?.hrv?.length > 0) {
    if (hrvData.hrv[0]?.minutes?.length > 0) {
      const minuteValues = hrvData.hrv[0].minutes
        .map(m => m.value?.rmssd)
        .filter(v => v != null);
      
      if (minuteValues.length > 0) {
        const avgHRV = minuteValues.reduce((a, b) => a + b, 0) / minuteValues.length;
        return avgHRV * 0.158;
      }
    }
    
    const dailyHRV = hrvData.hrv[0];
    if (dailyHRV?.value?.rmssd) {
      const hrv = dailyHRV.value.rmssd;
      return hrv * 0.158;
    }
  }
  
  const hr = calculateMeanHeartRate(heartRateData);
  return hr * 0.12;
};

const calculateMinHeartRate = (heartRateData) => {
  if (!heartRateData?.[0]?.value?.heartRateZones) {
    return 40;
  }
  
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
  
  const restingHR = heartRateData[0]?.value?.restingHeartRate;
  if (restingHR) {
    return Math.min(restingHR * 1.5, 120);
  }
  
  return 100;
};

const calculateHeartRateRange = (heartRateData) => {
  const max = calculateMaxHeartRate(heartRateData);
  const min = calculateMinHeartRate(heartRateData);
  return max - min;
};

const approximateRMS = (heartRateData, hrvData) => {
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
  if (hrvData?.hrv?.length > 0 && hrvData.hrv[0]?.minutes?.length > 0) {
    const lfValues = hrvData.hrv[0].minutes
      .map(m => m.value?.lf)
      .filter(v => v != null);
    
    if (lfValues.length > 0) {
      const avgLF = lfValues.reduce((a, b) => a + b, 0) / lfValues.length;
      return avgLF / 10000;
    }
  }
  
  const deepSleepMinutes = sleepData?.levels?.summary?.deep?.minutes || 60;
  const totalSleepMinutes = (sleepData?.duration || 28800000) / 60000;
  const deepSleepRatio = deepSleepMinutes / totalSleepMinutes;
  return 0.00015 * (1 + deepSleepRatio);
};

const calculateHFPower = (sleepData, hrvData) => {
  if (hrvData?.hrv?.length > 0 && hrvData.hrv[0]?.minutes?.length > 0) {
    const hfValues = hrvData.hrv[0].minutes
      .map(m => m.value?.hf)
      .filter(v => v != null);
    
    if (hfValues.length > 0) {
      const avgHF = hfValues.reduce((a, b) => a + b, 0) / hfValues.length;
      return avgHF / 10000;
    }
  }
  
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