// src/context/FitbitContext.js
import React, { createContext, useState, useContext, useEffect } from 'react';
import { useAuth } from './AuthContext';
import { supabase } from '../services/supabase';
import * as fitbitService from '../api/fitbitService';

const FitbitContext = createContext({});

export const useFitbit = () => useContext(FitbitContext);

const parseJsonField = (value, fallback = null) => {
  if (value == null) return fallback;

  if (typeof value === 'string') {
    try {
      return JSON.parse(value);
    } catch (error) {
      console.warn('⚠️ Failed to parse JSON field:', error);
      return fallback;
    }
  }

  return value;
};

const getMainSleep = (sleepData) => {
  const parsedSleepData = parseJsonField(sleepData, []);

  if (!Array.isArray(parsedSleepData) || parsedSleepData.length === 0) {
    return null;
  }

  return (
    parsedSleepData.find((sleep) => sleep.isMainSleep) ||
    parsedSleepData.find((sleep) => sleep.type === 'stages') ||
    parsedSleepData[0]
  );
};

const formatDateLocal = (date) => {
  if (typeof date === 'string') return date;

  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');

  return `${year}-${month}-${day}`;
};

const cleanPayload = (payload) => {
  return Object.fromEntries(
    Object.entries(payload).filter(([, value]) => value !== undefined)
  );
};

const hasRealFitbitData = ({ sleepData, heartRateData, hrvData, activityData }) => {
  const hasSleep = Array.isArray(sleepData) && sleepData.length > 0;

  const hasHeartRate =
    Array.isArray(heartRateData) &&
    heartRateData.some((entry) => entry?.value?.restingHeartRate || entry?.value?.heartRateZones);

  const hasHRV =
    hrvData?.hrv &&
    Array.isArray(hrvData.hrv) &&
    hrvData.hrv.some((entry) => Array.isArray(entry?.minutes) && entry.minutes.length > 0);

  const hasActivity = activityData != null;

  return hasSleep || hasHeartRate || hasHRV || hasActivity;
};

const getResultStatus = (result) => {
  return result?.status || result?.statusCode || result?.response?.status || null;
};

const isAuthOrScopeError = (...results) => {
  return results.some((result) => {
    const status = getResultStatus(result);
    const errorText = String(result?.error || result?.message || '').toLowerCase();

    return (
      status === 401 ||
      status === 403 ||
      errorText.includes('expired') ||
      errorText.includes('invalid token') ||
      errorText.includes('unauthorized') ||
      errorText.includes('forbidden') ||
      errorText.includes('scope') ||
      errorText.includes('insufficient')
    );
  });
};

export const FitbitProvider = ({ children }) => {
  const { user, profile } = useAuth();

  const [isConnected, setIsConnected] = useState(false);
  const [fitbitProfile, setFitbitProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [lastSync, setLastSync] = useState(null);

  useEffect(() => {
    console.log('🔍 FitbitProvider: Checking profile for token...');

    if (profile?.fitbit_token) {
      console.log('✅ Fitbit token found in profile');

      setIsConnected(true);
      fitbitService.setAccessTokenFromProfile(profile.fitbit_token);
      fetchFitbitUserProfile();
    } else {
      console.log('❌ No Fitbit token in profile');

      setIsConnected(false);
      setFitbitProfile(null);
    }
  }, [profile]);

  const fetchFitbitUserProfile = async () => {
    try {
      console.log('📡 Fetching Fitbit user profile...');

      const result = await fitbitService.getFitbitProfile();

      if (result.success) {
        console.log('✅ Fitbit profile fetched:', result.data);
        setFitbitProfile(result.data);
      } else {
        console.log('❌ Failed to fetch Fitbit profile:', result.error);

        if (isAuthOrScopeError(result)) {
          console.log('⚠️ Fitbit token may be expired, revoked, or missing scopes.');
        }
      }
    } catch (error) {
      console.error('❌ Error fetching Fitbit profile:', error);
    }
  };

  const saveFitbitDataToSupabase = async ({
    date,
    sleepData,
    heartRateData,
    hrvData,
    activityData,
  }) => {
    console.log(`💾 Saving data for ${date}...`);

    if (!user) {
      console.log('❌ No user logged in');
      return { success: false, error: 'No user logged in' };
    }

    const hasAnyNewData = hasRealFitbitData({
      sleepData,
      heartRateData,
      hrvData,
      activityData,
    });

    if (!hasAnyNewData) {
      console.log(`⚠️ No real Fitbit data returned for ${date}. Skipping database save.`);

      return {
        success: false,
        skipped: true,
        error: 'No Fitbit data returned for this date',
      };
    }

    try {
      const { data: existing, error: existingError } = await supabase
        .from('fitbit_data')
        .select('*')
        .eq('user_id', user.id)
        .eq('date', date)
        .maybeSingle();

      if (existingError) throw existingError;

      let result;

      if (existing) {
        console.log(`📝 Updating existing record for ${date}`);

        const updatePayload = cleanPayload({
          sleep_data: sleepData,
          heart_rate_data: heartRateData,
          hrv_data: hrvData,
          activity_data: activityData,
          updated_at: new Date().toISOString(),
        });

        result = await supabase
          .from('fitbit_data')
          .update(updatePayload)
          .eq('id', existing.id);
      } else {
        console.log(`📝 Inserting new record for ${date}`);

        const insertPayload = cleanPayload({
          user_id: user.id,
          date,
          sleep_data: sleepData,
          heart_rate_data: heartRateData,
          hrv_data: hrvData,
          activity_data: activityData,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });

        result = await supabase
          .from('fitbit_data')
          .insert([insertPayload]);
      }

      if (result.error) {
        console.error('❌ Supabase error:', result.error);
        throw result.error;
      }

      console.log(`✅ Data saved successfully for ${date}`);
      return { success: true };
    } catch (error) {
      console.error('❌ Save error:', error);

      return {
        success: false,
        error: error.message || 'Failed to save Fitbit data',
      };
    }
  };

  const syncDate = async (date) => {
    if (!isConnected) {
      console.log('❌ Not connected to Fitbit');
      return { success: false, error: 'Not connected to Fitbit' };
    }

    try {
      const dateStr = formatDateLocal(date);

      console.log(`🔄 Syncing data for ${dateStr}...`);

      const [sleepResult, heartRateResult, hrvResult] = await Promise.all([
        fitbitService.fetchFitbitSleepData(dateStr),
        fitbitService.fetchFitbitHeartRateData(dateStr),
        fitbitService.fetchFitbitHRVData(dateStr),
      ]);

      console.log(`Sleep data: ${sleepResult.success ? '✅' : '❌'}`, {
        count: Array.isArray(sleepResult.data) ? sleepResult.data.length : 0,
        status: getResultStatus(sleepResult),
        error: sleepResult.error,
      });

      console.log(`Heart rate data: ${heartRateResult.success ? '✅' : '❌'}`, {
        count: Array.isArray(heartRateResult.data) ? heartRateResult.data.length : 0,
        status: getResultStatus(heartRateResult),
        error: heartRateResult.error,
      });

      console.log(`HRV data: ${hrvResult.success ? '✅' : '❌'}`, {
        count: hrvResult.data?.hrv?.length || 0,
        status: getResultStatus(hrvResult),
        error: hrvResult.error,
      });

      if (isAuthOrScopeError(sleepResult, heartRateResult, hrvResult)) {
        console.log('❌ Fitbit auth/scope error detected. User should reconnect Fitbit.');

        return {
          success: false,
          needsReconnect: true,
          error:
            'Fitbit authorization failed. Your token may be expired, revoked, or missing required scopes. Please reconnect Fitbit.',
          details: {
            sleep: sleepResult.error,
            heartRate: heartRateResult.error,
            hrv: hrvResult.error,
          },
        };
      }

      const sleepData =
        sleepResult.success &&
        Array.isArray(sleepResult.data) &&
        sleepResult.data.length > 0
          ? sleepResult.data
          : undefined;

      const heartRateData =
        heartRateResult.success &&
        Array.isArray(heartRateResult.data) &&
        heartRateResult.data.length > 0
          ? heartRateResult.data
          : undefined;

      const hrvData =
        hrvResult.success &&
        hrvResult.data?.hrv &&
        Array.isArray(hrvResult.data.hrv) &&
        hrvResult.data.hrv.length > 0
          ? hrvResult.data
          : undefined;

      const saveResult = await saveFitbitDataToSupabase({
        date: dateStr,
        sleepData,
        heartRateData,
        hrvData,
        activityData: undefined,
      });

      if (!saveResult.success && !saveResult.skipped) {
        throw new Error(saveResult.error);
      }

      return {
        success: true,
        skipped: saveResult.skipped || false,
        data: {
          sleep: sleepResult.data,
          heartRate: heartRateResult.data,
          hrv: hrvResult.data,
        },
        endpointResults: {
          sleep: sleepResult,
          heartRate: heartRateResult,
          hrv: hrvResult,
        },
      };
    } catch (error) {
      console.error(`❌ Error syncing ${date}:`, error);

      return {
        success: false,
        error: error.message || 'Failed to sync Fitbit data',
      };
    }
  };

  const syncHistoricalData = async (days = 7) => {
    console.log('🔵 syncHistoricalData CALLED with days:', days);

    if (!isConnected) {
      console.log('❌ Not connected to Fitbit');
      return { success: false, error: 'Not connected to Fitbit' };
    }

    setSyncing(true);
    console.log(`🔄 Starting sync for last ${days} days...`);

    try {
      const dates = [];
      const today = new Date();

      for (let i = 0; i < days; i++) {
        const date = new Date(today);
        date.setDate(today.getDate() - i);
        dates.push(formatDateLocal(date));
      }

      console.log(`📅 Dates to sync: ${dates.join(', ')}`);

      const results = [];

      for (const date of dates) {
        console.log(`--- Syncing ${date} ---`);

        const result = await syncDate(date);
        results.push({ date, ...result });

        if (result.needsReconnect) {
          console.log('🛑 Stopping sync because Fitbit needs reconnect.');
          break;
        }

        await new Promise((resolve) => setTimeout(resolve, 1000));
      }

      const successful = results.filter((result) => result.success && !result.skipped).length;
      const skipped = results.filter((result) => result.skipped).length;
      const failed = results.filter((result) => !result.success && !result.skipped).length;

      setLastSync(new Date().toISOString());

      console.log('✅ Sync completed:', {
        successful,
        skipped,
        failed,
        results,
      });

      return {
        success: failed === 0 || successful > 0,
        summary: {
          successful,
          skipped,
          failed,
        },
        results,
      };
    } catch (error) {
      console.error('❌ Error syncing historical data:', error);

      return {
        success: false,
        error: error.message || 'Failed to sync historical Fitbit data',
      };
    } finally {
      setSyncing(false);
    }
  };

  const savePredictionToDatabase = async ({
    features,
    prediction,
    source,
    fitbitDate,
    metadata,
  }) => {
    if (!user) {
      return { success: false, error: 'No user logged in' };
    }

    try {
      console.log('💾 Saving prediction to database...');

      const { data, error } = await supabase
        .from('predictions')
        .insert({
          user_id: user.id,
          features,
          prediction_label: prediction.prediction_label,
          risk_level: prediction.risk_level,
          apnea_probability: prediction.apnea_probability,
          confidence: prediction.confidence,
          source,
          fitbit_date: fitbitDate,
          metadata,
        })
        .select();

      if (error) throw error;

      console.log('✅ Prediction saved to database:', data);

      return { success: true, data };
    } catch (error) {
      console.error('❌ Error saving prediction:', error);

      return {
        success: false,
        error: error.message || 'Failed to save prediction',
      };
    }
  };

  const aggregateSleepDataForModel = async (days = 7) => {
    if (!user) {
      return { success: false, error: 'No user logged in' };
    }

    try {
      console.log(`📊 Aggregating sleep data from last ${days} days...`);

      const { data, error } = await supabase
        .from('fitbit_data')
        .select('*')
        .eq('user_id', user.id)
        .not('sleep_data', 'is', null)
        .order('date', { ascending: false })
        .limit(days);

      if (error) throw error;

      if (!data || data.length === 0) {
        return {
          success: false,
          error: 'No sleep data found. Please sync your Fitbit data first.',
        };
      }

      console.log(`📊 Found ${data.length} sleep rows`);

      let validNights = 0;

      let totalMinutesAsleep = 0;
      let totalDeepMinutes = 0;
      let totalLightMinutes = 0;
      let totalRemMinutes = 0;
      let totalWakeMinutes = 0;
      let totalSleepEfficiency = 0;

      let totalRestingHeartRate = 0;
      let validHeartRateDays = 0;

      const hrvValues = [];

      for (const record of data) {
        const sleep = getMainSleep(record.sleep_data);

        const heartRateData = parseJsonField(record.heart_rate_data, []);
        const heartRateEntry = Array.isArray(heartRateData)
          ? heartRateData.find((entry) => entry?.value?.restingHeartRate)
          : null;

        const heartRate = heartRateEntry?.value || null;

        const hrvData = parseJsonField(record.hrv_data, { hrv: [] });
        const hrvMinutes = hrvData?.hrv?.[0]?.minutes || [];

        if (sleep && sleep.minutesAsleep > 0) {
          const summary = sleep.levels?.summary || {};

          validNights++;

          totalMinutesAsleep += sleep.minutesAsleep || 0;
          totalDeepMinutes += summary.deep?.minutes || 0;
          totalLightMinutes += summary.light?.minutes || summary.asleep?.minutes || 0;
          totalRemMinutes += summary.rem?.minutes || 0;
          totalWakeMinutes += sleep.minutesAwake || summary.wake?.minutes || summary.awake?.minutes || 0;
          totalSleepEfficiency += sleep.efficiency || 0;
        }

        if (heartRate?.restingHeartRate) {
          totalRestingHeartRate += heartRate.restingHeartRate;
          validHeartRateDays++;
        }

        const dailyHrvValues = Array.isArray(hrvMinutes)
          ? hrvMinutes
              .map((minute) => minute?.value?.rmssd)
              .filter((value) => typeof value === 'number' && !Number.isNaN(value))
          : [];

        if (dailyHrvValues.length > 0) {
          const avgDailyHrv =
            dailyHrvValues.reduce((sum, value) => sum + value, 0) / dailyHrvValues.length;

          hrvValues.push(avgDailyHrv);
        }
      }

      if (validNights === 0) {
        return {
          success: false,
          error: 'No valid sleep records found',
        };
      }

      const avgMinutesAsleep = totalMinutesAsleep / validNights;
      const avgDeepMinutes = totalDeepMinutes / validNights;
      const avgLightMinutes = totalLightMinutes / validNights;
      const avgRemMinutes = totalRemMinutes / validNights;
      const avgWakeMinutes = totalWakeMinutes / validNights;
      const avgSleepEfficiency = totalSleepEfficiency / validNights;

      const avgRestingHeartRate =
        validHeartRateDays > 0 ? totalRestingHeartRate / validHeartRateDays : 65;

      const avgHRV =
        hrvValues.length > 0
          ? hrvValues.reduce((sum, value) => sum + value, 0) / hrvValues.length
          : 35;

      const safeSleepStageTotal =
        avgDeepMinutes + avgLightMinutes + avgRemMinutes || avgMinutesAsleep || 1;

      const features = {
        mean: avgMinutesAsleep / 60,
        std: safeSleepStageTotal / 60 / 3,
        min: Math.min(avgDeepMinutes, avgLightMinutes, avgRemMinutes) / 60,
        max: Math.max(avgDeepMinutes, avgLightMinutes, avgRemMinutes) / 60,
        range:
          (Math.max(avgDeepMinutes, avgLightMinutes, avgRemMinutes) -
            Math.min(avgDeepMinutes, avgLightMinutes, avgRemMinutes)) /
          60,
        rms:
          Math.sqrt(
            (Math.pow(avgDeepMinutes, 2) +
              Math.pow(avgLightMinutes, 2) +
              Math.pow(avgRemMinutes, 2)) /
              3
          ) / 60,

        skewness:
          safeSleepStageTotal > 0
            ? (avgDeepMinutes - avgLightMinutes) / safeSleepStageTotal
            : 0,

        kurtosis:
          safeSleepStageTotal > 0
            ? (avgRemMinutes - avgDeepMinutes) / safeSleepStageTotal
            : 0,

        hr_mean: avgRestingHeartRate,
        hr_std: avgRestingHeartRate * 0.12,

        rr_mean: 60 / avgRestingHeartRate,
        rr_std: (60 / avgRestingHeartRate) * 0.12,

        lf_power: avgHRV * 0.004,
        hf_power: avgHRV * 0.002,
        lf_hf_ratio: 2.0,
      };

      const sleepQuality = avgMinutesAsleep > 0 ? avgDeepMinutes / avgMinutesAsleep : 0;
      features.lf_hf_ratio = sleepQuality > 0.2 ? 1.5 : 2.5;

      console.log('📊 Aggregated features:', {
        nights: validNights,
        avgSleepHours: (avgMinutesAsleep / 60).toFixed(1),
        avgDeepPct:
          avgMinutesAsleep > 0
            ? ((avgDeepMinutes / avgMinutesAsleep) * 100).toFixed(1)
            : '0.0',
        avgHRV: avgHRV.toFixed(1),
        avgRestingHR: avgRestingHeartRate.toFixed(1),
        avgWakeMinutes: avgWakeMinutes.toFixed(1),
        avgSleepEfficiency: avgSleepEfficiency.toFixed(1),
      });

      return {
        success: true,
        features,
        metadata: {
          nights: validNights,
          dateRange: {
            from: data[data.length - 1]?.date,
            to: data[0]?.date,
          },
          avgSleepHours: avgMinutesAsleep / 60,
          avgDeepPercentage:
            avgMinutesAsleep > 0 ? (avgDeepMinutes / avgMinutesAsleep) * 100 : 0,
          avgSleepEfficiency,
          avgRestingHeartRate,
          avgHRV,
        },
      };
    } catch (error) {
      console.error('❌ Error aggregating sleep data:', error);

      return {
        success: false,
        error: error.message || 'Failed to aggregate sleep data',
      };
    }
  };

  const getAggregatedPrediction = async (days = 7) => {
    try {
      console.log(`🔬 Running prediction using last ${days} days of data...`);

      const aggregationResult = await aggregateSleepDataForModel(days);

      if (!aggregationResult.success) {
        return aggregationResult;
      }

      const prediction = await fitbitService.predictApnea(aggregationResult.features);

      await savePredictionToDatabase({
        features: aggregationResult.features,
        prediction,
        source: 'fitbit_aggregated',
        fitbitDate: aggregationResult.metadata.dateRange?.to,
        metadata: aggregationResult.metadata,
      });

      console.log('✅ Aggregated prediction complete:', prediction.prediction_label);

      return {
        success: true,
        prediction,
        features: aggregationResult.features,
        metadata: aggregationResult.metadata,
      };
    } catch (error) {
      console.error('❌ Error getting aggregated prediction:', error);

      return {
        success: false,
        error: error.message || 'Failed to get aggregated prediction',
      };
    }
  };

  const connectFitbit = async () => {
    setLoading(true);
    console.log('🔗 Starting Fitbit connection...');

    try {
      const result = await fitbitService.authenticateWithFitbit();

      console.log('Auth result:', result.success ? 'SUCCESS' : 'FAILED');

      if (!result.success) {
        console.error('❌ Fitbit auth failed:', result.error);

        return {
          success: false,
          error: result.error || 'Fitbit authentication failed',
        };
      }

      console.log('✅ Fitbit auth successful, saving tokens...');

      const tokenData = {
        access_token: result.accessToken,
        refresh_token: result.refreshToken,
        expires_in: result.expiresIn,
        user_id: result.userId,
        connected_at: new Date().toISOString(),
      };

      const { error } = await supabase
        .from('profiles')
        .update({ fitbit_token: tokenData })
        .eq('id', user.id);

      if (error) {
        console.error('❌ Failed to save token:', error);
        throw error;
      }

      fitbitService.setAccessTokenFromProfile(tokenData);

      setIsConnected(true);

      console.log('✅ Token saved, starting historical sync...');

      await syncHistoricalData(7);

      return { success: true };
    } catch (error) {
      console.error('❌ Connect error:', error);

      return {
        success: false,
        error: error.message || 'Failed to connect Fitbit',
      };
    } finally {
      setLoading(false);
    }
  };

  const disconnectFitbit = async () => {
    setLoading(true);
    console.log('🔌 Disconnecting Fitbit...');

    try {
      if (profile?.fitbit_token?.access_token) {
        await fitbitService.revokeToken(profile.fitbit_token.access_token);
      }

      const { error } = await supabase
        .from('profiles')
        .update({ fitbit_token: null })
        .eq('id', user.id);

      if (error) throw error;

      setIsConnected(false);
      setFitbitProfile(null);
      setLastSync(null);

      console.log('✅ Disconnected successfully');

      return { success: true };
    } catch (error) {
      console.error('❌ Disconnect error:', error);

      return {
        success: false,
        error: error.message || 'Failed to disconnect Fitbit',
      };
    } finally {
      setLoading(false);
    }
  };

  const value = {
    isConnected,
    fitbitProfile,
    loading,
    syncing,
    lastSync,
    connectFitbit,
    disconnectFitbit,
    syncDate,
    syncHistoricalData,
    aggregateSleepDataForModel,
    getAggregatedPrediction,
  };

  return (
    <FitbitContext.Provider value={value}>
      {children}
    </FitbitContext.Provider>
  );
};