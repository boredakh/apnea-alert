// src/context/FitbitContext.js
import React, { createContext, useState, useContext, useEffect } from 'react';
import { Platform } from 'react-native';
import { useAuth } from './AuthContext';
import { supabase } from '../services/supabase';
import * as fitbitService from '../api/fitbitService';

const FitbitContext = createContext({});

export const useFitbit = () => useContext(FitbitContext);

export const FitbitProvider = ({ children }) => {
  const { user, profile } = useAuth();
  const [isConnected, setIsConnected] = useState(false);
  const [fitbitProfile, setFitbitProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [lastSync, setLastSync] = useState(null);

  // Check connection status when profile loads
  useEffect(() => {
    console.log('🔍 FitbitProvider: Checking profile for token...');
    if (profile?.fitbit_token) {
      console.log('✅ Fitbit token found in profile');
      setIsConnected(true);
      
      // 🔥 LOAD TOKEN INTO SERVICE
      fitbitService.setAccessTokenFromProfile(profile.fitbit_token);
      
      fetchFitbitUserProfile();
    } else {
      console.log('❌ No Fitbit token in profile');
      setIsConnected(false);
      setFitbitProfile(null);
    }
  }, [profile]);

  // Fetch Fitbit user profile
  const fetchFitbitUserProfile = async () => {
    try {
      console.log('📡 Fetching Fitbit user profile...');
      const result = await fitbitService.getFitbitProfile();
      if (result.success) {
        console.log('✅ Fitbit profile fetched:', result.data);
        setFitbitProfile(result.data);
      } else {
        console.log('❌ Failed to fetch Fitbit profile:', result.error);
      }
    } catch (error) {
      console.error('Error fetching Fitbit profile:', error);
    }
  };

  // Save Fitbit data to Supabase
  const saveFitbitDataToSupabase = async ({ date, sleepData, heartRateData, hrvData, activityData }) => {
    console.log(`💾 Saving data for ${date}...`);
    
    if (!user) {
      console.log('❌ No user logged in');
      return { success: false, error: 'No user logged in' };
    }

    try {
      const { data: existing } = await supabase
        .from('fitbit_data')
        .select('id')
        .eq('user_id', user.id)
        .eq('date', date)
        .maybeSingle();

      let result;
      
      if (existing) {
        console.log(`📝 Updating existing record for ${date}`);
        result = await supabase
          .from('fitbit_data')
          .update({
            sleep_data: sleepData,
            heart_rate_data: heartRateData,
            hrv_data: hrvData,
            activity_data: activityData,
            updated_at: new Date().toISOString()
          })
          .eq('id', existing.id);
      } else {
        console.log(`📝 Inserting new record for ${date}`);
        result = await supabase
          .from('fitbit_data')
          .insert([{
            user_id: user.id,
            date: date,
            sleep_data: sleepData,
            heart_rate_data: heartRateData,
            hrv_data: hrvData,
            activity_data: activityData,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          }]);
      }

      if (result.error) {
        console.error('❌ Supabase error:', result.error);
        throw result.error;
      }
      
      console.log(`✅ Data saved successfully for ${date}`);
      return { success: true };
    } catch (error) {
      console.error('❌ Save error:', error);
      return { success: false, error: error.message };
    }
  };

  // Sync Fitbit data for a specific date
  const syncDate = async (date) => {
    if (!isConnected) {
      console.log('❌ Not connected to Fitbit');
      return { success: false, error: 'Not connected to Fitbit' };
    }

    try {
      const dateStr = typeof date === 'string' ? date : date.toISOString().split('T')[0];
      console.log(`🔄 Syncing data for ${dateStr}...`);
      
      const sleepResult = await fitbitService.fetchFitbitSleepData(dateStr);
      console.log(`Sleep data: ${sleepResult.success ? '✅' : '❌'}`, sleepResult.data?.length || 0);
      
      const heartRateResult = await fitbitService.fetchFitbitHeartRateData(dateStr);
      console.log(`Heart rate data: ${heartRateResult.success ? '✅' : '❌'}`);
      
      const hrvResult = await fitbitService.fetchFitbitHRVData(dateStr);
      console.log(`HRV data: ${hrvResult.success ? '✅' : '❌'}`);

      const saveResult = await saveFitbitDataToSupabase({
        date: dateStr,
        sleepData: sleepResult.success ? sleepResult.data : null,
        heartRateData: heartRateResult.success ? heartRateResult.data : null,
        hrvData: hrvResult.success ? hrvResult.data : null,
        activityData: null
      });

      if (!saveResult.success) {
        throw new Error(saveResult.error);
      }

      return { 
        success: true, 
        data: {
          sleep: sleepResult.data,
          heartRate: heartRateResult.data,
          hrv: hrvResult.data
        }
      };
    } catch (error) {
      console.error(`❌ Error syncing ${date}:`, error);
      return { success: false, error: error.message };
    }
  };

  // Sync last N days of data
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
        date.setDate(date.getDate() - i);
        dates.push(date.toISOString().split('T')[0]);
      }

      console.log(`📅 Dates to sync: ${dates.join(', ')}`);
      
      const results = [];
      for (const date of dates) {
        console.log(`--- Syncing ${date} ---`);
        const result = await syncDate(date);
        results.push({ date, ...result });
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      setLastSync(new Date().toISOString());
      console.log(`✅ Sync completed. Results:`, results);
      
      return { success: true, results };
    } catch (error) {
      console.error('❌ Error syncing historical data:', error);
      return { success: false, error: error.message };
    } finally {
      setSyncing(false);
    }
  };

  // Save prediction to database
  const savePredictionToDatabase = async ({ features, prediction, source, fitbitDate, metadata }) => {
    try {
      console.log('💾 Saving prediction to database...');
      
      const { data, error } = await supabase
        .from('predictions')
        .insert({
          user_id: user.id,
          features: features,
          prediction_label: prediction.prediction_label,
          risk_level: prediction.risk_level,
          apnea_probability: prediction.apnea_probability,
          confidence: prediction.confidence,
          source: source,
          fitbit_date: fitbitDate,
          metadata: metadata
        })
        .select();
      
      if (error) throw error;
      console.log('✅ Prediction saved to database:', data);
      return { success: true, data };
    } catch (error) {
      console.error('❌ Error saving prediction:', error);
      return { success: false, error: error.message };
    }
  };

  // Aggregate multiple nights of sleep data into averaged features
  const aggregateSleepDataForModel = async (days = 7) => {
    if (!user) return { success: false, error: 'No user logged in' };

    try {
      console.log(`📊 Aggregating sleep data from last ${days} days...`);
      
      // Get last N days of sleep data (only where sleep_data exists)
      const { data, error } = await supabase
        .from('fitbit_data')
        .select('*')
        .eq('user_id', user.id)
        .not('sleep_data', 'is', null)
        .order('date', { ascending: false })
        .limit(days);

      if (error) throw error;
      
      if (!data || data.length === 0) {
        return { success: false, error: 'No sleep data found. Please sync your Fitbit data first.' };
      }

      console.log(`📊 Found ${data.length} nights of sleep data`);
      
      // Track valid nights
      let validNights = 0;
      let totalMinutesAsleep = 0;
      let totalDeepMinutes = 0;
      let totalLightMinutes = 0;
      let totalRemMinutes = 0;
      let totalWakeMinutes = 0;
      let totalSleepEfficiency = 0;
      let totalRestingHeartRate = 0;
      
      // HRV tracking
      let hrvValues = [];
      
      for (const record of data) {
        const sleep = record.sleep_data?.[0];
        const heartRate = record.heart_rate_data?.[0]?.value;
        const hrv = record.hrv_data?.hrv?.[0]?.value;
        
        if (sleep && sleep.minutesAsleep > 0) {
          validNights++;
          totalMinutesAsleep += sleep.minutesAsleep || 0;
          totalDeepMinutes += sleep.levels?.summary?.deep?.minutes || 0;
          totalLightMinutes += sleep.levels?.summary?.light?.minutes || 0;
          totalRemMinutes += sleep.levels?.summary?.rem?.minutes || 0;
          totalWakeMinutes += sleep.minutesAwake || 0;
          totalSleepEfficiency += sleep.efficiency || 0;
        }
        
        if (heartRate?.restingHeartRate) {
          totalRestingHeartRate += heartRate.restingHeartRate;
        }
        
        if (hrv?.rmssd) {
          hrvValues.push(hrv.rmssd);
        }
      }
      
      if (validNights === 0) {
        return { success: false, error: 'No valid sleep records found' };
      }
      
      // Calculate averages
      const avgMinutesAsleep = totalMinutesAsleep / validNights;
      const avgDeepMinutes = totalDeepMinutes / validNights;
      const avgLightMinutes = totalLightMinutes / validNights;
      const avgRemMinutes = totalRemMinutes / validNights;
      const avgWakeMinutes = totalWakeMinutes / validNights;
      const avgSleepEfficiency = totalSleepEfficiency / validNights;
      const avgRestingHeartRate = totalRestingHeartRate / validNights;
      
      // Calculate HRV stats
      const avgHRV = hrvValues.length > 0 
        ? hrvValues.reduce((a, b) => a + b, 0) / hrvValues.length 
        : 35; // Default fallback
      
      // Calculate derived features for the model
      const features = {
        // Sleep duration features
        mean: avgMinutesAsleep / 60, // Convert to hours
        std: (avgDeepMinutes + avgLightMinutes + avgRemMinutes) / 60 / 3,
        min: Math.min(avgDeepMinutes, avgLightMinutes, avgRemMinutes) / 60,
        max: Math.max(avgDeepMinutes, avgLightMinutes, avgRemMinutes) / 60,
        range: (Math.max(avgDeepMinutes, avgLightMinutes, avgRemMinutes) - Math.min(avgDeepMinutes, avgLightMinutes, avgRemMinutes)) / 60,
        rms: Math.sqrt((Math.pow(avgDeepMinutes, 2) + Math.pow(avgLightMinutes, 2) + Math.pow(avgRemMinutes, 2)) / 3) / 60,
        
        // Sleep architecture
        skewness: (avgDeepMinutes - avgLightMinutes) / (avgDeepMinutes + avgLightMinutes + avgRemMinutes),
        kurtosis: (avgRemMinutes - avgDeepMinutes) / (avgDeepMinutes + avgLightMinutes + avgRemMinutes),
        
        // Heart rate features
        hr_mean: avgRestingHeartRate,
        hr_std: avgRestingHeartRate * 0.12,
        
        // RR interval (derived from heart rate)
        rr_mean: 60 / avgRestingHeartRate,
        rr_std: (60 / avgRestingHeartRate) * 0.12,
        
        // HRV features (simplified)
        lf_power: avgHRV * 0.004,
        hf_power: avgHRV * 0.002,
        lf_hf_ratio: 2.0, // Default healthy ratio
      };
      
      // Adjust LF/HF ratio based on sleep quality
      const sleepQuality = avgDeepMinutes / avgMinutesAsleep;
      features.lf_hf_ratio = sleepQuality > 0.2 ? 1.5 : 2.5;
      
      console.log('📊 Aggregated features:', {
        nights: validNights,
        avgSleepHours: (avgMinutesAsleep / 60).toFixed(1),
        avgDeepPct: ((avgDeepMinutes / avgMinutesAsleep) * 100).toFixed(1),
        avgHRV: avgHRV.toFixed(1),
        avgRestingHR: avgRestingHeartRate.toFixed(1)
      });
      
      return { 
        success: true, 
        features,
        metadata: {
          nights: validNights,
          dateRange: {
            from: data[data.length - 1]?.date,
            to: data[0]?.date
          },
          avgSleepHours: avgMinutesAsleep / 60,
          avgDeepPercentage: (avgDeepMinutes / avgMinutesAsleep) * 100
        }
      };
    } catch (error) {
      console.error('❌ Error aggregating sleep data:', error);
      return { success: false, error: error.message };
    }
  };

  // Run prediction using aggregated data
  const getAggregatedPrediction = async (days = 7) => {
    try {
      console.log(`🔬 Running prediction using last ${days} days of data...`);
      
      // Get aggregated features
      const aggregationResult = await aggregateSleepDataForModel(days);
      
      if (!aggregationResult.success) {
        return aggregationResult;
      }
      
      // Run prediction
      const prediction = await fitbitService.predictApnea(aggregationResult.features);
      
      // Save prediction to database with metadata
      await savePredictionToDatabase({
        features: aggregationResult.features,
        prediction: prediction,
        source: 'fitbit_aggregated',
        fitbitDate: aggregationResult.metadata.dateRange?.to,
        metadata: aggregationResult.metadata
      });
      
      console.log('✅ Aggregated prediction complete:', prediction.prediction_label);
      
      return { 
        success: true, 
        prediction, 
        features: aggregationResult.features,
        metadata: aggregationResult.metadata
      };
    } catch (error) {
      console.error('❌ Error getting aggregated prediction:', error);
      return { success: false, error: error.message };
    }
  };

  // Connect Fitbit
  const connectFitbit = async () => {
    setLoading(true);
    console.log('🔗 Starting Fitbit connection...');
    
    try {
      const result = await fitbitService.authenticateWithFitbit();
      console.log('Auth result:', result.success ? 'SUCCESS' : 'FAILED');
      
      if (result.success) {
        console.log('✅ Fitbit auth successful, saving tokens...');
        
        const tokenData = {
          access_token: result.accessToken,
          refresh_token: result.refreshToken,
          expires_in: result.expiresIn,
          user_id: result.userId,
          connected_at: new Date().toISOString()
        };
        
        const { error } = await supabase
          .from('profiles')
          .update({ fitbit_token: tokenData })
          .eq('id', user.id);

        if (error) {
          console.error('❌ Failed to save token:', error);
          throw error;
        }

        // 🔥 LOAD TOKEN INTO SERVICE
        fitbitService.setAccessTokenFromProfile(tokenData);

        setIsConnected(true);
        
        console.log('✅ Token saved, starting historical sync...');
        await syncHistoricalData(7);
        
        return { success: true };
      } else {
        console.error('❌ Fitbit auth failed:', result.error);
        return { success: false, error: result.error };
      }
    } catch (error) {
      console.error('❌ Connect error:', error);
      return { success: false, error: error.message };
    } finally {
      setLoading(false);
    }
  };

  // Disconnect Fitbit
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
      
      console.log('✅ Disconnected successfully');
      return { success: true };
    } catch (error) {
      console.error('❌ Disconnect error:', error);
      return { success: false, error: error.message };
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