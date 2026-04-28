// src/services/predictionService.js
import { supabase } from './supabase';

// Scale the probability correctly based on your observations
const scaleProbability = (rawProbability) => {
  const rawPercent = rawProbability * 100;
  
  let scaledPercent;
  if (rawPercent < 50) {
    scaledPercent = (rawPercent / 50) * 10;
  } else if (rawPercent < 63) {
    scaledPercent = 10 + ((rawPercent - 50) / 12) * 40;
  } else {
    scaledPercent = 50 + ((rawPercent - 63) / 37) * 50;
  }
  
  const finalPercent = Math.min(scaledPercent, 100);
  return finalPercent / 100;
};

const getRiskLevel = (probability) => {
  const percent = probability * 100;
  if (percent > 70) return 'High';
  if (percent > 30) return 'Moderate';
  return 'Low';
};

// Then update your getAggregatedPrediction function
export const getAggregatedPrediction = async (days = 7) => {
  try {
    // ... your existing code to fetch sleep data and get prediction from ML model ...
    
    // Let's say the ML model returns rawPrediction like:
    // rawPrediction = { apnea_probability: 0.49, confidence: 0.51, ... }
    
    const rawPrediction = await callMLModel(features); // Your existing call
    
    // SCALE the values
    const scaledProbability = scaleProbability(rawPrediction.apnea_probability);
    const scaledConfidence = Math.min(rawPrediction.confidence * 1.1, 0.98);
    const scaledRiskLevel = getRiskLevel(scaledProbability);
    
    // Generate label based on scaled risk
    let predictionLabel = '';
    if (scaledProbability > 0.7) {
      predictionLabel = 'High Risk - Consult Doctor';
    } else if (scaledProbability > 0.3) {
      predictionLabel = 'Moderate Risk - Monitor Symptoms';
    } else {
      predictionLabel = 'Low Risk - Maintain Habits';
    }
    
    // Create the prediction object with SCALED values
    const scaledPrediction = {
      features: features,
      prediction: {
        apnea_probability: scaledProbability,  // SCALED value
        confidence: scaledConfidence,
        risk_level: scaledRiskLevel,
        prediction_label: predictionLabel,
      },
      source: 'fitbit_aggregated',
      fitbit_date: new Date().toISOString().split('T')[0],
      metadata: {
        raw_apnea_probability: rawPrediction.apnea_probability,  // Keep raw for reference
        raw_confidence: rawPrediction.confidence,
        nights: days,
        avgSleepHours: features.mean,
      }
    };
    
    // Save the scaled prediction
    await savePrediction(userId, scaledPrediction);
    
    return { success: true, prediction: scaledPrediction.prediction, features: features };
    
  } catch (error) {
    console.error('Error in getAggregatedPrediction:', error);
    return { success: false, error: error.message };
  }
};

const getRiskLevel = (probability) => {
  const percent = probability * 100;
  if (percent > 70) return 'High';
  if (percent > 30) return 'Moderate';
  return 'Low';
};

export const savePrediction = async (userId, predictionData) => {
  try {
    // Calculate scaled values
    const scaledProbability = scaleProbability(predictionData.prediction.apnea_probability);
    const scaledConfidence = Math.min(predictionData.prediction.confidence * 1.1, 0.98);
    const scaledRiskLevel = getRiskLevel(scaledProbability);
    
    // Generate prediction label based on scaled probability
    let predictionLabel = '';
    if (scaledProbability > 0.7) {
      predictionLabel = 'High Risk - Consult Doctor';
    } else if (scaledProbability > 0.3) {
      predictionLabel = 'Moderate Risk - Monitor Symptoms';
    } else {
      predictionLabel = 'Low Risk - Maintain Habits';
    }
    
    const { data, error } = await supabase
      .from('predictions')
      .insert([
        {
          user_id: userId,
          features: predictionData.features,
          prediction_label: predictionLabel,
          risk_level: scaledRiskLevel,  // Use scaled risk level
          apnea_probability: scaledProbability,  // Use scaled probability
          confidence: scaledConfidence,
          source: predictionData.source || 'manual',
          fitbit_date: predictionData.fitbit_date || null,
          metadata: {
            raw_apnea_probability: predictionData.prediction.apnea_probability,  // Store original raw value
            raw_confidence: predictionData.prediction.confidence,
            scaling_applied: true,
          }
        }
      ])
      .select()
      .single();

    if (error) throw error;
    return { data, error: null };
  } catch (error) {
    console.error('Error saving prediction:', error);
    return { data: null, error };
  }
};

export const getUserPredictions = async (userId) => {
  try {
    const { data, error } = await supabase
      .from('predictions')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) throw error;
    return { data, error: null };
  } catch (error) {
    console.error('Error fetching predictions:', error);
    return { data: null, error };
  }
};

export const deletePrediction = async (predictionId) => {
  try {
    const { error } = await supabase
      .from('predictions')
      .delete()
      .eq('id', predictionId);

    if (error) throw error;
    return { error: null };
  } catch (error) {
    console.error('Error deleting prediction:', error);
    return { error };
  }
};