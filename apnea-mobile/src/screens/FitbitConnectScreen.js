// src/screens/FitbitConnectScreen.js
import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  ScrollView,
} from 'react-native';
import { globalStyles } from '../styles/globalStyles';
import { authenticateWithFitbit, fetchFitbitSleepData, fetchFitbitHeartRateData, transformFitbitDataToModelInputs } from '../api/fitbitService';
import { predictApnea } from '../api/apneaService';

export default function FitbitConnectScreen({ navigation }) {
  const [loading, setLoading] = useState(false);
  const [connecting, setConnecting] = useState(false);

  const handleConnectFitbit = async () => {
    setConnecting(true);
    try {
      const result = await authenticateWithFitbit();
      
      if (result.success) {
        Alert.alert('Success', 'Connected to Fitbit successfully!');
        // Automatically fetch and analyze last night's sleep
        await handleAnalyzeLastNight();
      } else {
        Alert.alert('Connection Failed', result.error || 'Could not connect to Fitbit');
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setConnecting(false);
    }
  };

  const handleAnalyzeLastNight = async () => {
    setLoading(true);
    try {
      // Fetch yesterday's data
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const dateStr = yesterday.toISOString().split('T')[0];
      
      // Fetch sleep and heart rate data
      const [sleepResult, heartRateResult] = await Promise.all([
        fetchFitbitSleepData(dateStr),
        fetchFitbitHeartRateData(dateStr)
      ]);

      if (!sleepResult.success) {
        Alert.alert('Error', 'Could not fetch sleep data. Make sure you wore your Fitbit to bed.');
        return;
      }

      if (!sleepResult.data || sleepResult.data.length === 0) {
        Alert.alert('No Sleep Data', 'No sleep data found for last night.');
        return;
      }

      // Transform Fitbit data to model inputs - ADDED AWAIT HERE
      const modelInputs = await transformFitbitDataToModelInputs(
        sleepResult.data,
        heartRateResult.success ? heartRateResult.data : null
      );

      if (!modelInputs) {
        Alert.alert('Error', 'Could not process sleep data.');
        return;
      }

      // Send to your ML model
      const prediction = await predictApnea(modelInputs);
      
      // Navigate to results with the prediction
      navigation.navigate('Results', {
        prediction: prediction,
        features: modelInputs,
        source: 'fitbit',
        date: dateStr,
      });

    } catch (error) {
      Alert.alert('Analysis Failed', error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={globalStyles.container}>
      <View style={globalStyles.header}>
        <Text style={globalStyles.title}>Connect Fitbit</Text>
        <Text style={globalStyles.subtitle}>
          Analyze your sleep data with ML
        </Text>
      </View>

      <View style={globalStyles.card}>
        <Text style={[globalStyles.cardTitle, { textAlign: 'center' }]}>
          🏃‍♂️ Fitbit Integration
        </Text>
        
        <Text style={{ 
          fontSize: 16, 
          color: '#7f8c8d', 
          textAlign: 'center',
          marginBottom: 25,
          lineHeight: 22,
        }}>
          Connect your Fitbit account to automatically analyze your sleep data 
          using our ML model. The app will fetch your sleep stages, heart rate,
          and other metrics to detect potential sleep apnea.
        </Text>

        {!connecting && !loading && (
          <TouchableOpacity
            style={[globalStyles.button, globalStyles.primaryButton]}
            onPress={handleConnectFitbit}
          >
            <Text style={globalStyles.buttonText}>Connect Fitbit Account</Text>
          </TouchableOpacity>
        )}

        {(connecting || loading) && (
          <View style={{ alignItems: 'center', padding: 20 }}>
            <ActivityIndicator size="large" color="#3498db" />
            <Text style={{ marginTop: 15, color: '#7f8c8d' }}>
              {connecting ? 'Connecting to Fitbit...' : 'Analyzing sleep data...'}
            </Text>
          </View>
        )}

        <View style={{ marginTop: 30, padding: 15, backgroundColor: '#f8f9fa', borderRadius: 10 }}>
          <Text style={{ fontWeight: 'bold', color: '#2c3e50', marginBottom: 10 }}>
            What happens next?
          </Text>
          <Text style={{ color: '#7f8c8d', marginBottom: 5 }}>1. You'll authorize access to your Fitbit data</Text>
          <Text style={{ color: '#7f8c8d', marginBottom: 5 }}>2. We'll fetch your last night's sleep data</Text>
          <Text style={{ color: '#7f8c8d', marginBottom: 5 }}>3. Our ML model analyzes 15 key features</Text>
          <Text style={{ color: '#7f8c8d' }}>4. Get instant apnea risk assessment</Text>
        </View>

        <View style={{ marginTop: 20 }}>
          <Text style={{ fontSize: 12, color: '#95a5a6', fontStyle: 'italic' }}>
            Note: This is a screening tool only. Always consult with healthcare professionals for medical advice.
          </Text>
        </View>
      </View>

      <TouchableOpacity
        style={{ padding: 15, alignItems: 'center' }}
        onPress={() => navigation.navigate('Home')}
      >
        <Text style={{ color: '#3498db', fontSize: 16 }}>
          ← Back to Manual Input
        </Text>
      </TouchableOpacity>
    </ScrollView>
  );
}