// src/screens/HomeScreen.js - UPDATED WITH FEATURE INPUTS
import React, { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { globalStyles } from '../styles/globalStyles';
import { predictApnea, EXAMPLE_FEATURES, FEATURE_DESCRIPTIONS } from '../api/apneaService';

export default function HomeScreen({ navigation }) {
  const [features, setFeatures] = useState(EXAMPLE_FEATURES);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    // Validate all features are numbers
    for (const [key, value] of Object.entries(features)) {
      if (isNaN(value) || value === '') {
        Alert.alert('Invalid Input', `Please enter a valid number for ${key}`);
        return;
      }
    }

    setLoading(true);
    try {
      const result = await predictApnea(features);
      
      // Navigate to results screen
      navigation.navigate('Results', {
        prediction: result,
        features: features,
      });
    } catch (error) {
      Alert.alert(
        'Prediction Failed',
        error.message || 'Could not connect to the API'
      );
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUseExample = () => {
    setFeatures(EXAMPLE_FEATURES);
    Alert.alert('Example Loaded', 'Example values loaded. Click "Analyze" to test.');
  };

  const handleReset = () => {
    setFeatures({
      mean: 0, std: 0, min: 0, max: 0, range: 0,
      rms: 0, skewness: 0, kurtosis: 0,
      hr_mean: 0, hr_std: 0, rr_mean: 0, rr_std: 0,
      lf_power: 0, hf_power: 0, lf_hf_ratio: 0
    });
  };

  const updateFeature = (key, value) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      setFeatures(prev => ({
        ...prev,
        [key]: numValue,
      }));
    } else if (value === '' || value === '-') {
      setFeatures(prev => ({
        ...prev,
        [key]: value,
      }));
    }
  };

  // Group features for better organization
  const featureGroups = [
    {
      title: 'ECG Signal Features',
      features: ['mean', 'std', 'min', 'max', 'range', 'rms', 'skewness', 'kurtosis']
    },
    {
      title: 'Heart Rate Features',
      features: ['hr_mean', 'hr_std', 'rr_mean', 'rr_std']
    },
    {
      title: 'Frequency Domain Features',
      features: ['lf_power', 'hf_power', 'lf_hf_ratio']
    }
  ];

  return (
    <ScrollView style={globalStyles.container} contentContainerStyle={globalStyles.scrollContainer}>
      <View style={globalStyles.header}>
        <Text style={globalStyles.title}>ApneaAlert</Text>
        <Text style={globalStyles.subtitle}>Enter 15 ECG Features</Text>
      </View>

      {featureGroups.map((group, groupIndex) => (
        <View key={groupIndex} style={globalStyles.card}>
          <Text style={globalStyles.cardTitle}>{group.title}</Text>
          
          {group.features.map((key) => (
            <View key={key} style={globalStyles.inputGroup}>
              <Text style={globalStyles.inputLabel}>
                {key.replace('_', ' ').toUpperCase()}
              </Text>
              <Text style={globalStyles.inputDescription}>
                {FEATURE_DESCRIPTIONS[key]}
              </Text>
              <TextInput
                style={globalStyles.input}
                value={String(features[key])}
                onChangeText={(text) => updateFeature(key, text)}
                keyboardType="numeric"
                placeholder={`Enter ${key}`}
                placeholderTextColor="#95a5a6"
              />
            </View>
          ))}
        </View>
      ))}

      {/* Action Buttons */}
      <View style={globalStyles.card}>
        <View style={globalStyles.buttonGroup}>
          <TouchableOpacity
            style={[globalStyles.button, globalStyles.secondaryButton]}
            onPress={handleUseExample}
            disabled={loading}
          >
            <Text style={globalStyles.buttonText}>Load Example</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[globalStyles.button, globalStyles.secondaryButton]}
            onPress={handleReset}
            disabled={loading}
          >
            <Text style={globalStyles.buttonText}>Clear All</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          style={[globalStyles.button, globalStyles.primaryButton, { marginTop: 10 }]}
          onPress={handlePredict}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={globalStyles.buttonText}>Analyze for Apnea</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={{ padding: 15, alignItems: 'center' }}
          onPress={() => navigation.navigate('History')}
        >
          <Text style={{ color: '#3498db', fontSize: 16 }}>
            View Prediction History →
          </Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity
  style={[globalStyles.button, { backgroundColor: '#00A0DF', marginTop: 15 }]}
  onPress={() => navigation.navigate('FitbitConnect')}
>
  <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
    <Text style={{ color: '#fff', fontSize: 18, marginRight: 10 }}>⌚</Text>
    <Text style={globalStyles.buttonText}>Analyze with Fitbit Data</Text>
  </View>
</TouchableOpacity>

      <View style={globalStyles.footer}>
        <Text style={globalStyles.footerText}>
          Machine Learning Model: Random Forest (97.6% accuracy)
        </Text>
        <Text style={globalStyles.footerText}>
          API: https://apnea-alert.onrender.com
        </Text>
      </View>
    </ScrollView>
  );
}