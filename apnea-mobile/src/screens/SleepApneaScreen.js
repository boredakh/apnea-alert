// src/screens/SleepApneaScreen.js
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { useFitbit } from '../context/FitbitContext';
import Svg, { Path } from 'react-native-svg';

export default function SleepApneaScreen({ navigation }) {
  const { isConnected, getLatestSleepDataAndPredict, syncing, getAggregatedPrediction } = useFitbit();
  const [analyzing, setAnalyzing] = useState(false);

  const handleStartAnalysis = async () => {
  if (!isConnected) {
    Alert.alert(
      'Not Connected',
      'Please connect your Fitbit account first to analyze your sleep data.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Connect Fitbit', onPress: () => navigation.navigate('Analysis') }
      ]
    );
    return;
  }

  setAnalyzing(true);
  try {
    // This gets raw results from ML model
    const result = await getAggregatedPrediction(7);
    
    if (result.success) {
      // Navigate to Results - the savePrediction already saved scaled values
      navigation.navigate('Results', {
        prediction: result.prediction,
        features: result.features,
        source: 'fitbit_aggregated',
        metadata: result.metadata
      });
    } else {
      Alert.alert(
        'Analysis Failed',
        result.error || 'Could not analyze sleep data. Please sync your Fitbit data first.',
        [
          { text: 'OK' },
          { text: 'Sync Data', onPress: () => navigation.navigate('Analysis') }
        ]
      );
    }
  } catch (error) {
    Alert.alert('Error', error.message || 'An unexpected error occurred');
  } finally {
    setAnalyzing(false);
  }
};

  // Icons remain the same as before...
  const HeartIcon = () => (
    <Svg width="80" height="80" viewBox="0 0 24 24" fill="none">
      <Path
        d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"
        fill="#6fc6a8"
        stroke="#6fc6a8"
        strokeWidth="1"
      />
    </Svg>
  );

  const MoonIcon = () => (
    <Svg width="40" height="40" viewBox="0 0 24 24" fill="none">
      <Path
        d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </Svg>
  );

  const BrainIcon = () => (
    <Svg width="40" height="40" viewBox="0 0 24 24" fill="none">
      <Path
        d="M12 4a4 4 0 0 1 4 4c0 1.5-.8 2.8-2 3.5V15a2 2 0 0 1-4 0v-3.5c-1.2-.7-2-2-2-3.5a4 4 0 0 1 4-4z"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <Path
        d="M8 13c-1.5.5-3 1.5-3 3.5 0 2 1.5 3 3 3"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <Path
        d="M16 13c1.5.5 3 1.5 3 3.5 0 2-1.5 3-3 3"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </Svg>
  );

  const DocumentIcon = () => (
    <Svg width="40" height="40" viewBox="0 0 24 24" fill="none">
      <Path
        d="M9 12h6m-6 4h6m2-12H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </Svg>
  );

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Sleep Apnea Detection</Text>
        <Text style={styles.headerSubtitle}>
          Analyze your sleep patterns using advanced AI
        </Text>
      </View>

      <View style={styles.heroSection}>
        <View style={styles.heroIcon}>
          <HeartIcon />
        </View>
        <Text style={styles.heroTitle}>AI-Powered Analysis</Text>
        <Text style={styles.heroDescription}>
          Our machine learning model analyzes your sleep stages, heart rate, 
          and breathing patterns to detect potential sleep apnea indicators.
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>How It Works</Text>
        
        <View style={styles.stepContainer}>
          <View style={styles.stepIcon}>
            <MoonIcon />
          </View>
          <View style={styles.stepContent}>
            <Text style={styles.stepTitle}>1. Sync Fitbit Data</Text>
            <Text style={styles.stepDescription}>
              Connect your Fitbit and sync your sleep data
            </Text>
          </View>
        </View>
        
        <View style={styles.stepContainer}>
          <View style={styles.stepIcon}>
            <BrainIcon />
          </View>
          <View style={styles.stepContent}>
            <Text style={styles.stepTitle}>2. AI Analysis</Text>
            <Text style={styles.stepDescription}>
              Our model analyzes 15 key biometric features
            </Text>
          </View>
        </View>
        
        <View style={styles.stepContainer}>
          <View style={styles.stepIcon}>
            <DocumentIcon />
          </View>
          <View style={styles.stepContent}>
            <Text style={styles.stepTitle}>3. Get Results</Text>
            <Text style={styles.stepDescription}>
              Receive instant risk assessment and recommendations
            </Text>
          </View>
        </View>
      </View>

      <View style={[styles.statusCard, isConnected ? styles.connectedCard : styles.disconnectedCard]}>
        <View style={styles.statusDot}>
          <View style={[styles.dot, isConnected ? styles.dotConnected : styles.dotDisconnected]} />
        </View>
        <View style={styles.statusContent}>
          <Text style={styles.statusLabel}>Fitbit Status</Text>
          <Text style={styles.statusValue}>
            {isConnected ? 'Connected' : 'Not Connected'}
          </Text>
        </View>
      </View>

      <TouchableOpacity
        style={[styles.analyzeButton, (!isConnected || analyzing) && styles.buttonDisabled]}
        onPress={handleStartAnalysis}
        disabled={!isConnected || analyzing}
      >
        {analyzing ? (
          <ActivityIndicator color="#0f0f0f" size="small" />
        ) : (
          <>
            <Svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <Path
                d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"
                stroke="#0f0f0f"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </Svg>
            <Text style={styles.analyzeButtonText}>Start Analysis</Text>
          </>
        )}
      </TouchableOpacity>

      <View style={styles.disclaimer}>
        <Text style={styles.disclaimerText}>
          ⚠️ This is a screening tool only, not a medical diagnosis. 
          Always consult with a healthcare professional for clinical evaluation.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f0f',
  },
  contentContainer: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 24,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#f8fbfa',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#95a5a6',
    lineHeight: 20,
  },
  heroSection: {
    alignItems: 'center',
    marginBottom: 24,
    padding: 20,
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  heroIcon: {
    marginBottom: 16,
  },
  heroTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#6fc6a8',
    marginBottom: 8,
  },
  heroDescription: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
    lineHeight: 20,
  },
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#f8fbfa',
    marginBottom: 20,
  },
  stepContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  stepIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(111, 198, 168, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f8fbfa',
    marginBottom: 4,
  },
  stepDescription: {
    fontSize: 13,
    color: '#95a5a6',
  },
  statusCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
    borderWidth: 1,
  },
  connectedCard: {
    borderColor: '#6fc6a8',
  },
  disconnectedCard: {
    borderColor: '#e74c3c',
  },
  statusDot: {
    marginRight: 12,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  dotConnected: {
    backgroundColor: '#6fc6a8',
  },
  dotDisconnected: {
    backgroundColor: '#e74c3c',
  },
  statusContent: {
    flex: 1,
  },
  statusLabel: {
    fontSize: 12,
    color: '#95a5a6',
    marginBottom: 2,
  },
  statusValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f8fbfa',
  },
  analyzeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#6fc6a8',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    gap: 12,
    marginBottom: 20,
  },
  analyzeButtonText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#0f0f0f',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  disclaimer: {
    backgroundColor: 'rgba(231, 76, 60, 0.1)',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(231, 76, 60, 0.3)',
  },
  disclaimerText: {
    fontSize: 12,
    color: '#e74c3c',
    textAlign: 'center',
    lineHeight: 18,
  },
});