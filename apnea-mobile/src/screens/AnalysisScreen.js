// src/screens/AnalysisScreen.js
import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Animated,
  Platform,
  Easing,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Svg, { Path } from 'react-native-svg';
import { useFitbit } from '../context/FitbitContext';
import FitbitConnectModal from '../components/FitbitConnectModal';

export default function AnalysisScreen({ navigation }) {
  const { isConnected, loading: fitbitLoading, syncHistoricalData, syncing } = useFitbit();
  const [modalVisible, setModalVisible] = useState(false);
  const [shakeAnimation] = useState(new Animated.Value(0));
  const spinValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (syncing) {
      // Start spinning animation when syncing begins
      Animated.loop(
        Animated.timing(spinValue, {
          toValue: 1,
          duration: 1000,
          easing: Easing.linear,
          useNativeDriver: true,
        })
      ).start();
    } else {
      // Reset animation when syncing stops
      spinValue.setValue(0);
    }
  }, [syncing]);

  const spin = spinValue.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  const startShake = () => {
    Animated.sequence([
      Animated.timing(shakeAnimation, { toValue: 10, duration: 100, useNativeDriver: true }),
      Animated.timing(shakeAnimation, { toValue: -10, duration: 100, useNativeDriver: true }),
      Animated.timing(shakeAnimation, { toValue: 10, duration: 100, useNativeDriver: true }),
      Animated.timing(shakeAnimation, { toValue: 0, duration: 100, useNativeDriver: true }),
    ]).start();
  };

  const handleSync = () => {
    if (!isConnected) {
      startShake();
      Alert.alert(
        'Not Connected',
        'Please connect your Fitbit account first',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Connect', onPress: () => setModalVisible(true) }
        ]
      );
      return;
    }

    const performSync = async () => {
      const result = await syncHistoricalData(7);
      if (result.success) {
        Alert.alert('Success', 'Data synced successfully!');
      } else {
        Alert.alert('Error', result.error || 'Failed to sync data');
      }
    };
    
    performSync();
  };

  // Smartwatch Icon
  const WatchIcon = () => (
    <Svg width="24" height="24" viewBox="0 0 24 24" fill="none">
      <Path
        d="M6 9a3 3 0 0 1 3-3h6a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H9a3 3 0 0 1-3-3zm3 9v3h6v-3M9 6V3h6v3"
        stroke={isConnected ? "#6fc6a8" : "#e74c3c"}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );

  // Apnea Icon
  const ApneaIcon = () => (
    <Svg width="24" height="24" viewBox="0 0 24 24" fill="none">
      <Path
        d="M6.081 20C7.693 20 9 18.665 9 17.02V7.257C9 6.563 8.448 6 7.768 6c-.205 0-.405.052-.584.15l-.13.083C5.594 7.292 4.622 8.88 3.65 12.057q-.63 2.055-.648 4.775c-.012 1.675 1.261 3.054 2.877 3.161zm11.839 0C16.307 20 15 18.665 15 17.02V7.257C15 6.563 15.552 6 16.233 6c.204 0 .405.052.584.15l.13.083c1.46 1.059 2.432 2.647 3.405 5.824q.63 2.055.648 4.775c.012 1.675-1.261 3.054-2.878 3.161zM9 12a3 3 0 0 0 3-3a3 3 0 0 0 3 3m-3-8v5"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );

  // AI Icon
  const AIIcon = () => (
    <Svg width="24" height="24" viewBox="0 0 24 24" fill="none">
      <Path
        d="M12 18V5m3 8a4.17 4.17 0 0 1-3-4a4.17 4.17 0 0 1-3 4m8.598-6.5A3 3 0 1 0 12 5a3 3 0 1 0-5.598 1.5"
        stroke="#f39c12"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Path d="M17.997 5.125a4 4 0 0 1 2.526 5.77" stroke="#f39c12" strokeWidth="2" />
      <Path d="M18 18a4 4 0 0 0 2-7.464" stroke="#f39c12" strokeWidth="2" />
      <Path d="M19.967 17.483A4 4 0 1 1 12 18a4 4 0 1 1-7.967-.517" stroke="#f39c12" strokeWidth="2" />
      <Path d="M6 18a4 4 0 0 1-2-7.464" stroke="#f39c12" strokeWidth="2" />
      <Path d="M6.003 5.125a4 4 0 0 0-2.526 5.77" stroke="#f39c12" strokeWidth="2" />
    </Svg>
  );

  // Right Arrow Icon
  const ArrowIcon = () => (
    <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <Path
        d="m9 18l6-6l-6-6"
        stroke="#95a5a6"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );

  // Spinner Icon (continuous)
  const SpinnerIcon = () => (
    <Animated.View style={{ transform: [{ rotate: spin }] }}>
      <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <Path
          d="M21 12a9 9 0 1 1-6-8.5"
          stroke="#0f0f0f"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </Svg>
    </Animated.View>
  );

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView 
        style={styles.container} 
        contentContainerStyle={styles.contentContainer}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Start Analysis</Text>
          <Text style={styles.headerSubtitle}>
            Connect your Fitbit to get personalized sleep insights
          </Text>
        </View>

        {/* Status Card */}
        <Animated.View 
          style={[
            styles.statusCard,
            { transform: [{ translateX: shakeAnimation }] }
          ]}
        >
          <TouchableOpacity 
            style={styles.statusCardContent}
            onPress={() => setModalVisible(true)}
            activeOpacity={0.7}
          >
            <View style={[styles.iconWrapper, { backgroundColor: isConnected ? 'rgba(111, 198, 168, 0.15)' : 'rgba(231, 76, 60, 0.15)' }]}>
              <WatchIcon />
            </View>
            <View style={styles.statusInfo}>
              <Text style={styles.statusLabel}>Fitbit Account</Text>
              <View style={[
                styles.statusBadge,
                { backgroundColor: isConnected ? 'rgba(111, 198, 168, 0.2)' : 'rgba(231, 76, 60, 0.2)' }
              ]}>
                <View style={[
                  styles.badgeDot,
                  { backgroundColor: isConnected ? '#6fc6a8' : '#e74c3c' }
                ]} />
                <Text style={[
                  styles.badgeText,
                  { color: isConnected ? '#6fc6a8' : '#e74c3c' }
                ]}>
                  {fitbitLoading ? 'Loading...' : (isConnected ? 'Connected' : 'Connect')}
                </Text>
              </View>
            </View>
            <View style={styles.chevronIcon}>
              <ArrowIcon />
            </View>
          </TouchableOpacity>
        </Animated.View>

        {/* Sync Button */}
        <TouchableOpacity
          style={[styles.syncButton, (!isConnected || syncing) && styles.syncButtonDisabled]}
          onPress={handleSync}
          disabled={!isConnected || syncing}
          activeOpacity={0.8}
        >
          {syncing ? (
            <>
              <SpinnerIcon />
              <Text style={styles.syncButtonText}>Syncing...</Text>
            </>
          ) : (
            <Text style={styles.syncButtonText}>Sync Fitbit Data</Text>
          )}
        </TouchableOpacity>

        {/* Analysis Cards */}
        <View style={styles.analysisGrid}>
          {/* Apnea Detection Card */}
          <TouchableOpacity
            style={styles.analysisCard}
            onPress={() => navigation.navigate('SleepApnea')}
            activeOpacity={0.7}
          >
            <View style={[styles.cardIconBox, { backgroundColor: 'rgba(111, 198, 168, 0.15)' }]}>
              <ApneaIcon />
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>Sleep Apnea Detection</Text>
              <Text style={styles.cardDescription}>
                Non-clinical breathing pattern analysis using biometric sensors
              </Text>
            </View>
            <View style={styles.cardArrow}>
              <ArrowIcon />
            </View>
          </TouchableOpacity>

          {/* SleepAI Card */}
          <TouchableOpacity
            style={styles.analysisCard}
            onPress={() => navigation.navigate('SleepAI')}
            activeOpacity={0.7}
          >
            <View style={[styles.cardIconBox, { backgroundColor: 'rgba(243, 156, 18, 0.15)' }]}>
              <AIIcon />
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>SleepAI Analyst</Text>
              <Text style={styles.cardDescription}>
                Personalized insights and neural sleep optimization
              </Text>
            </View>
            <View style={styles.cardArrow}>
              <ArrowIcon />
            </View>
          </TouchableOpacity>
        </View>

        {/* Info Footer */}
        <View style={styles.infoFooter}>
          <Text style={styles.infoText}>
            💡 Your data is processed locally and never shared without permission
          </Text>
        </View>
      </ScrollView>

      <FitbitConnectModal 
        visible={modalVisible}
        onClose={() => setModalVisible(false)}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0f0f0f',
  },
  container: {
    flex: 1,
    backgroundColor: '#0f0f0f',
  },
  contentContainer: {
    paddingHorizontal: 20,
    paddingBottom: 40,
    paddingTop: Platform.OS === 'ios' ? 10 : 20,
  },
  header: {
    marginBottom: 28,
    marginTop: Platform.OS === 'ios' ? 0 : 10,
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#f8fbfa',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 15,
    color: '#95a5a6',
    lineHeight: 22,
  },
  statusCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#2a2a2a',
    overflow: 'hidden',
  },
  statusCardContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  iconWrapper: {
    width: 52,
    height: 52,
    borderRadius: 26,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  statusInfo: {
    flex: 1,
  },
  statusLabel: {
    fontSize: 16,
    fontWeight: '500',
    color: '#f8fbfa',
    marginBottom: 6,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 20,
    alignSelf: 'flex-start',
  },
  badgeDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  badgeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  chevronIcon: {
    opacity: 0.5,
  },
  syncButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#6fc6a8',
    paddingVertical: 14,
    borderRadius: 12,
    gap: 10,
    marginBottom: 28,
  },
  syncButtonDisabled: {
    opacity: 0.5,
  },
  syncButtonText: {
    color: '#0f0f0f',
    fontSize: 16,
    fontWeight: '600',
  },
  analysisGrid: {
    gap: 16,
  },
  analysisCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  cardIconBox: {
    width: 56,
    height: 56,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  cardContent: {
    flex: 1,
    marginRight: 8,
  },
  cardTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: '#f8fbfa',
    marginBottom: 4,
  },
  cardDescription: {
    fontSize: 13,
    color: '#95a5a6',
    lineHeight: 18,
  },
  cardArrow: {
    opacity: 0.5,
  },
  infoFooter: {
    marginTop: 32,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  infoText: {
    fontSize: 12,
    color: '#6fc6a8',
    textAlign: 'center',
    lineHeight: 18,
  },
});