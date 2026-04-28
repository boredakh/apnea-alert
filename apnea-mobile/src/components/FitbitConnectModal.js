// src/components/FitbitConnectModal.js
import React from 'react';
import {
  View,
  Text,
  Modal,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
  Alert,
} from 'react-native';
import { useFitbit } from '../context/FitbitContext';
import Svg, { Path } from 'react-native-svg';

const FitbitIcon = () => (
  <Svg width="48" height="48" viewBox="0 0 24 24" fill="none">
    <Path
      d="M6 9a3 3 0 0 1 3-3h6a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H9a3 3 0 0 1-3-3zM9 18v3h6v-3M9 6V3h6v3"
      stroke="#6fc6a8"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </Svg>
);

export default function FitbitConnectModal({ visible, onClose }) {
  const { isConnected, fitbitProfile, loading, connectFitbit, disconnectFitbit } = useFitbit();

  const handleConnect = async () => {
    const result = await connectFitbit();
    if (result.success) {
      Alert.alert('Success', 'Connected to Fitbit successfully!');
      onClose();
    } else {
      Alert.alert('Connection Failed', result.error || 'Could not connect to Fitbit');
    }
  };

  const handleDisconnect = () => {
    Alert.alert(
      'Disconnect Fitbit',
      'Are you sure you want to disconnect your Fitbit account?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Disconnect',
          style: 'destructive',
          onPress: async () => {
            const result = await disconnectFitbit();
            if (result.success) {
              Alert.alert('Success', 'Disconnected from Fitbit');
              onClose();
            } else {
              Alert.alert('Error', result.error || 'Failed to disconnect');
            }
          },
        },
      ]
    );
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <FitbitIcon />
            <Text style={styles.modalTitle}>Fitbit Connection</Text>
          </View>

          {loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#6fc6a8" />
              <Text style={styles.loadingText}>
                Opening Fitbit authorization...
              </Text>
              <Text style={styles.loadingSubText}>
                A popup window will open. Please allow popups for this site.
              </Text>
            </View>
          ) : isConnected && fitbitProfile ? (
            <View style={styles.connectedContent}>
              <View style={styles.profileInfo}>
                <Text style={styles.profileName}>
                  {fitbitProfile.displayName || fitbitProfile.fullName || 'Fitbit User'}
                </Text>
                <View style={[styles.badge, { backgroundColor: 'rgba(111, 198, 168, 0.2)' }]}>
                  <View style={[styles.badgeDot, { backgroundColor: '#6fc6a8' }]} />
                  <Text style={[styles.badgeText, { color: '#6fc6a8' }]}>Connected</Text>
                </View>
              </View>

              <View style={styles.statsContainer}>
                <Text style={styles.statsText}>
                  Your Fitbit account is connected and syncing data.
                </Text>
              </View>

              <TouchableOpacity
                style={[styles.button, styles.disconnectButton]}
                onPress={handleDisconnect}
              >
                <Text style={styles.disconnectButtonText}>Disconnect Fitbit</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <View style={styles.disconnectedContent}>
              <Text style={styles.description}>
                Connect your Fitbit account to automatically analyze your sleep data and get personalized insights.
              </Text>

              <View style={styles.featureList}>
                <Text style={styles.featureItem}>• Automatic sleep analysis</Text>
                <Text style={styles.featureItem}>• Heart rate monitoring</Text>
                <Text style={styles.featureItem}>• Sleep stage tracking</Text>
                <Text style={styles.featureItem}>• Personalized insights</Text>
              </View>

              <TouchableOpacity
                style={[styles.button, styles.connectButton]}
                onPress={handleConnect}
              >
                <Text style={styles.connectButtonText}>Connect Fitbit</Text>
              </TouchableOpacity>
            </View>
          )}

          {!loading && (
            <TouchableOpacity style={styles.closeButton} onPress={onClose}>
              <Text style={styles.closeButtonText}>Close</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 24,
    width: '90%',
    maxWidth: 400,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  modalHeader: {
    alignItems: 'center',
    marginBottom: 24,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#f8fbfa',
    marginTop: 12,
  },
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  loadingText: {
    color: '#f8fbfa',
    fontSize: 16,
    marginTop: 16,
    textAlign: 'center',
  },
  loadingSubText: {
    color: '#95a5a6',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  disconnectedContent: {
    alignItems: 'center',
  },
  description: {
    fontSize: 16,
    color: '#95a5a6',
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 22,
  },
  featureList: {
    alignSelf: 'stretch',
    marginBottom: 24,
  },
  featureItem: {
    fontSize: 14,
    color: '#f8fbfa',
    marginBottom: 8,
  },
  connectedContent: {
    alignItems: 'center',
  },
  profileInfo: {
    alignItems: 'center',
    marginBottom: 20,
  },
  profileName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#f8fbfa',
    marginBottom: 8,
  },
  statsContainer: {
    backgroundColor: '#0f0f0f',
    padding: 16,
    borderRadius: 8,
    marginBottom: 24,
    width: '100%',
  },
  statsText: {
    color: '#95a5a6',
    fontSize: 14,
    textAlign: 'center',
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 16,
  },
  badgeDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  badgeText: {
    fontSize: 14,
    fontWeight: '600',
  },
  button: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    width: '100%',
    marginBottom: 12,
  },
  connectButton: {
    backgroundColor: '#6fc6a8',
  },
  connectButtonText: {
    color: '#0f0f0f',
    fontSize: 16,
    fontWeight: '600',
  },
  disconnectButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#e74c3c',
  },
  disconnectButtonText: {
    color: '#e74c3c',
    fontSize: 16,
    fontWeight: '600',
  },
  closeButton: {
    padding: 12,
    alignItems: 'center',
  },
  closeButtonText: {
    color: '#6fc6a8',
    fontSize: 14,
  },
});