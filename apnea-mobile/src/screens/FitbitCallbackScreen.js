// src/screens/FitbitCallbackScreen.js
import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator, Platform } from 'react-native';
import { useFitbit } from '../context/FitbitContext';

export default function FitbitCallbackScreen({ navigation }) {
  const { handleOAuthCallback } = useFitbit();

  useEffect(() => {
    const handleCallback = async () => {
      if (Platform.OS === 'web') {
        // For web, the callback is handled by the HTML page and postMessage
        // Just navigate back to the app after a short delay
        setTimeout(() => {
          navigation.navigate('Main');
        }, 2000);
      } else {
        // For native, handle the deep link
        // Get the code from the URL (this will be implemented when testing on mobile)
        console.log('Native callback - implement deep link handling');
      }
    };

    handleCallback();
  }, []);

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#0f0f0f' }}>
      <ActivityIndicator size="large" color="#6fc6a8" />
      <Text style={{ color: '#fff', marginTop: 10, fontSize: 16 }}>
        Completing Fitbit connection...
      </Text>
      <Text style={{ color: '#95a5a6', marginTop: 5, fontSize: 14 }}>
        You can close the popup window
      </Text>
    </View>
  );
}