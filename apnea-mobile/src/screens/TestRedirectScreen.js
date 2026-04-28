// src/screens/TestRedirectScreen.js
import React, { useEffect } from 'react';
import { View, Text, ScrollView } from 'react-native';
import * as AuthSession from 'expo-auth-session';
import Constants from 'expo-constants';
import { Platform } from 'react-native';

export default function TestRedirectScreen() {
  useEffect(() => {
    const testRedirect = async () => {
      // Test all possible redirect URIs
      const uris = [];
      
      // Method 1: makeRedirectUri with scheme
      uris.push({
        method: 'makeRedirectUri with apneaalert scheme',
        uri: AuthSession.makeRedirectUri({ scheme: 'apneaalert' })
      });
      
      uris.push({
        method: 'makeRedirectUri with exp scheme',
        uri: AuthSession.makeRedirectUri({ scheme: 'exp' })
      });
      
      uris.push({
        method: 'makeRedirectUri with path',
        uri: AuthSession.makeRedirectUri({ 
          scheme: 'apneaalert', 
          path: 'fitbit-callback' 
        })
      });
      
      // Method 2: Default makeRedirectUri
      uris.push({
        method: 'makeRedirectUri default',
        uri: AuthSession.makeRedirectUri()
      });
      
      // Method 3: Hardcoded localhost
      uris.push({
        method: 'localhost:8081',
        uri: 'http://localhost:8081/'
      });
      
      uris.push({
        method: 'localhost:19000',
        uri: 'http://localhost:19000/'
      });
      
      // Method 4: exp:// format
      uris.push({
        method: 'exp://localhost:19000',
        uri: 'exp://localhost:19000/--/fitbit-callback'
      });
      
      console.log('📋 ===== REDIRECT URI TEST =====');
      console.log('Environment:', Constants.executionEnvironment);
      console.log('Platform:', Platform.OS);
      console.log('Is Dev:', __DEV__);
      console.log('All possible URIs:');
      uris.forEach((item, index) => {
        console.log(`${index + 1}. ${item.method}: ${item.uri}`);
      });
    };
    
    testRedirect();
  }, []);
  
  return (
    <ScrollView style={{ flex: 1, backgroundColor: '#0f0f0f', padding: 20 }}>
      <Text style={{ color: '#6fc6a8', fontSize: 24, marginBottom: 20 }}>Redirect URI Test</Text>
      <Text style={{ color: '#fff', marginBottom: 10 }}>Check the console for results!</Text>
      <Text style={{ color: '#95a5a6', marginTop: 20 }}>
        Environment: {Constants.executionEnvironment}{'\n'}
        Platform: {Platform.OS}{'\n'}
        Dev Mode: {__DEV__ ? 'Yes' : 'No'}
      </Text>
    </ScrollView>
  );
}