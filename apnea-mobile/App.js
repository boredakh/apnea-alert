// C:\Users\s4303218\Documents\codebase\apnea-alert\apnea-mobile\App.js
import React, { useEffect, useState, useRef } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';
import { View, ActivityIndicator, Text, Alert } from 'react-native';
import * as Linking from 'expo-linking';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { AuthProvider, useAuth } from './src/context/AuthContext';
import { supabase } from './src/services/supabase';
import HomeScreen from './src/screens/HomeScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import FitbitConnectScreen from './src/screens/FitbitConnectScreen';
import LoginScreen from './src/screens/LoginScreen';
import SignUpScreen from './src/screens/SignUpScreen';
import VerificationScreen from './src/screens/VerificationScreen';
import OnboardingScreen from './src/screens/OnboardingScreen';
import ForgotPasswordScreen from './src/screens/ForgotPasswordScreen';
import ResetPasswordScreen from './src/screens/ResetPasswordScreen';
import BottomTabNavigator from './src/navigation/BottomTabNavigator';
import { FitbitProvider } from './src/context/FitbitContext';
import TestRedirectScreen from './src/screens/TestRedirectScreen';
import FitbitCallbackScreen from './src/screens/FitbitCallbackScreen';
import SleepApneaScreen from './src/screens/SleepApneaScreen';
import SleepAIScreen from './src/screens/SleepAIScreen';
import { SafeAreaProvider } from 'react-native-safe-area-context';

const Stack = createStackNavigator();

// Loading component
const LoadingScreen = () => (
  <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#0f0f0f' }}>
    <ActivityIndicator size="large" color="#6fc6a8" />
    <Text style={{ color: '#fff', marginTop: 10, fontSize: 16 }}>
      Loading...
    </Text>
  </View>
);

function AppNavigator() {
  const { user, profile, loading, getUserFlowState } = useAuth();
  const [flowState, setFlowState] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [isPasswordResetFlow, setIsPasswordResetFlow] = useState(false);
  const [processingToken, setProcessingToken] = useState(false);
  const timeoutRef = useRef(null);

  // Handle Fitbit deep link callback (Single version - removed duplicate)
  useEffect(() => {
    const handleDeepLink = async (event) => {
      console.log('🔗 Deep link received:', event.url);
      const url = event.url;
      
      if (url && url.includes('fitbit-callback')) {
        const code = url.split('code=')[1]?.split('&')[0];
        if (code) {
          console.log('✅ Got code from deep link:', code);
          await AsyncStorage.setItem('fitbit_auth_code', code);
          await AsyncStorage.setItem('fitbit_auth_time', Date.now().toString());
        }
      }
      
      // Also handle password reset deep links
      if (url && url.includes('reset-password')) {
        console.log('📱 Password reset deep link detected:', url);
        const hashMatch = url.match(/#(.*)/);
        if (hashMatch) {
          const params = new URLSearchParams(hashMatch[1]);
          const accessToken = params.get('access_token');
          const refreshToken = params.get('refresh_token');
          const type = params.get('type');
          
          console.log('Reset params:', { type, hasAccessToken: !!accessToken });
          
          if (accessToken && type === 'recovery') {
            setProcessingToken(true);
            const { error } = await supabase.auth.setSession({
              access_token: accessToken,
              refresh_token: refreshToken || '',
            });
            
            if (error) {
              console.error('Error setting session:', error);
              Alert.alert('Error', 'Invalid or expired reset link');
            } else {
              console.log('✅ Session set successfully from deep link');
              setIsPasswordResetFlow(true);
            }
            setProcessingToken(false);
          }
        }
      }
    };

    // Listen for deep links
    const subscription = Linking.addEventListener('url', handleDeepLink);
    
    // Check initial URL
    Linking.getInitialURL().then(url => {
      if (url) handleDeepLink({ url });
    });

    return () => subscription.remove();
  }, []);

  // Main navigation logic
  useEffect(() => {
    const determineFlowState = async () => {
      console.log('🔄 AppNavigator - loading:', loading, 'user:', !!user, 'profile:', !!profile, 'isPasswordResetFlow:', isPasswordResetFlow, 'processingToken:', processingToken);
      
      // Clear any existing timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      // If still processing token or loading auth, wait
      if (processingToken || loading) {
        console.log('⏳ Still processing token or loading auth...');
        return;
      }

      // SPECIAL CASE: If this is a password reset flow AND we have a user (signed in via reset link)
      if (isPasswordResetFlow && user) {
        console.log('🔑 Password reset flow detected with user - showing ResetPassword screen');
        setFlowState('resetPassword');
        setIsReady(true);
        return;
      }

      // If no user, set flowState to login and mark as ready
      if (!user) {
        console.log('🚪 No user, going to Login');
        setFlowState('login');
        setIsReady(true);
        // Clear password reset flag if user is logged out (prevents getting stuck)
        if (isPasswordResetFlow) {
          console.log('🧹 Clearing password reset flag (user logged out)');
          setIsPasswordResetFlow(false);
        }
        return;
      }

      // We have a user, determine flow state
      console.log('✅ User found, determining flow state');
      
      // If profile exists, use it to determine flow state
      if (profile) {
        console.log('📊 Profile available:', { 
          email_verified: profile.email_verified, 
          onboarding_complete: profile.onboarding_complete 
        });
        
        try {
          const state = await getUserFlowState();
          console.log('📍 Flow state from profile:', state);
          setFlowState(state);
          setIsReady(true);
        } catch (error) {
          console.error('❌ Error getting flow state:', error);
          setFlowState('home');
          setIsReady(true);
        }
        return;
      }

      // Profile not loaded yet, wait for it with timeout
      console.log('⏳ Profile not loaded, waiting...');
      timeoutRef.current = setTimeout(async () => {
        console.log('⚠️ Profile load timeout - checking if we can determine flow state without profile');
        
        if (user) {
          // Try to get flow state directly from database without using cached profile
          try {
            const state = await getUserFlowState();
            console.log('📍 Flow state from direct DB query:', state);
            setFlowState(state);
          } catch (err) {
            console.error('❌ Error getting flow state on timeout:', err);
            setFlowState('home');
          }
        } else {
          setFlowState('login');
        }
        setIsReady(true);
      }, 5000);
    };

    determineFlowState();

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [user, profile, loading, getUserFlowState, isPasswordResetFlow, processingToken]);

  // Handle profile updates after initial render
  useEffect(() => {
    // If we're already ready but profile just loaded and flowState might not be correct
    if (isReady && user && profile) {
      const updateFlowStateIfNeeded = async () => {
        // Don't update if we're in a special flow like password reset
        if (isPasswordResetFlow) return;
        
        try {
          const correctState = await getUserFlowState();
          
          // If the current flowState doesn't match what it should be, update it
          if (correctState !== flowState && flowState !== 'resetPassword') {
            console.log('🔄 Updating flow state from', flowState, 'to', correctState, 'after profile load');
            setFlowState(correctState);
            // Note: We don't set isReady to false, just update the flowState
          }
        } catch (error) {
          console.error('Error checking flow state after profile load:', error);
        }
      };
      
      updateFlowStateIfNeeded();
    }
  }, [profile, isReady, user, flowState, getUserFlowState, isPasswordResetFlow]);

  // Show loading while determining flow state
  if (!isReady) {
    return <LoadingScreen />;
  }

  console.log('🎯 Rendering with flowState:', flowState, 'user:', !!user, 'isPasswordResetFlow:', isPasswordResetFlow);

  // Always render a valid navigator structure
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      {flowState === 'resetPassword' && user ? (
        // Password reset flow - show only ResetPassword screen
        <Stack.Screen name="ResetPassword" component={ResetPasswordScreen} />
      ) : !user ? (
        // Auth stack - user not logged in
        <>
          <Stack.Screen name="Login" component={LoginScreen} />
          <Stack.Screen name="SignUp" component={SignUpScreen} />
          <Stack.Screen name="ForgotPassword" component={ForgotPasswordScreen} />
          <Stack.Screen name="ResetPassword" component={ResetPasswordScreen} />
        </>
      ) : (
        // User is logged in - show the main app with bottom tabs
        <>
          {flowState === 'verification' && (
            <Stack.Screen 
              name="Verification" 
              component={VerificationScreen}
              initialParams={{ userId: user.id, email: user.email }}
            />
          )}
          {flowState === 'onboarding' && (
            <Stack.Screen name="Onboarding" component={OnboardingScreen} />
          )}
          {/* Main app with bottom tabs - this contains Home, History, Analysis, Profile */}
          {(flowState === 'home' || !flowState) && (
            <Stack.Screen 
              name="Main" 
              component={BottomTabNavigator}
              options={{ headerShown: false }}
            />
          )}
          {/* Additional screens that can be navigated to from the main app */}
          <Stack.Screen name="TestRedirect" component={TestRedirectScreen} />
          <Stack.Screen 
            name="FitbitConnect" 
            component={FitbitConnectScreen}
            options={{ 
              headerShown: true,
              title: 'Connect Fitbit',
              headerStyle: { backgroundColor: '#2c3e50' },
              headerTintColor: '#fff',
            }}
          />
          <Stack.Screen 
            name="Results" 
            component={ResultsScreen}
            options={{ 
              headerShown: true,
              title: 'Results',
              headerStyle: { backgroundColor: '#2c3e50' },
              headerTintColor: '#fff',
            }}
          />
          <Stack.Screen 
            name="SleepApnea" 
            component={SleepApneaScreen}
            options={{ 
              headerShown: true,
              title: 'Sleep Apnea Detection',
              headerStyle: { backgroundColor: '#1a1a1a' },
              headerTintColor: '#f8fbfa',
              headerTitleStyle: { fontWeight: '600' },
              headerBackTitle: 'Back',
            }}
          />
          <Stack.Screen 
            name="SleepAI" 
            component={SleepAIScreen}
            options={{ 
              headerShown: true,
              title: 'SleepAI Analyst',
              headerStyle: { backgroundColor: '#1a1a1a' },
              headerTintColor: '#f8fbfa',
              headerTitleStyle: { fontWeight: '600' },
              headerBackTitle: 'Back',
            }}
          />
        </>
      )}
    </Stack.Navigator>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AuthProvider>
        <FitbitProvider>
          <NavigationContainer>
            <StatusBar style="auto" />
            <AppNavigator />
          </NavigationContainer>
        </FitbitProvider>
      </AuthProvider>
    </SafeAreaProvider>
  );
}