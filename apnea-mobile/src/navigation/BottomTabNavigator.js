// src/navigation/BottomTabNavigator.js
import React from 'react';
import { StyleSheet, Platform, Text } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

// Import your screens
import HomeScreen from '../screens/HomeScreen';
import HistoryScreen from '../screens/HistoryScreen';
import AnalysisScreen from '../screens/AnalysisScreen';
import ProfileScreen from '../screens/ProfileScreen';

const Tab = createBottomTabNavigator();

// Emoji icons for web
const WebTabIcon = ({ icon, color, size }) => (
  <Text style={{ fontSize: size, color: color, textAlign: 'center' }}>
    {icon}
  </Text>
);

// Conditionally render icons based on platform
const TabBarIcon = ({ focused, color, size, iconName, emoji }) => {
  if (Platform.OS === 'web') {
    return <WebTabIcon icon={emoji} color={color} size={size} />;
  }
  return <Ionicons name={iconName} size={size} color={color} />;
};

export default function BottomTabNavigator() {
  console.log('✅ BottomTabNavigator is rendering');
  
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#0f0f0f' }} edges={['bottom']}>
      <Tab.Navigator
        screenOptions={{
          tabBarStyle: {
            backgroundColor: '#0f0f0f',
            borderTopWidth: 1,
            borderTopColor: '#2a2a2a',
            height: Platform.OS === 'ios' ? 85 : 70,
            paddingBottom: Platform.OS === 'ios' ? 20 : 10,
            paddingTop: 5,
          },
          tabBarActiveTintColor: '#6fc6a8',
          tabBarInactiveTintColor: '#95a5a6',
          tabBarLabelStyle: {
            fontSize: 12,
            fontWeight: '500',
            marginBottom: Platform.OS === 'ios' ? 0 : 5,
          },
          headerShown: false,
        }}
      >
        <Tab.Screen
          name="Home"
          component={HomeScreen}
          options={{
            tabBarLabel: 'Home',
            tabBarIcon: ({ focused, color, size }) => (
              <TabBarIcon 
                focused={focused}
                color={color} 
                size={size} 
                iconName="home"
                emoji="🏠"
              />
            ),
          }}
        />
        
        <Tab.Screen
          name="History"
          component={HistoryScreen}
          options={{
            tabBarLabel: 'History',
            tabBarIcon: ({ focused, color, size }) => (
              <TabBarIcon 
                focused={focused}
                color={color} 
                size={size} 
                iconName="time"
                emoji="📋"
              />
            ),
          }}
        />
        
        <Tab.Screen
          name="Analysis"
          component={AnalysisScreen}
          options={{
            tabBarLabel: 'Analysis',
            tabBarIcon: ({ focused, color, size }) => (
              <TabBarIcon 
                focused={focused}
                color={color} 
                size={size} 
                iconName="analytics"
                emoji="📊"
              />
            ),
          }}
        />
        
        <Tab.Screen
          name="Profile"
          component={ProfileScreen}
          options={{
            tabBarLabel: 'Profile',
            tabBarIcon: ({ focused, color, size }) => (
              <TabBarIcon 
                focused={focused}
                color={color} 
                size={size} 
                iconName="person"
                emoji="👤"
              />
            ),
          }}
        />
      </Tab.Navigator>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({});