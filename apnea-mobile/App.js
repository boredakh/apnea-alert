import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';

import HomeScreen from './src/screens/HomeScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import FitbitConnectScreen from './src/screens/FitbitConnectScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="auto" />
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen 
          name="Home" 
          component={HomeScreen}
          options={{ 
            title: 'ApneaAlert',
            headerStyle: { backgroundColor: '#3498db' },
            headerTintColor: '#fff',
            headerTitleStyle: { fontWeight: 'bold' },
          }}
        />
        <Stack.Screen 
          name="FitbitConnect" 
          component={FitbitConnectScreen}
          options={{ 
            title: 'Connect Fitbit',
            headerStyle: { backgroundColor: '#2c3e50' },
            headerTintColor: '#fff',
          }}
        />
        <Stack.Screen 
          name="Results" 
          component={ResultsScreen}
          options={{ 
            title: 'Results',
            headerStyle: { backgroundColor: '#2c3e50' },
            headerTintColor: '#fff',
          }}
        />
        <Stack.Screen 
          name="History" 
          component={HistoryScreen}
          options={{ title: 'History' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}