// src/screens/HistoryScreen.js - SIMPLE VERSION
import React from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

export default function HistoryScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>History</Text>
      <Text style={styles.subtitle}>Past Predictions</Text>
      
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Coming Soon</Text>
        <Text style={styles.description}>
          This screen will show your past apnea predictions.
        </Text>
        
        <Button
          title="Back to Home"
          onPress={() => navigation.navigate('Home')}
          color="#3498db"
        />
      </View>
      
      <Text style={styles.footer}>
        Will be implemented with AsyncStorage
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    textAlign: 'center',
    marginTop: 20,
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 18,
    color: '#7f8c8d',
    textAlign: 'center',
    marginBottom: 40,
  },
  card: {
    backgroundColor: '#fff',
    padding: 25,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15,
  },
  description: {
    fontSize: 16,
    color: '#7f8c8d',
    marginBottom: 25,
    lineHeight: 22,
  },
  footer: {
    position: 'absolute',
    bottom: 20,
    left: 0,
    right: 0,
    textAlign: 'center',
    color: '#95a5a6',
    fontSize: 14,
  },
});