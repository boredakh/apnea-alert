// src/screens/ResetPasswordScreen.js
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ImageBackground,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import { supabase } from '../services/supabase';

export default function ResetPasswordScreen({ navigation }) {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleResetPassword = async () => {
    if (!password || !confirmPassword) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    if (password !== confirmPassword) {
      Alert.alert('Error', 'Passwords do not match');
      return;
    }

    if (password.length < 6) {
      Alert.alert('Error', 'Password must be at least 6 characters');
      return;
    }

    setLoading(true);
    try {
      const { error } = await supabase.auth.updateUser({
        password: password
      });

      if (error) throw error;

      // Show success and redirect
      setSuccess(true);
      Alert.alert(
        'Success!',
        'Your password has been reset successfully.',
        [
          {
            text: 'Go to Login',
            onPress: async () => {
              await supabase.auth.signOut();
              navigation.reset({
                index: 0,
                routes: [{ name: 'Login' }],
              });
            }
          }
        ]
      );
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to reset password');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <ImageBackground
        source={{ uri: 'https://images.pexels.com/photos/8306794/pexels-photo-8306794.jpeg?auto=compress&cs=tinysrgb&w=1500' }}
        style={styles.backgroundImage}
      >
        <View style={styles.container}>
          <View style={styles.modal}>
            <View style={styles.iconContainer}>
              <Text style={styles.icon}>✓</Text>
            </View>
            <Text style={[styles.title, { color: '#6fc6a8' }]}>Password Reset!</Text>
            <Text style={styles.subtitle}>
              Your password has been successfully changed.
            </Text>
            <TouchableOpacity
              style={[styles.button, styles.primaryButton]}
              onPress={async () => {
                await supabase.auth.signOut();
                navigation.reset({
                  index: 0,
                  routes: [{ name: 'Login' }],
                });
              }}
            >
              <Text style={styles.buttonText}>Go to Login</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ImageBackground>
    );
  }

  return (
    <ImageBackground
      source={{ uri: 'https://images.pexels.com/photos/8306794/pexels-photo-8306794.jpeg?auto=compress&cs=tinysrgb&w=1500' }}
      style={styles.backgroundImage}
    >
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.container}
      >
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          <View style={styles.modal}>
            <View style={styles.iconContainer}>
              <Text style={styles.icon}>🔑</Text>
            </View>

            <View style={styles.header}>
              <Text style={styles.title}>Reset Password</Text>
              <Text style={styles.subtitle}>
                Enter your new password below.
              </Text>
            </View>

            <View style={styles.form}>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>New Password</Text>
                <View style={styles.passwordWrapper}>
                  <TextInput
                    style={[styles.input, styles.passwordInput]}
                    value={password}
                    onChangeText={setPassword}
                    placeholder="Enter new password"
                    placeholderTextColor="#666"
                    secureTextEntry={!showPassword}
                    editable={!loading}
                  />
                  <TouchableOpacity
                    style={styles.passwordToggle}
                    onPress={() => setShowPassword(!showPassword)}
                  >
                    <Text style={styles.toggleText}>
                      {showPassword ? '👁️' : '👁️‍🗨️'}
                    </Text>
                  </TouchableOpacity>
                </View>
                <Text style={styles.hintText}>Minimum 6 characters</Text>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.label}>Confirm New Password</Text>
                <TextInput
                  style={styles.input}
                  value={confirmPassword}
                  onChangeText={setConfirmPassword}
                  placeholder="Re-enter new password"
                  placeholderTextColor="#666"
                  secureTextEntry={!showPassword}
                  editable={!loading}
                />
              </View>

              <TouchableOpacity
                style={[styles.button, styles.primaryButton, loading && styles.buttonDisabled]}
                onPress={handleResetPassword}
                disabled={loading}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.buttonText}>Reset Password</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </ImageBackground>
  );
}

const styles = {
  backgroundImage: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  container: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 20,
  },
  modal: {
    backgroundColor: '#0f0f0f',
    borderRadius: 10,
    padding: 30,
    borderWidth: 1,
    borderColor: '#2a2a2a',
    alignItems: 'center',
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(111, 198, 168, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  icon: {
    fontSize: 40,
  },
  header: {
    marginBottom: 30,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#f8fbfa',
    marginBottom: 10,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: '#d6dfdd',
    textAlign: 'center',
    opacity: 0.85,
  },
  form: {
    gap: 15,
    width: '100%',
  },
  inputGroup: {
    marginBottom: 15,
  },
  label: {
    color: '#f8fbfa',
    marginBottom: 5,
    fontSize: 14,
    fontWeight: '500',
  },
  hintText: {
    color: '#d6dfdd',
    fontSize: 12,
    marginTop: 5,
    opacity: 0.7,
  },
  input: {
    backgroundColor: '#1a1a1a',
    borderWidth: 1,
    borderColor: '#2a2a2a',
    borderRadius: 4,
    padding: 12,
    color: '#f8fbfa',
    fontSize: 16,
  },
  passwordWrapper: {
    position: 'relative',
  },
  passwordInput: {
    paddingRight: 45,
  },
  passwordToggle: {
    position: 'absolute',
    right: 12,
    top: 12,
  },
  toggleText: {
    fontSize: 20,
  },
  button: {
    padding: 15,
    borderRadius: 4,
    alignItems: 'center',
    marginVertical: 5,
  },
  primaryButton: {
    backgroundColor: '#6fc6a8',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    color: '#0f0f0f',
    fontSize: 16,
    fontWeight: '600',
  },
};