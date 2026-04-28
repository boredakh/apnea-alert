// C:\Users\s4303218\Documents\codebase\apnea-alert\apnea-mobile\src\screens\ForgotPasswordScreen.js
import React, { useState } from 'react';
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
import { useAuth } from '../context/AuthContext';

export default function ForgotPasswordScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [emailSent, setEmailSent] = useState(false);
  const { resetPassword } = useAuth();

  const handleResetPassword = async () => {
    if (!email) {
      Alert.alert('Error', 'Please enter your email address');
      return;
    }

    setLoading(true);
    try {
      const { error } = await resetPassword(email);
      
      if (error) throw error;

      // Show success state instead of alert
      setEmailSent(true);
      
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to send reset email');
    } finally {
      setLoading(false);
    }
  };

  const handleResendEmail = async () => {
    setLoading(true);
    try {
      const { error } = await resetPassword(email);
      
      if (error) throw error;
      
      // Show a small confirmation toast/alert
      Alert.alert('Success', 'Reset link resent! Please check your email.');
      
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to resend email');
    } finally {
      setLoading(false);
    }
  };

  if (emailSent) {
    return (
      <ImageBackground
        source={{ uri: 'https://images.pexels.com/photos/8306794/pexels-photo-8306794.jpeg?auto=compress&cs=tinysrgb&w=1500' }}
        style={styles.backgroundImage}
      >
        <View style={styles.container}>
          <ScrollView contentContainerStyle={styles.scrollContainer}>
            <View style={styles.modal}>
              <View style={[styles.iconContainer, { backgroundColor: 'rgba(111, 198, 168, 0.2)' }]}>
                <Text style={[styles.icon, { color: '#6fc6a8' }]}>✉️</Text>
              </View>

              <View style={styles.header}>
                <Text style={[styles.title, { color: '#6fc6a8' }]}>Check Your Email</Text>
                <Text style={styles.subtitle}>
                  We've sent a password reset link to:
                </Text>
                <Text style={styles.emailText}>{email}</Text>
              </View>

              <View style={styles.successInfo}>
                <Text style={styles.infoText}>
                  Click the link in the email to reset your password. The link will expire in 1 hour.
                </Text>
              </View>

              <View style={styles.divider} />

              <View style={styles.resendSection}>
                <Text style={styles.resendText}>Didn't receive the email?</Text>
                <TouchableOpacity 
                  onPress={handleResendEmail}
                  disabled={loading}
                >
                  <Text style={styles.resendLink}>
                    {loading ? 'Sending...' : 'Click to resend'}
                  </Text>
                </TouchableOpacity>
              </View>

              <TouchableOpacity
                style={[styles.button, styles.secondaryButton, loading && styles.buttonDisabled]}
                onPress={() => {
                  setEmailSent(false);
                  setEmail('');
                }}
                disabled={loading}
              >
                <Text style={styles.secondaryButtonText}>Use Different Email</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.backButton}
                onPress={() => navigation.navigate('Login')}
                disabled={loading}
              >
                <Text style={styles.backButtonText}>← Back to Login</Text>
              </TouchableOpacity>

              <Text style={styles.spamNote}>
                Don't forget to check your spam folder!
              </Text>
            </View>
          </ScrollView>
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
              <Text style={styles.icon}>🔐</Text>
            </View>

            <View style={styles.header}>
              <Text style={styles.title}>Forgot Password?</Text>
              <Text style={styles.subtitle}>
                Enter your email address and we'll send you a link to reset your password.
              </Text>
            </View>

            <View style={styles.form}>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>Email Address</Text>
                <TextInput
                  style={styles.input}
                  value={email}
                  onChangeText={setEmail}
                  placeholder="name@example.com"
                  placeholderTextColor="#666"
                  keyboardType="email-address"
                  autoCapitalize="none"
                  autoCorrect={false}
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
                  <Text style={styles.buttonText}>Send Reset Link</Text>
                )}
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.backButton}
                onPress={() => navigation.navigate('Login')}
                disabled={loading}
              >
                <Text style={styles.backButtonText}>← Back to Login</Text>
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
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(111, 198, 168, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
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
    lineHeight: 20,
  },
  emailText: {
    color: '#6fc6a8',
    fontSize: 16,
    fontWeight: '500',
    marginTop: 10,
    textAlign: 'center',
  },
  form: {
    gap: 15,
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
  input: {
    backgroundColor: '#1a1a1a',
    borderWidth: 1,
    borderColor: '#2a2a2a',
    borderRadius: 4,
    padding: 12,
    color: '#f8fbfa',
    fontSize: 16,
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
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#6fc6a8',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    color: '#0f0f0f',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButtonText: {
    color: '#6fc6a8',
    fontSize: 16,
    fontWeight: '600',
  },
  backButton: {
    alignItems: 'center',
    marginTop: 10,
  },
  backButtonText: {
    color: '#d6dfdd',
    fontSize: 14,
  },
  // Success screen styles
  successInfo: {
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    padding: 15,
    marginVertical: 20,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  infoText: {
    color: '#d6dfdd',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  divider: {
    height: 1,
    backgroundColor: '#2a2a2a',
    marginVertical: 20,
  },
  resendSection: {
    alignItems: 'center',
    marginBottom: 20,
  },
  resendText: {
    color: '#d6dfdd',
    fontSize: 14,
    marginBottom: 5,
  },
  resendLink: {
    color: '#6fc6a8',
    fontSize: 16,
    fontWeight: '600',
  },
  spamNote: {
    color: '#95a5a6',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 20,
    fontStyle: 'italic',
  },
};