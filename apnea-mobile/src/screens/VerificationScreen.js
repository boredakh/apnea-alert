// C:\Users\s4303218\Documents\codebase\apnea-alert\apnea-mobile\src\screens\VerificationScreen.js
import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Alert,
  ImageBackground,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  ActivityIndicator,
  Modal,
} from 'react-native';
import { useAuth } from '../context/AuthContext';

export default function VerificationScreen({ navigation, route }) {
  // Get userId and email from route params OR from the logged in user
  const { userId: routeUserId, email: routeEmail } = route.params || {};
  const { user, verifyCode, resendVerificationCode, signOut } = useAuth();
  
  // Use route params if available, otherwise use from user object
  const userId = routeUserId || user?.id;
  const email = routeEmail || user?.email;
  
  const [code, setCode] = useState(['', '', '', '', '', '']);
  const [loading, setLoading] = useState(false);
  const [resendLoading, setResendLoading] = useState(false);
  const [logoutLoading, setLogoutLoading] = useState(false);
  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [timer, setTimer] = useState(60);
  const [canResend, setCanResend] = useState(false);
  const inputRefs = useRef([]);

  // Timer for resend button
  useEffect(() => {
    let interval = null;
    if (timer > 0 && !canResend) {
      interval = setInterval(() => {
        setTimer((prevTimer) => prevTimer - 1);
      }, 1000);
    } else if (timer === 0) {
      setCanResend(true);
    }
    return () => clearInterval(interval);
  }, [timer, canResend]);

  // If no userId, something is wrong - logout
  useEffect(() => {
    if (!userId && !loading && !resendLoading && !logoutLoading) {
      console.log('❌ No userId found, redirecting to login');
      Alert.alert(
        'Error',
        'Unable to verify your account. Please sign in again.',
        [
          {
            text: 'OK',
            onPress: () => setShowLogoutModal(true)
          }
        ]
      );
    }
  }, [userId]);

  const handleCodeChange = (text, index) => {
    const newCode = [...code];
    newCode[index] = text;
    setCode(newCode);

    // Auto-focus next input
    if (text && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }
  };

  const handleKeyPress = (e, index) => {
    // Handle backspace
    if (e.nativeEvent.key === 'Backspace' && !code[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  const handleVerify = async () => {
    const verificationCode = code.join('');
    if (verificationCode.length !== 6) {
      Alert.alert('Error', 'Please enter the 6-digit verification code');
      return;
    }

    if (!userId) {
      Alert.alert('Error', 'User information missing. Please sign in again.');
      setShowLogoutModal(true);
      return;
    }

    console.log(`🚀 Attempting to verify code: ${verificationCode} for user: ${userId}`);
    setLoading(true);
    
    try {
      const { success, error } = await verifyCode(userId, verificationCode);
      
      console.log('Verification result:', { success, error });
      
      if (error) {
        if (error.includes('expired')) {
          Alert.alert(
            'Code Expired',
            'Your verification code has expired. Would you like to resend?',
            [
              {
                text: 'Resend Code',
                onPress: handleResendCode
              },
              {
                text: 'Cancel',
                style: 'cancel'
              }
            ]
          );
        } else {
          Alert.alert('Verification Failed', error);
        }
        setCode(['', '', '', '', '', '']);
        inputRefs.current[0]?.focus();
        return;
      }

      if (success) {
        // Show success message and navigate immediately
        Alert.alert(
          'Success!',
          'Your email has been verified.',
          [
            {
              text: 'Continue to Onboarding',
              onPress: () => {
                console.log('Navigating to Onboarding');
                navigation.replace('Onboarding');
              }
            }
          ]
        );
      }
    } catch (error) {
      console.error('Unexpected error:', error);
      Alert.alert('Error', 'An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleResendCode = async () => {
    if (!canResend) return;
    
    if (!userId) {
      Alert.alert('Error', 'User information missing. Please sign in again.');
      setShowLogoutModal(true);
      return;
    }

    setResendLoading(true);
    try {
      const { success, error } = await resendVerificationCode(userId);
      
      if (error) {
        Alert.alert('Error', error.message || 'Failed to resend code');
        return;
      }

      if (success) {
        Alert.alert(
          'Code Sent',
          'A new verification code has been sent to your email.'
        );
        setTimer(60);
        setCanResend(false);
        setCode(['', '', '', '', '', '']);
        inputRefs.current[0]?.focus();
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to resend code');
    } finally {
      setResendLoading(false);
    }
  };

  const performLogout = async () => {
    console.log('🚀 performLogout started');
    setLogoutLoading(true);
    setShowLogoutModal(false);
    
    try {
      console.log('📞 Calling signOut()...');
      await signOut();
      console.log('✅ signOut completed successfully');
      // Navigation will be handled by AppNavigator when user becomes null
    } catch (error) {
      console.error('❌ Logout error:', error);
      Alert.alert('Error', 'Failed to log out');
    } finally {
      console.log('🏁 Setting logoutLoading to false');
      setLogoutLoading(false);
    }
  };

  const handleBackToLogin = () => {
    console.log('🔙 Back to Login button pressed - showing modal');
    setShowLogoutModal(true);
  };

  return (
    <ImageBackground
      source={{ uri: 'https://images.pexels.com/photos/8306794/pexels-photo-8306794.jpeg?auto=compress&cs=tinysrgb&w=1500' }}
      style={styles.backgroundImage}
    >
      {/* Logout Confirmation Modal */}
      <Modal
        visible={showLogoutModal}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowLogoutModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Logout</Text>
            <Text style={styles.modalMessage}>Are you sure you want to logout?</Text>
            
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.cancelButton]}
                onPress={() => setShowLogoutModal(false)}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={[styles.modalButton, styles.confirmButton]}
                onPress={performLogout}
                disabled={logoutLoading}
              >
                {logoutLoading ? (
                  <ActivityIndicator size="small" color="#fff" />
                ) : (
                  <Text style={styles.confirmButtonText}>Logout</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.container}
      >
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          <View style={styles.modal}>
            <View style={styles.iconContainer}>
              <Text style={styles.icon}>✉️</Text>
            </View>

            <View style={styles.header}>
              <Text style={styles.title}>Verify Your Email</Text>
              <Text style={styles.subtitle}>
                We've sent a 6-digit verification code to
              </Text>
              <Text style={styles.email}>{email || 'your email'}</Text>
            </View>

            <View style={styles.codeContainer}>
              {code.map((digit, index) => (
                <TextInput
                  key={index}
                  ref={ref => inputRefs.current[index] = ref}
                  style={[
                    styles.codeInput,
                    digit ? styles.codeInputFilled : null
                  ]}
                  value={digit}
                  onChangeText={(text) => handleCodeChange(text, index)}
                  onKeyPress={(e) => handleKeyPress(e, index)}
                  keyboardType="numeric"
                  maxLength={1}
                  selectTextOnFocus
                  editable={!loading && !resendLoading && !logoutLoading}
                />
              ))}
            </View>

            {(loading || resendLoading || logoutLoading) && (
              <ActivityIndicator size="large" color="#6fc6a8" style={styles.loader} />
            )}

            <TouchableOpacity
              style={[styles.button, styles.primaryButton, (loading || resendLoading || logoutLoading) && styles.buttonDisabled]}
              onPress={handleVerify}
              disabled={loading || resendLoading || logoutLoading}
            >
              <Text style={styles.buttonText}>Verify Code</Text>
            </TouchableOpacity>

            <View style={styles.resendContainer}>
              <Text style={styles.resendText}>Didn't receive the code? </Text>
              <TouchableOpacity onPress={handleResendCode} disabled={!canResend || loading || resendLoading || logoutLoading}>
                <Text style={[
                  styles.resendLink,
                  (!canResend || loading || resendLoading || logoutLoading) && styles.resendLinkDisabled
                ]}>
                  Resend {!canResend && `(${timer}s)`}
                </Text>
              </TouchableOpacity>
            </View>

            <TouchableOpacity
              style={styles.backToLoginButton}
              onPress={handleBackToLogin}
              disabled={loading || resendLoading || logoutLoading}
            >
              <Text style={styles.backToLoginText}>← Back to Login</Text>
            </TouchableOpacity>
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
  },
  email: {
    color: '#6fc6a8',
    fontSize: 16,
    fontWeight: '500',
    marginTop: 5,
    textAlign: 'center',
  },
  codeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 30,
    gap: 8,
  },
  codeInput: {
    width: 45,
    height: 55,
    backgroundColor: '#1a1a1a',
    borderWidth: 1,
    borderColor: '#2a2a2a',
    borderRadius: 8,
    color: '#f8fbfa',
    fontSize: 24,
    textAlign: 'center',
    fontWeight: '600',
  },
  codeInputFilled: {
    borderColor: '#6fc6a8',
    backgroundColor: 'rgba(111, 198, 168, 0.1)',
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#0f0f0f',
    borderRadius: 10,
    padding: 20,
    width: '80%',
    maxWidth: 400,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
    color: '#f8fbfa',
  },
  modalMessage: {
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
    color: '#d6dfdd',
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  modalButton: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 5,
    minWidth: 100,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: '#2a2a2a',
  },
  confirmButton: {
    backgroundColor: '#e74c3c',
  },
  cancelButtonText: {
    color: '#f8fbfa',
    fontWeight: '600',
  },
  confirmButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  loader: {
    marginBottom: 20,
  },
  button: {
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 20,
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
  resendContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  resendText: {
    color: '#d6dfdd',
    fontSize: 14,
  },
  resendLink: {
    color: '#6fc6a8',
    fontSize: 14,
    fontWeight: '600',
  },
  resendLinkDisabled: {
    opacity: 0.5,
  },
  backToLoginButton: {
    alignItems: 'center',
    padding: 10,
  },
  backToLoginText: {
    color: '#d6dfdd',
    fontSize: 14,
  },
};