// C:\Users\s4303218\Documents\codebase\apnea-alert\apnea-mobile\src\screens\SignUpScreen.js
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
import { useAuth } from '../context/AuthContext';

export default function SignUpScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [emailStatus, setEmailStatus] = useState({
    checking: false,
    exists: false,
    message: ''
  });
  
  const { signUp, signIn } = useAuth();

  // Debounced email check
  useEffect(() => {
    const checkEmail = async () => {
      if (!email || email.length < 5 || !email.includes('@')) {
        setEmailStatus({ checking: false, exists: false, message: '' });
        return;
      }

      setEmailStatus(prev => ({ ...prev, checking: true }));

      try {
        // Check if email exists in profiles
        const { data: profiles, error } = await supabase
          .from('profiles')
          .select('id')
          .eq('email', email);

        if (error) {
          console.error('Error checking email:', error);
          setEmailStatus({ checking: false, exists: false, message: '' });
          return;
        }

        if (profiles && profiles.length > 0) {
          setEmailStatus({
            checking: false,
            exists: true,
            message: '⚠️ This email is already registered. Please sign in instead.'
          });
        } else {
          setEmailStatus({
            checking: false,
            exists: false,
            message: '✓ Email is available'
          });
        }
      } catch (error) {
        console.error('Error in email check:', error);
        setEmailStatus({ checking: false, exists: false, message: '' });
      }
    };

    const timer = setTimeout(checkEmail, 500);
    return () => clearTimeout(timer);
  }, [email]);

  const getEmailMessageStyle = () => {
    if (!emailStatus.message) return {};
    if (emailStatus.message.includes('✓')) {
      return { color: '#6fc6a8' };
    }
    return { color: '#e74c3c' };
  };

  const handleSignUp = async () => {
    if (!email || !password || !confirmPassword) {
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

    // If email exists (verified or unverified), don't allow signup
    if (emailStatus.exists) {
      Alert.alert(
        'Account Already Exists',
        'This email is already registered. Please sign in instead.',
        [
          {
            text: 'Go to Login',
            onPress: () => navigation.navigate('Login')
          },
          {
            text: 'Use Different Email',
            style: 'cancel',
            onPress: () => {
              setEmail('');
              setPassword('');
              setConfirmPassword('');
            }
          }
        ]
      );
      return;
    }

    // Email is available, proceed with signup
    setLoading(true);
    try {
      const { data, error } = await signUp(email, password);
      
      if (error) throw error;
      
      if (data?.user) {
        // Auto sign in after signup
        const { error: signInError } = await signIn(email, password);
        
        if (signInError) {
          console.error('Auto-signin failed:', signInError);
          Alert.alert('Error', 'Account created but auto-login failed. Please sign in manually.');
          navigation.replace('Login');
          return;
        }
        
        // Navigate directly to verification screen
        console.log('✅ Signup successful, navigating to verification');
        navigation.replace('Verification', { 
          userId: data.user.id,
          email: email 
        });
      }
    } catch (error) {
      Alert.alert('Sign Up Failed', error.message);
    } finally {
      setLoading(false);
    }
  };

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
            <View style={styles.header}>
              <Text style={styles.title}>Create Account</Text>
              <Text style={styles.subtitle}>
                Your journey to peaceful rest starts here.
              </Text>
            </View>

            <View style={styles.form}>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>Email Address *</Text>
                <TextInput
                  style={[
                    styles.input,
                    emailStatus.exists && styles.inputError
                  ]}
                  value={email}
                  onChangeText={setEmail}
                  placeholder="name@example.com"
                  placeholderTextColor="#666"
                  keyboardType="email-address"
                  autoCapitalize="none"
                  autoCorrect={false}
                  editable={!loading}
                />
                
                {/* Real-time email status message */}
                {emailStatus.message ? (
                  <View style={styles.emailStatusContainer}>
                    {emailStatus.checking ? (
                      <ActivityIndicator size="small" color="#6fc6a8" />
                    ) : (
                      <>
                        <Text style={[styles.emailStatusText, getEmailMessageStyle()]}>
                          {emailStatus.message}
                        </Text>
                        
                        {/* Quick sign in link for existing accounts */}
                        {emailStatus.exists && (
                          <TouchableOpacity 
                            onPress={() => navigation.navigate('Login')}
                            disabled={loading}
                          >
                            <Text style={styles.loginLink}>Sign In</Text>
                          </TouchableOpacity>
                        )}
                      </>
                    )}
                  </View>
                ) : null}
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.label}>Password *</Text>
                <View style={styles.passwordWrapper}>
                  <TextInput
                    style={[styles.input, styles.passwordInput]}
                    value={password}
                    onChangeText={setPassword}
                    placeholder="Create a secure password"
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
                <Text style={styles.label}>Confirm Password *</Text>
                <TextInput
                  style={styles.input}
                  value={confirmPassword}
                  onChangeText={setConfirmPassword}
                  placeholder="Re-enter your password"
                  placeholderTextColor="#666"
                  secureTextEntry={!showPassword}
                  editable={!loading}
                />
              </View>

              <TouchableOpacity
                style={[
                  styles.button, 
                  styles.primaryButton, 
                  (loading || emailStatus.exists) && styles.buttonDisabled
                ]}
                onPress={handleSignUp}
                disabled={loading || emailStatus.exists}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.buttonText}>Create Account</Text>
                )}
              </TouchableOpacity>

              <View style={styles.divider}>
                <View style={styles.dividerLine} />
                <Text style={styles.dividerText}>or</Text>
                <View style={styles.dividerLine} />
              </View>

              <TouchableOpacity
                style={[styles.button, styles.secondaryButton]}
                onPress={() => navigation.navigate('Login')}
                disabled={loading}
              >
                <Text style={styles.secondaryButtonText}>Sign In Instead</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.footer}>
              <Text style={styles.footerText}>
                By signing up, you agree to our Terms of Service and Privacy Policy.
              </Text>
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
  header: {
    marginBottom: 30,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
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
  inputError: {
    borderColor: '#e74c3c',
    borderWidth: 2,
  },
  emailStatusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 5,
    gap: 10,
  },
  emailStatusText: {
    fontSize: 12,
    flex: 1,
  },
  loginLink: {
    color: '#6fc6a8',
    fontSize: 12,
    fontWeight: '600',
  },
  hintText: {
    color: '#d6dfdd',
    fontSize: 12,
    marginTop: 5,
    opacity: 0.7,
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
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 15,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#2a2a2a',
  },
  dividerText: {
    color: '#d6dfdd',
    paddingHorizontal: 10,
    fontSize: 14,
  },
  footer: {
    marginTop: 20,
    alignItems: 'center',
  },
  footerText: {
    color: '#95a5a6',
    fontSize: 12,
    textAlign: 'center',
    fontStyle: 'italic',
  },
};