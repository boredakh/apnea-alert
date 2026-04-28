// C:\Users\s4303218\Documents\codebase\apnea-alert\apnea-mobile\src\screens\LoginScreen.js
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
import { useAuth } from '../context/AuthContext';

export default function LoginScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const { signIn, user, profile, loading: authLoading, getUserFlowState } = useAuth();

  // Handle navigation after successful login
  useEffect(() => {
    const handlePostLoginNavigation = async () => {
      // Only proceed if we have a user and auth is done loading
      if (!authLoading && user) {
        console.log('🔍 User detected, waiting for profile...');
        
        // Wait for profile to load (check every 500ms for up to 5 seconds)
        let attempts = 0;
        const maxAttempts = 10;
        
        const checkProfile = setInterval(async () => {
          attempts++;
          console.log(`⏳ Profile check attempt ${attempts}/${maxAttempts}`, { hasProfile: !!profile });
          
          if (profile) {
            clearInterval(checkProfile);
            console.log('✅ Profile loaded, determining flow state');
            
            try {
              const flowState = await getUserFlowState();
              console.log('📍 Flow state determined:', flowState);
              
              switch (flowState) {
                case 'verification':
                  console.log('➡️ Navigating to Verification');
                  navigation.replace('Verification', { userId: user.id, email: user.email });
                  break;
                case 'onboarding':
                  console.log('➡️ Navigating to Onboarding');
                  navigation.replace('Onboarding');
                  break;
                case 'home':
                  console.log('➡️ Navigating to Home');
                  navigation.replace('Home');
                  break;
                default:
                  console.log('➡️ Navigating to Home (default)');
                  navigation.replace('Home');
              }
            } catch (error) {
              console.error('❌ Error getting flow state:', error);
              navigation.replace('Home');
            }
          } else if (attempts >= maxAttempts) {
            clearInterval(checkProfile);
            console.log('⚠️ Profile load timeout - proceeding with user only');
            
            // Even without profile, try to determine flow state
            try {
              const flowState = await getUserFlowState();
              if (flowState === 'verification') {
                navigation.replace('Verification', { userId: user.id, email: user.email });
              } else {
                navigation.replace('Home');
              }
            } catch {
              navigation.replace('Home');
            }
          }
        }, 500);
      }
    };
    
    handlePostLoginNavigation();
  }, [user, profile, authLoading]);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    setLoading(true);
    try {
      const { data, error } = await signIn(email, password);
      
      if (error) {
        if (error.message.includes('Invalid login credentials')) {
          Alert.alert('Login Failed', 'Invalid email or password');
        } else {
          Alert.alert('Login Failed', error.message);
        }
        setLoading(false);
        return;
      }

      // Don't navigate here - let the useEffect handle it
      console.log('✅ Login successful, waiting for profile load');
      
    } catch (error) {
      Alert.alert('Login Failed', error.message || 'An unexpected error occurred');
      setLoading(false);
    }
  };

  const handleForgotPassword = () => {
    navigation.navigate('ForgotPassword');
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
              <Text style={styles.title}>Welcome Back</Text>
              <Text style={styles.subtitle}>
                Sign in to continue your journey to better rest
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

              <View style={styles.inputGroup}>
                <Text style={styles.label}>Password</Text>
                <View style={styles.passwordWrapper}>
                  <TextInput
                    style={[styles.input, styles.passwordInput]}
                    value={password}
                    onChangeText={setPassword}
                    placeholder="••••••••"
                    placeholderTextColor="#666"
                    secureTextEntry={!showPassword}
                    editable={!loading}
                  />
                  <TouchableOpacity
                    style={styles.passwordToggle}
                    onPress={() => setShowPassword(!showPassword)}
                    disabled={loading}
                  >
                    <Text style={styles.toggleText}>
                      {showPassword ? '👁️' : '👁️‍🗨️'}
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>

              <TouchableOpacity
                onPress={handleForgotPassword}
                disabled={loading}
              >
                <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.button, styles.primaryButton, loading && styles.buttonDisabled]}
                onPress={handleLogin}
                disabled={loading}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.buttonText}>Sign In</Text>
                )}
              </TouchableOpacity>

              <View style={styles.divider}>
                <View style={styles.dividerLine} />
                <Text style={styles.dividerText}>or</Text>
                <View style={styles.dividerLine} />
              </View>

              <TouchableOpacity
                style={[styles.button, styles.googleButton]}
                onPress={() => Alert.alert('Coming Soon', 'Google Sign-In coming soon')}
                disabled={loading}
              >
                <Text style={styles.googleButtonText}>Continue with Google</Text>
              </TouchableOpacity>

              <View style={styles.footer}>
                <TouchableOpacity onPress={() => navigation.navigate('SignUp')} disabled={loading}>
                  <Text style={styles.footerText}>
                    New to ApneaAlert? <Text style={styles.link}>Create an account</Text>
                  </Text>
                </TouchableOpacity>
              </View>
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
  forgotPasswordText: {
    color: '#6fc6a8',
    fontSize: 14,
    textAlign: 'right',
    marginBottom: 10,
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
  googleButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#6fc6a8',
  },
  googleButtonText: {
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
    color: '#d6dfdd',
    fontSize: 14,
  },
  link: {
    color: '#6fc6a8',
    fontWeight: '500',
  },
};