// C:\Users\s4303218\Documents\codebase\apnea-alert\apnea-mobile\src\screens\OnboardingScreen.js
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

export default function OnboardingScreen({ navigation }) {
  const [fullName, setFullName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [weight, setWeight] = useState('');
  const [height, setHeight] = useState('');
  const [loading, setLoading] = useState(false);
  const { updateProfile, completeOnboarding } = useAuth();

  const handleComplete = async () => {
    if (!fullName) {
      Alert.alert('Error', 'Please enter your name');
      return;
    }

    setLoading(true);
    try {
      // Update profile with user info
      const { error: updateError } = await updateProfile({
        full_name: fullName,
        age: age ? parseInt(age) : null,
        gender: gender || null,
        height_cm: height ? parseInt(height) : null,
        weight_kg: weight ? parseFloat(weight) : null,
      });

      if (updateError) throw updateError;

      // Mark onboarding as complete
      const { error: completeError } = await completeOnboarding();
      if (completeError) throw completeError;

      // Navigate to home screen
      navigation.replace('Home');
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to save profile');
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
              <Text style={styles.title}>Complete Your Profile</Text>
              <Text style={styles.subtitle}>
                This helps us provide more accurate sleep analysis
              </Text>
            </View>

            <View style={styles.form}>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>Full Name *</Text>
                <TextInput
                  style={styles.input}
                  value={fullName}
                  onChangeText={setFullName}
                  placeholder="Enter your full name"
                  placeholderTextColor="#666"
                />
              </View>

              <View style={styles.row}>
                <View style={[styles.inputGroup, styles.halfWidth]}>
                  <Text style={styles.label}>Age</Text>
                  <TextInput
                    style={styles.input}
                    value={age}
                    onChangeText={setAge}
                    placeholder="Years"
                    placeholderTextColor="#666"
                    keyboardType="numeric"
                  />
                </View>

                <View style={[styles.inputGroup, styles.halfWidth]}>
                  <Text style={styles.label}>Weight (kg)</Text>
                  <TextInput
                    style={styles.input}
                    value={weight}
                    onChangeText={setWeight}
                    placeholder="kg"
                    placeholderTextColor="#666"
                    keyboardType="numeric"
                  />
                </View>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.label}>Height (cm)</Text>
                <TextInput
                  style={styles.input}
                  value={height}
                  onChangeText={setHeight}
                  placeholder="cm"
                  placeholderTextColor="#666"
                  keyboardType="numeric"
                />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.label}>Gender</Text>
                <View style={styles.genderContainer}>
                  {['Male', 'Female', 'Other', 'Prefer not to say'].map((option) => (
                    <TouchableOpacity
                      key={option}
                      style={[
                        styles.genderOption,
                        gender === option && styles.genderOptionSelected,
                      ]}
                      onPress={() => setGender(option)}
                    >
                      <Text
                        style={[
                          styles.genderText,
                          gender === option && styles.genderTextSelected,
                        ]}
                      >
                        {option}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>

              <TouchableOpacity
                style={[styles.button, styles.primaryButton]}
                onPress={handleComplete}
                disabled={loading}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.buttonText}>Complete Setup</Text>
                )}
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => navigation.replace('Home')}
              >
                <Text style={styles.skipText}>Skip for now</Text>
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
  row: {
    flexDirection: 'row',
    gap: 10,
  },
  halfWidth: {
    flex: 1,
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
  genderContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  genderOption: {
    flex: 1,
    minWidth: '45%',
    padding: 12,
    borderWidth: 1,
    borderColor: '#2a2a2a',
    borderRadius: 4,
    alignItems: 'center',
  },
  genderOptionSelected: {
    borderColor: '#6fc6a8',
    backgroundColor: 'rgba(111, 198, 168, 0.1)',
  },
  genderText: {
    color: '#d6dfdd',
    fontSize: 14,
  },
  genderTextSelected: {
    color: '#6fc6a8',
  },
  button: {
    padding: 15,
    borderRadius: 4,
    alignItems: 'center',
    marginTop: 20,
  },
  primaryButton: {
    backgroundColor: '#6fc6a8',
  },
  buttonText: {
    color: '#0f0f0f',
    fontSize: 16,
    fontWeight: '600',
  },
  skipText: {
    color: '#6fc6a8',
    textAlign: 'center',
    marginTop: 15,
    fontSize: 14,
  },
};