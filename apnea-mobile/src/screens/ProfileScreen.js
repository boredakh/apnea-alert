// src/screens/ProfileScreen.js
import { Platform } from 'react-native';
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  Modal,
} from 'react-native';
import { useAuth } from '../context/AuthContext';
import { supabase } from '../services/supabase';
import { fetchUserDataForExport, exportData } from '../services/exportService';
import Svg, { Path, Circle, Line, Rect } from 'react-native-svg';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export default function ProfileScreen({ navigation }) {
  const { user, signOut } = useAuth();
  const insets = useSafeAreaInsets();
  
  // Profile data states
  const [profile, setProfile] = useState({
    fullName: '',
    age: '',
    gender: 'prefer-not-to-say',
    height: '',
    weight: '',
  });
  
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [logoutLoading, setLogoutLoading] = useState(false);
  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  
  // Export states
  const [exporting, setExporting] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState('csv');
  const [dateRangeType, setDateRangeType] = useState('all');

  // Fetch actual profile data on mount
  useEffect(() => {
    fetchProfile();
  }, []);

  const getHeaderPadding = () => {
    if (Platform.OS === 'ios') {
      return insets.top + 10;
    }
    return insets.top + 10;
  };

  const fetchProfile = async () => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', user?.id)
        .single();

      if (error) throw error;
      
      if (data) {
        setProfile({
          fullName: data.full_name || '',
          age: data.age?.toString() || '',
          gender: data.gender || 'prefer-not-to-say',
          height: data.height_cm?.toString() || '',
          weight: data.weight_kg?.toString() || '',
        });
      }
    } catch (error) {
      console.error('Error fetching profile:', error);
    }
  };

  const handleEditToggle = () => {
    if (isEditing) {
      fetchProfile();
    }
    setIsEditing(!isEditing);
  };

  const handleSave = async () => {
    setLoading(true);
    
    try {
      const { error } = await supabase
        .from('profiles')
        .update({
          full_name: profile.fullName,
          age: parseInt(profile.age) || null,
          gender: profile.gender,
          height_cm: parseInt(profile.height) || null,
          weight_kg: parseInt(profile.weight) || null,
          updated_at: new Date(),
        })
        .eq('id', user?.id);

      if (error) throw error;

      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
      setIsEditing(false);
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to save changes');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    setExporting(true);
    setShowExportModal(false);
    
    try {
      let dateRangeQuery = null;
      let dateRangeText = 'All time';
      
      switch (dateRangeType) {
        case 'last7':
          const last7 = new Date();
          last7.setDate(last7.getDate() - 7);
          dateRangeQuery = { startDate: last7.toISOString().split('T')[0] };
          dateRangeText = 'Last 7 days';
          break;
        case 'last30':
          const last30 = new Date();
          last30.setDate(last30.getDate() - 30);
          dateRangeQuery = { startDate: last30.toISOString().split('T')[0] };
          dateRangeText = 'Last 30 days';
          break;
        default:
          dateRangeQuery = null;
          dateRangeText = 'All time';
      }
      
      const data = await fetchUserDataForExport(user.id, dateRangeQuery);
      
      if (data.length === 0) {
        Alert.alert('No Data', 'No sleep data found for the selected date range.');
        setExporting(false);
        return;
      }
      
      await exportData(selectedFormat, data, profile.fullName || user.email, dateRangeText);

      if (Platform.OS !== 'ios' && Platform.OS !== 'android') {
        Alert.alert('Success', `Report exported as ${selectedFormat.toUpperCase()}`);
      }
      
      if (Platform.OS === 'ios') {
        Alert.alert(
          'Export Complete',
          'Your file has been saved. You can find it in the Files app under "On My iPhone" > "ApneaAlert".\n\nTip: Look for the file in the "Recents" section or browse to "On My iPhone" > "ApneaAlert".'
        );
      } else if (Platform.OS === 'android') {
        Alert.alert('Success', `Report exported as ${selectedFormat.toUpperCase()}`);
      } else {
        Alert.alert('Success', `Report exported as ${selectedFormat.toUpperCase()}`);
      }
    } catch (error) {
      console.error('Export error:', error);
      Alert.alert('Export Failed', error.message);
    } finally {
      setExporting(false);
    }
  };

  const handleLogout = () => {
    setShowLogoutModal(true);
  };

  const performLogout = async () => {
    setLogoutLoading(true);
    setShowLogoutModal(false);
    
    try {
      await signOut();
    } catch (error) {
      console.error('Logout error:', error);
      Alert.alert('Error', 'Failed to log out');
    } finally {
      setLogoutLoading(false);
    }
  };

  const handleDeleteAccount = () => {
    Alert.alert(
      'Delete Account',
      'Are you absolutely sure? This will permanently delete all your data and cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            Alert.alert('Coming Soon', 'Account deletion will be available soon');
          },
        },
      ]
    );
  };

  // Icons
  const UserIcon = () => (
    <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <Path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" stroke="#6fc6a8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <Circle cx="12" cy="7" r="4" stroke="#6fc6a8" strokeWidth="2" />
    </Svg>
  );

  const CalendarIcon = () => (
    <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <Rect x="3" y="4" width="18" height="18" rx="2" ry="2" stroke="#6fc6a8" strokeWidth="2" />
      <Line x1="16" y1="2" x2="16" y2="6" stroke="#6fc6a8" strokeWidth="2" />
      <Line x1="8" y1="2" x2="8" y2="6" stroke="#6fc6a8" strokeWidth="2" />
      <Line x1="3" y1="10" x2="21" y2="10" stroke="#6fc6a8" strokeWidth="2" />
    </Svg>
  );

  const GenderIcon = () => (
    <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <Circle cx="12" cy="12" r="4" stroke="#6fc6a8" strokeWidth="2" />
      <Path d="M12 16v6m2-2h-4m8-18h4v4M2 2l7.17 7.17M2 5.355V2h3.357M22 2l-7.17 7.17M8 5L5 8" stroke="#6fc6a8" strokeWidth="2" strokeLinecap="round" />
    </Svg>
  );

  const RulerIcon = () => (
    <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <Path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0L21.3 15.3Z" stroke="#6fc6a8" strokeWidth="2" />
      <Path d="m14.5 12.5 2-2" stroke="#6fc6a8" strokeWidth="2" />
      <Path d="m11.5 9.5 2-2" stroke="#6fc6a8" strokeWidth="2" />
      <Path d="m8.5 6.5 2-2" stroke="#6fc6a8" strokeWidth="2" />
      <Path d="m17.5 15.5 2-2" stroke="#6fc6a8" strokeWidth="2" />
    </Svg>
  );

  const WeightIcon = () => (
    <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <Circle cx="12" cy="5" r="3" stroke="#6fc6a8" strokeWidth="2" />
      <Path d="M6.5 8a2 2 0 0 0-1.905 1.46L2.1 18.5A2 2 0 0 0 4 21h16a2 2 0 0 0 1.925-2.54L19.4 9.5A2 2 0 0 0 17.48 8Z" stroke="#6fc6a8" strokeWidth="2" />
    </Svg>
  );

  const EditIcon = () => (
    <Svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <Path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <Path d="M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z" stroke="currentColor" strokeWidth="2" />
    </Svg>
  );

  const CancelIcon = () => (
    <Svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <Path d="M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <Path d="M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </Svg>
  );

  const SaveIcon = () => (
    <Svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <Path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z" stroke="currentColor" strokeWidth="2" />
      <Path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7" stroke="currentColor" strokeWidth="2" />
      <Path d="M7 3v4a1 1 0 0 0 1 1h7" stroke="currentColor" strokeWidth="2" />
    </Svg>
  );

  const LogoutIcon = () => (
    <Svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <Path d="M14 8V6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2v-2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <Path d="M9 12h12l-3-3m0 6l3-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </Svg>
  );

  const DeleteIcon = () => (
    <Svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <Path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <Line x1="10" y1="11" x2="10" y2="17" stroke="currentColor" strokeWidth="2" />
      <Line x1="14" y1="11" x2="14" y2="17" stroke="currentColor" strokeWidth="2" />
    </Svg>
  );

  const ExportIcon = () => (
    <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <Path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2M7 11l5 5 5-5M12 4v12" stroke="#6fc6a8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );

  return (
    <ScrollView 
      style={styles.container} 
      contentContainerStyle={[
        styles.contentContainer,
        { paddingTop: getHeaderPadding() }
      ]}
    >
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Profile Settings</Text>
        <Text style={styles.headerSubtitle}>
          Manage your personal health data to optimize your sleep insights.
        </Text>
      </View>

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

      {/* Export Modal */}
      <Modal
        visible={showExportModal}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowExportModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Export Sleep Data</Text>
            
            <Text style={styles.modalLabel}>Format</Text>
            <View style={styles.formatOptions}>
              {['txt', 'csv', 'json', 'html'].map(format => (
                <TouchableOpacity
                  key={format}
                  style={[
                    styles.formatOption,
                    selectedFormat === format && styles.formatOptionSelected,
                  ]}
                  onPress={() => setSelectedFormat(format)}
                >
                  <Text style={[
                    styles.formatOptionText,
                    selectedFormat === format && styles.formatOptionTextSelected,
                  ]}>
                    {format === 'txt' ? '📝 TEXT' : format.toUpperCase()}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
            
            <Text style={styles.modalLabel}>Date Range</Text>
            <View style={styles.dateRangeOptions}>
              {[
                { key: 'all', label: 'All Time' },
                { key: 'last7', label: 'Last 7 Days' },
                { key: 'last30', label: 'Last 30 Days' },
              ].map(option => (
                <TouchableOpacity
                  key={option.key}
                  style={[
                    styles.dateOption,
                    dateRangeType === option.key && styles.dateOptionSelected,
                  ]}
                  onPress={() => setDateRangeType(option.key)}
                >
                  <Text style={[
                    styles.dateOptionText,
                    dateRangeType === option.key && styles.dateOptionTextSelected,
                  ]}>{option.label}</Text>
                </TouchableOpacity>
              ))}
            </View>
            
            <TouchableOpacity
              style={styles.exportModalButton}
              onPress={handleExport}
              disabled={exporting}
            >
              {exporting ? (
                <ActivityIndicator size="small" color="#0f0f0f" />
              ) : (
                <Text style={styles.exportModalButtonText}>Save & Share</Text>
              )}
            </TouchableOpacity>
            
            <TouchableOpacity
              style={styles.cancelModalButton}
              onPress={() => setShowExportModal(false)}
            >
              <Text style={styles.cancelModalButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Profile Form Card */}
      <View style={styles.card}>
        <View style={styles.formGrid}>
          <View style={styles.field}>
            <View style={styles.labelContainer}>
              <UserIcon />
              <Text style={styles.label}>Full Name</Text>
            </View>
            <TextInput
              style={[styles.input, !isEditing && styles.inputDisabled]}
              value={profile.fullName}
              onChangeText={(text) => setProfile({ ...profile, fullName: text })}
              editable={isEditing}
              placeholder="Enter your full name"
              placeholderTextColor="#666"
            />
          </View>

          <View style={styles.field}>
            <View style={styles.labelContainer}>
              <CalendarIcon />
              <Text style={styles.label}>Age</Text>
            </View>
            <TextInput
              style={[styles.input, !isEditing && styles.inputDisabled]}
              value={profile.age}
              onChangeText={(text) => setProfile({ ...profile, age: text })}
              editable={isEditing}
              keyboardType="numeric"
              placeholder="Enter your age"
              placeholderTextColor="#666"
            />
          </View>

          <View style={styles.field}>
            <View style={styles.labelContainer}>
              <GenderIcon />
              <Text style={styles.label}>Gender</Text>
            </View>
            <View style={styles.pickerContainer}>
              {['male', 'female', 'prefer-not-to-say'].map(option => (
                <TouchableOpacity
                  key={option}
                  style={[
                    styles.genderOption,
                    profile.gender === option && styles.genderOptionSelected,
                    !isEditing && styles.genderOptionDisabled,
                  ]}
                  onPress={() => isEditing && setProfile({ ...profile, gender: option })}
                  disabled={!isEditing}
                >
                  <Text style={[
                    styles.genderOptionText,
                    profile.gender === option && styles.genderOptionTextSelected,
                  ]}>{option === 'prefer-not-to-say' ? 'Prefer not to say' : option.charAt(0).toUpperCase() + option.slice(1)}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          <View style={styles.field}>
            <View style={styles.labelContainer}>
              <RulerIcon />
              <Text style={styles.label}>Height (cm)</Text>
            </View>
            <TextInput
              style={[styles.input, !isEditing && styles.inputDisabled]}
              value={profile.height}
              onChangeText={(text) => setProfile({ ...profile, height: text })}
              editable={isEditing}
              keyboardType="numeric"
              placeholder="Enter your height"
              placeholderTextColor="#666"
            />
          </View>

          <View style={styles.field}>
            <View style={styles.labelContainer}>
              <WeightIcon />
              <Text style={styles.label}>Weight (kg)</Text>
            </View>
            <TextInput
              style={[styles.input, !isEditing && styles.inputDisabled]}
              value={profile.weight}
              onChangeText={(text) => setProfile({ ...profile, weight: text })}
              editable={isEditing}
              keyboardType="numeric"
              placeholder="Enter your weight"
              placeholderTextColor="#666"
            />
          </View>
        </View>

        {saveSuccess && (
          <View style={styles.successMessage}>
            <Text style={styles.successMessageText}>Changes saved successfully!</Text>
          </View>
        )}

        <View style={styles.actionButtons}>
          <TouchableOpacity
            style={[
              styles.button,
              isEditing ? styles.buttonSecondary : styles.buttonOutline,
            ]}
            onPress={handleEditToggle}
            disabled={logoutLoading}
          >
            {isEditing ? <CancelIcon /> : <EditIcon />}
            <Text style={[
              styles.buttonText,
              isEditing ? styles.buttonTextSecondary : styles.buttonTextOutline,
            ]}>
              {isEditing ? 'Cancel Editing' : 'Enable Editing'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.button,
              styles.buttonPrimary,
              (!isEditing || loading || logoutLoading) && styles.buttonDisabled,
            ]}
            onPress={handleSave}
            disabled={!isEditing || loading || logoutLoading}
          >
            {loading ? (
              <ActivityIndicator size="small" color="#0f0f0f" />
            ) : (
              <>
                <SaveIcon />
                <Text style={styles.buttonTextPrimary}>Save Changes</Text>
              </>
            )}
          </TouchableOpacity>
        </View>

        {/* Export Section */}
        <View style={styles.exportSection}>
          <Text style={styles.sectionTitle}>📊 Data Export</Text>
          <Text style={styles.sectionDescription}>
            Export your sleep data for personal records or to share with your healthcare provider.
          </Text>
          
          <View style={styles.formatInfoContainer}>
            <Text style={styles.formatInfoTitle}>Available Formats:</Text>
            <View style={styles.formatInfoList}>
              <Text style={styles.formatInfoItem}>📝 • TEXT: Human-readable format with emojis and clear summaries (Best for iOS/mobile)</Text>
              <Text style={styles.formatInfoItem}>📊 • CSV: Spreadsheet compatible for data analysis</Text>
              <Text style={styles.formatInfoItem}>🔧 • JSON: Raw data format for developers</Text>
              <Text style={styles.formatInfoItem}>🌐 • HTML: Interactive web report</Text>
            </View>
          </View>
          
          <TouchableOpacity
            style={styles.exportButton}
            onPress={() => setShowExportModal(true)}
            disabled={exporting}
          >
            {exporting ? (
              <ActivityIndicator size="small" color="#0f0f0f" />
            ) : (
              <>
                <ExportIcon />
                <Text style={styles.exportButtonText}>Export Sleep Data</Text>
              </>
            )}
          </TouchableOpacity>
          <Text style={styles.exportNote}>
            💡 TEXT format is human-readable with emojis and summaries. Other formats keep their original structure.
          </Text>
        </View>

        <TouchableOpacity
          style={styles.logoutButton}
          onPress={handleLogout}
          disabled={logoutLoading}
        >
          {logoutLoading ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <>
              <LogoutIcon />
              <Text style={styles.logoutButtonText}>Log Out</Text>
            </>
          )}
        </TouchableOpacity>

        <View style={styles.dangerZone}>
          <Text style={styles.dangerZoneTitle}>⚠️ Danger Zone</Text>
          <TouchableOpacity
            style={styles.dangerButton}
            onPress={handleDeleteAccount}
            disabled={logoutLoading}
          >
            <DeleteIcon />
            <Text style={styles.dangerButtonText}>
              Delete Account and permanently delete data
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f0f',
  },
  contentContainer: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 24,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#f8fbfa',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#95a5a6',
    lineHeight: 20,
  },
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  formGrid: {
    gap: 20,
  },
  field: {
    gap: 8,
  },
  labelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
    color: '#f8fbfa',
  },
  input: {
    backgroundColor: '#0f0f0f',
    borderWidth: 1,
    borderColor: '#2a2a2a',
    borderRadius: 8,
    padding: 12,
    color: '#f8fbfa',
    fontSize: 16,
  },
  inputDisabled: {
    backgroundColor: '#1a1a1a',
    color: '#95a5a6',
    opacity: 0.7,
  },
  pickerContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  genderOption: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#2a2a2a',
    backgroundColor: '#0f0f0f',
  },
  genderOptionSelected: {
    borderColor: '#6fc6a8',
    backgroundColor: 'rgba(111, 198, 168, 0.1)',
  },
  genderOptionDisabled: {
    opacity: 0.5,
  },
  genderOptionText: {
    color: '#95a5a6',
    fontSize: 14,
  },
  genderOptionTextSelected: {
    color: '#6fc6a8',
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 24,
    marginBottom: 12,
  },
  button: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    borderWidth: 1,
  },
  buttonPrimary: {
    backgroundColor: '#6fc6a8',
    borderColor: '#6fc6a8',
  },
  buttonOutline: {
    backgroundColor: 'transparent',
    borderColor: '#6fc6a8',
  },
  buttonSecondary: {
    backgroundColor: '#2a2a2a',
    borderColor: '#2a2a2a',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  buttonTextPrimary: {
    color: '#0f0f0f',
  },
  buttonTextOutline: {
    color: '#6fc6a8',
  },
  buttonTextSecondary: {
    color: '#f8fbfa',
  },
  exportSection: {
    marginTop: 16,
    marginBottom: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#f8fbfa',
    marginBottom: 8,
  },
  sectionDescription: {
    fontSize: 14,
    color: '#95a5a6',
    marginBottom: 16,
    lineHeight: 20,
  },
  exportButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#6fc6a8',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 10,
    gap: 10,
  },
  exportButtonText: {
    color: '#0f0f0f',
    fontSize: 16,
    fontWeight: '600',
  },
  exportNote: {
    fontSize: 12,
    color: '#95a5a6',
    marginTop: 12,
    textAlign: 'center',
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#e74c3c',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 8,
    gap: 8,
    marginTop: 8,
    marginBottom: 16,
  },
  logoutButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 24,
    width: '85%',
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
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
  modalLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#f8fbfa',
    marginBottom: 10,
    marginTop: 10,
  },
  formatOptions: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  formatOption: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    backgroundColor: '#0f0f0f',
    borderWidth: 1,
    borderColor: '#2a2a2a',
    alignItems: 'center',
  },
  formatOptionSelected: {
    borderColor: '#6fc6a8',
    backgroundColor: 'rgba(111, 198, 168, 0.1)',
  },
  formatOptionText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#95a5a6',
  },
  formatOptionTextSelected: {
    color: '#6fc6a8',
  },
  dateRangeOptions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 20,
  },
  dateOption: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    backgroundColor: '#0f0f0f',
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  dateOptionSelected: {
    borderColor: '#6fc6a8',
    backgroundColor: 'rgba(111, 198, 168, 0.1)',
  },
  dateOptionText: {
    fontSize: 14,
    color: '#95a5a6',
  },
  dateOptionTextSelected: {
    color: '#6fc6a8',
  },
  exportModalButton: {
    backgroundColor: '#6fc6a8',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 10,
  },
  exportModalButtonText: {
    color: '#0f0f0f',
    fontSize: 16,
    fontWeight: '600',
  },
  cancelModalButton: {
    alignItems: 'center',
    paddingVertical: 12,
    marginTop: 8,
  },
  cancelModalButtonText: {
    color: '#95a5a6',
    fontSize: 14,
  },
  successMessage: {
    marginTop: 16,
    padding: 12,
    backgroundColor: 'rgba(46, 204, 113, 0.1)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2ecc71',
  },
  successMessageText: {
    color: '#2ecc71',
    fontSize: 14,
    textAlign: 'center',
  },
  dangerZone: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  dangerZoneTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ff4d4d',
    marginBottom: 16,
  },
  dangerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 16,
    backgroundColor: 'rgba(255, 77, 77, 0.1)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ff4d4d',
  },
  dangerButtonText: {
    flex: 1,
    color: '#ff4d4d',
    fontSize: 14,
    fontWeight: '500',
  },
  formatInfoContainer: {
    marginBottom: 16,
    padding: 12,
    backgroundColor: 'rgba(111, 198, 168, 0.05)',
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#6fc6a8',
  },
  formatInfoTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6fc6a8',
    marginBottom: 8,
  },
  formatInfoList: {
    gap: 6,
  },
  formatInfoItem: {
    fontSize: 12,
    color: '#95a5a6',
    lineHeight: 18,
  },
});