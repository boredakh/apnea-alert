// src/screens/HistoryScreen.js
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  Animated,
  Modal,
  Platform,
  SafeAreaView,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { useAuth } from '../context/AuthContext';
import { supabase } from '../services/supabase';
import Svg, { Path, Circle } from 'react-native-svg';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export default function HistoryScreen({ navigation }) {
  const { user } = useAuth();
  const insets = useSafeAreaInsets();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);
  const [reportModalVisible, setReportModalVisible] = useState(false);
  const [animatedValues, setAnimatedValues] = useState({});
  const [filter, setFilter] = useState('all');

  // Auto-refresh when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      if (user) {
        fetchHistory(true);
      }
      return () => {};
    }, [user])
  );

  // Scaling function for apnea probability
  const scaleProbability = (rawProbability) => {
    if (!rawProbability && rawProbability !== 0) return 0;
    const rawPercent = rawProbability * 100;
    
    let scaledPercent;
    if (rawPercent < 50) {
      scaledPercent = (rawPercent / 50) * 10;
    } else if (rawPercent < 63) {
      scaledPercent = 10 + ((rawPercent - 50) / 12) * 40;
    } else {
      scaledPercent = 50 + ((rawPercent - 63) / 37) * 50;
    }
    
    const finalPercent = Math.min(scaledPercent, 100);
    return finalPercent / 100;
  };

  const getRiskLevel = (probability) => {
    const percent = probability * 100;
    if (percent > 70) return 'High';
    if (percent > 30) return 'Moderate';
    return 'Low';
  };

  const fetchHistory = async (silent = false) => {
    if (!user) return;
    
    if (!silent) {
      setLoading(true);
    }
    
    try {
      const { data, error } = await supabase
        .from('predictions')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      
      setItems(data || []);
      
      const newAnimatedValues = {};
      (data || []).forEach(item => {
        if (!animatedValues[item.id]) {
          newAnimatedValues[item.id] = new Animated.Value(0);
        } else {
          newAnimatedValues[item.id] = animatedValues[item.id];
        }
      });
      setAnimatedValues(newAnimatedValues);
    } catch (error) {
      console.error('Error fetching history:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchHistory(false);
  };

  const toggleExpand = (id) => {
    const isExpanding = expandedId !== id;
    
    if (isExpanding) {
      setExpandedId(id);
      Animated.timing(animatedValues[id], {
        toValue: 1,
        duration: 300,
        useNativeDriver: false,
      }).start();
    } else {
      Animated.timing(animatedValues[id], {
        toValue: 0,
        duration: 300,
        useNativeDriver: false,
      }).start(() => {
        setExpandedId(null);
      });
    }
  };

  const openFullReport = (item) => {
    setSelectedReport(item);
    setReportModalVisible(true);
  };

  const getRiskColor = (riskLevel) => {
    switch(riskLevel) {
      case 'High': return '#e74c3c';
      case 'Moderate': return '#f39c12';
      case 'Low': return '#27ae60';
      default: return '#6fc6a8';
    }
  };

  const getRiskBgColor = (riskLevel) => {
    switch(riskLevel) {
      case 'High': return 'rgba(231, 76, 60, 0.15)';
      case 'Moderate': return 'rgba(243, 156, 18, 0.15)';
      case 'Low': return 'rgba(39, 174, 96, 0.15)';
      default: return 'rgba(111, 198, 168, 0.15)';
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-GB', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    });
  };

  const formatTime = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString('en-GB', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const parseMetadata = (metadata) => {
    if (!metadata) return null;
    if (typeof metadata === 'string') {
      try {
        return JSON.parse(metadata);
      } catch {
        return null;
      }
    }
    return metadata;
  };

  const parseFeatures = (features) => {
    if (!features) return null;
    if (typeof features === 'string') {
      try {
        return JSON.parse(features);
      } catch {
        return null;
      }
    }
    return features;
  };

  const isSleepAnalysis = (item) => {
    return item.source === 'sleep_ai_analysis' || item.source === 'sleep_ai_analysis_v2';
  };

  const getRecoveryColor = (score) => {
    if (score >= 70) return '#2ecc71';
    if (score >= 50) return '#f39c12';
    return '#e74c3c';
  };

  // Icons
  const SleepIcon = () => (
    <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <Path
        d="M12 2a10 10 0 0 0-10 10 10 10 0 0 0 10 10 10 10 0 0 0 10-10c0-1.5-.3-2.9-.9-4.2A8 8 0 0 1 12 4a8 8 0 0 1-6.6 3.4"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </Svg>
  );

  const RiskIcon = ({ riskLevel }) => {
    const color = getRiskColor(riskLevel);
    return (
      <Svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <Path
          d="M12 8v4m0 4h.01M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </Svg>
    );
  };

  const FilterButton = ({ title, active, onPress }) => (
    <TouchableOpacity
      style={[styles.filterButton, active && styles.filterButtonActive]}
      onPress={onPress}
    >
      <Text style={[styles.filterButtonText, active && styles.filterButtonTextActive]}>
        {title}
      </Text>
    </TouchableOpacity>
  );

  // Updated renderSleepAnalysisCard to match your data structure
  const renderSleepAnalysisCard = (item) => {
    const features = parseFeatures(item.features);
    const metadata = parseMetadata(item.metadata);
    
    // Get data from features
    const recoveryScore = features?.recovery_score || metadata?.recoveryScore || 50;
    const durationHours = features?.duration_hours || '0';
    const qualityScore = features?.quality_score || 0;
    const deepPercent = features?.deep_percent || 0;
    const recoveryColor = getRecoveryColor(recoveryScore);
    
    // Get summary from metadata
    const summary = metadata?.summary || 'Tap to view full sleep report';
    
    return (
      <TouchableOpacity
        style={[styles.card, { borderLeftColor: recoveryColor, borderLeftWidth: 4 }]}
        onPress={() => openFullReport(item)}
        activeOpacity={0.7}
      >
        <View style={styles.cardHeader}>
          <View style={[styles.typeBadge, { backgroundColor: 'rgba(111, 198, 168, 0.15)' }]}>
            <SleepIcon />
            <Text style={[styles.typeText, { color: '#6fc6a8' }]}>
              Sleep Analysis
            </Text>
          </View>
          <Text style={styles.cardDate}>{formatDate(item.created_at)}</Text>
        </View>

        <Text style={styles.predictionLabel}>
          {item.prediction_label || 'Sleep Quality Report'}
        </Text>

        <View style={styles.recoveryContainer}>
          <View style={[styles.recoveryCircle, { borderColor: recoveryColor }]}>
            <Text style={[styles.recoveryScore, { color: recoveryColor }]}>
              {recoveryScore}
            </Text>
            <Text style={styles.recoveryLabel}>Recovery</Text>
          </View>
          <View style={styles.recoveryStats}>
            <View style={styles.recoveryStat}>
              <Text style={styles.recoveryStatValue}>{durationHours}h</Text>
              <Text style={styles.recoveryStatLabel}>Sleep</Text>
            </View>
            <View style={styles.recoveryStat}>
              <Text style={styles.recoveryStatValue}>{qualityScore}%</Text>
              <Text style={styles.recoveryStatLabel}>Quality</Text>
            </View>
            <View style={styles.recoveryStat}>
              <Text style={styles.recoveryStatValue}>{deepPercent}%</Text>
              <Text style={styles.recoveryStatLabel}>Deep</Text>
            </View>
          </View>
        </View>

        <Text style={styles.previewText} numberOfLines={2}>
          {summary}
        </Text>
      </TouchableOpacity>
    );
  };

  const renderApneaCard = (item) => {
    const displayProbability = scaleProbability(item.apnea_probability);
    const displayRiskLevel = getRiskLevel(displayProbability);
    const riskColor = getRiskColor(displayRiskLevel);
    const riskBgColor = getRiskBgColor(displayRiskLevel);
    const features = parseFeatures(item.features);
    
    const isExpanded = expandedId === item.id;
    const animation = animatedValues[item.id] || new Animated.Value(0);
    
    return (
      <TouchableOpacity
        style={[styles.card, { borderLeftColor: riskColor, borderLeftWidth: 4 }]}
        onPress={() => toggleExpand(item.id)}
        activeOpacity={0.7}
      >
        <View style={styles.cardHeader}>
          <View style={[styles.typeBadge, { backgroundColor: riskBgColor }]}>
            <RiskIcon riskLevel={displayRiskLevel} />
            <Text style={[styles.typeText, { color: riskColor }]}>
              Apnea Detection
            </Text>
          </View>
          <Text style={styles.cardDate}>{formatDate(item.created_at)}</Text>
        </View>

        <Text style={styles.predictionLabel}>
          {displayRiskLevel} Risk - {item.prediction_label || 'Analysis'}
        </Text>

        <View style={styles.apneaStats}>
          <View style={styles.apneaStat}>
            <Text style={[styles.apneaStatValue, { color: riskColor }]}>
              {(displayProbability * 100).toFixed(1)}%
            </Text>
            <Text style={styles.apneaStatLabel}>Risk Probability</Text>
          </View>
          <View style={styles.apneaStat}>
            <Text style={styles.apneaStatValue}>
              {item.fitbit_date ? formatDate(item.fitbit_date) : formatDate(item.created_at)}
            </Text>
            <Text style={styles.apneaStatLabel}>Data From</Text>
          </View>
        </View>

        {features && (
          <View style={styles.featurePreview}>
            <Text style={styles.featurePreviewText}>
              💓 HR: {features.hr_mean?.toFixed(0) || 'N/A'} • 
              🧬 LF/HF: {features.lf_hf_ratio?.toFixed(1) || 'N/A'}
            </Text>
          </View>
        )}

        {isExpanded && (
          <Animated.View style={[styles.expandedContent, { opacity: animation }]}>
            <View style={styles.expandedSection}>
              <Text style={styles.expandedTitle}>📊 Key Metrics</Text>
              <Text style={styles.expandedText}>• Heart Rate Variability: {features?.hrv_rmssd?.toFixed(1) || 'N/A'} ms</Text>
              <Text style={styles.expandedText}>• Resting Heart Rate: {features?.resting_hr?.toFixed(0) || 'N/A'} bpm</Text>
              <Text style={styles.expandedText}>• Sleep Duration: {features?.sleep_hours?.toFixed(1) || 'N/A'} hours</Text>
              <Text style={styles.expandedText}>• Apnea-Hypopnea Index: {features?.ahi_estimate?.toFixed(1) || 'N/A'} events/hour</Text>
            </View>
            
            <View style={styles.expandedSection}>
              <Text style={styles.expandedTitle}>💡 Recommendations</Text>
              <Text style={styles.expandedText}>
                {displayRiskLevel === 'High' 
                  ? '⚠️ Please consult a healthcare provider for proper diagnosis and treatment options.'
                  : displayRiskLevel === 'Moderate'
                  ? '🔄 Consider lifestyle changes and monitor symptoms. Follow up with your doctor.'
                  : '✅ Maintain healthy sleep habits. Continue monitoring your sleep quality.'}
              </Text>
            </View>

            <TouchableOpacity 
              style={styles.fullReportButton}
              onPress={() => openFullReport(item)}
            >
              <Text style={styles.fullReportButtonText}>View Full Analysis →</Text>
            </TouchableOpacity>
          </Animated.View>
        )}
      </TouchableOpacity>
    );
  };

  const renderItem = (item) => {
    if (isSleepAnalysis(item)) {
      return renderSleepAnalysisCard(item);
    } else {
      return renderApneaCard(item);
    }
  };

  const filteredItems = items.filter(item => {
    if (filter === 'all') return true;
    if (filter === 'apnea') return !isSleepAnalysis(item);
    if (filter === 'sleep') return isSleepAnalysis(item);
    return true;
  });

  // Updated Full Report Modal to match your data structure
  const FullReportModal = () => {
    if (!selectedReport) return null;
    
    const metadata = parseMetadata(selectedReport.metadata);
    const features = parseFeatures(selectedReport.features);
    const isSleep = isSleepAnalysis(selectedReport);
    
    if (!isSleep) {
      const displayProbability = scaleProbability(selectedReport.apnea_probability);
      const displayRiskLevel = getRiskLevel(displayProbability);
      const riskColor = getRiskColor(displayRiskLevel);
      
      return (
        <Modal
          visible={reportModalVisible}
          animationType="slide"
          transparent={false}
          onRequestClose={() => setReportModalVisible(false)}
        >
          <SafeAreaView style={styles.modalSafeArea} edges={['top', 'bottom']}>
            <ScrollView 
              style={styles.modalContainer} 
              contentContainerStyle={styles.modalContent}
              showsVerticalScrollIndicator={false}
            >
              <View style={styles.modalHeader}>
                <TouchableOpacity 
                  style={styles.modalBackButton}
                  onPress={() => setReportModalVisible(false)}
                  hitSlop={{ top: 20, bottom: 20, left: 20, right: 20 }}
                >
                  <Text style={styles.modalClose}>← Back</Text>
                </TouchableOpacity>
                <Text style={styles.modalTitle}>Apnea Detection Report</Text>
                <View style={{ width: 50 }} />
              </View>

              <Text style={styles.modalDate}>{formatDate(selectedReport.created_at)}</Text>

              <View style={[styles.modalScoreCard, { borderColor: riskColor }]}>
                <View style={[styles.modalScoreCircle, { borderColor: riskColor }]}>
                  <Text style={[styles.modalScoreValue, { color: riskColor }]}>
                    {(displayProbability * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.modalScoreLabel}>Risk Probability</Text>
                </View>
                <View style={styles.riskBadgeContainer}>
                  <View style={[styles.riskBadge, { backgroundColor: getRiskBgColor(displayRiskLevel) }]}>
                    <Text style={[styles.riskBadgeText, { color: riskColor }]}>
                      {displayRiskLevel} Risk
                    </Text>
                  </View>
                </View>
              </View>

              {features && (
                <View style={styles.modalSection}>
                  <Text style={styles.modalSectionTitle}>📊 Detailed Metrics</Text>
                  <View style={styles.metricsGrid}>
                    <View style={styles.metricCard}>
                      <Text style={styles.metricValue}>{features.hrv_rmssd?.toFixed(1) || 'N/A'}</Text>
                      <Text style={styles.metricLabel}>HRV (ms)</Text>
                    </View>
                    <View style={styles.metricCard}>
                      <Text style={styles.metricValue}>{features.resting_hr?.toFixed(0) || 'N/A'}</Text>
                      <Text style={styles.metricLabel}>Resting HR</Text>
                    </View>
                    <View style={styles.metricCard}>
                      <Text style={styles.metricValue}>{features.sleep_hours?.toFixed(1) || 'N/A'}</Text>
                      <Text style={styles.metricLabel}>Sleep Hours</Text>
                    </View>
                    <View style={styles.metricCard}>
                      <Text style={styles.metricValue}>{features.ahi_estimate?.toFixed(1) || 'N/A'}</Text>
                      <Text style={styles.metricLabel}>AHI Estimate</Text>
                    </View>
                  </View>
                </View>
              )}

              <View style={styles.modalSection}>
                <Text style={styles.modalSectionTitle}>💡 Understanding Your Results</Text>
                <Text style={styles.modalTextLight}>
                  {displayRiskLevel === 'High' 
                    ? 'Your results indicate a high probability of sleep apnea. This means you show strong patterns consistent with sleep-disordered breathing. We strongly recommend consulting with a sleep specialist for a proper diagnosis and treatment options.'
                    : displayRiskLevel === 'Moderate'
                    ? 'Your results show moderate indicators of sleep apnea. While not definitive, there are patterns that warrant further investigation. Consider discussing these results with your healthcare provider and monitoring for symptoms like loud snoring, daytime fatigue, or morning headaches.'
                    : 'Your results show low indicators of sleep apnea. Your sleep patterns appear consistent with normal breathing during sleep. Continue maintaining healthy sleep habits and monitor any symptoms that may develop.'}
                </Text>
              </View>

              <View style={styles.modalDisclaimer}>
                <Text style={styles.modalDisclaimerText}>
                  ⚠️ This is an AI-powered screening tool and not a medical diagnosis. Only a healthcare professional can properly diagnose sleep apnea through clinical evaluation and sleep studies.
                </Text>
              </View>
            </ScrollView>
          </SafeAreaView>
        </Modal>
      );
    }
    
    // Sleep Analysis Modal - Updated to match your data structure
    const recoveryScore = features?.recovery_score || metadata?.recoveryScore || 50;
    const circadianScore = features?.circadian_score || metadata?.circadianScore || 50;
    const sleepDebt = features?.sleep_debt || metadata?.sleepDebt || 0;
    const recoveryColor = getRecoveryColor(recoveryScore);
    
    // Get the data from metadata
    const summary = metadata?.summary || 'No summary available';
    const insights = metadata?.insights || [];
    const recommendations = metadata?.recommendations || [];
    const positiveNote = metadata?.positiveNote || '';
    const riskAssessment = metadata?.riskAssessment || '';
    const trends = metadata?.trends || {};
    
    return (
      <Modal
        visible={reportModalVisible}
        animationType="slide"
        transparent={false}
        onRequestClose={() => setReportModalVisible(false)}
      >
        <SafeAreaView style={styles.modalSafeArea} edges={['top', 'bottom']}>
          <ScrollView 
            style={styles.modalContainer} 
            contentContainerStyle={styles.modalContent}
            showsVerticalScrollIndicator={false}
          >
            <View style={styles.modalHeader}>
              <TouchableOpacity 
                style={styles.modalBackButton}
                onPress={() => setReportModalVisible(false)}
                hitSlop={{ top: 20, bottom: 20, left: 20, right: 20 }}
              >
                <Text style={styles.modalClose}>← Back</Text>
              </TouchableOpacity>
              <Text style={styles.modalTitle}>Sleep Report</Text>
              <View style={{ width: 50 }} />
            </View>

            <Text style={styles.modalDate}>{formatDate(selectedReport.created_at)}</Text>

            {/* Scores */}
            <View style={styles.modalScoreCard}>
              <View style={[styles.modalScoreCircle, { borderColor: recoveryColor }]}>
                <Text style={[styles.modalScoreValue, { color: recoveryColor }]}>
                  {recoveryScore}
                </Text>
                <Text style={styles.modalScoreLabel}>Recovery Score</Text>
              </View>
              <View style={styles.modalScoreDetails}>
                <Text style={styles.modalScoreDetailText}>🕐 Circadian: {circadianScore}/100</Text>
                {sleepDebt > 0 && <Text style={styles.modalScoreDetailText}>📊 Sleep Debt: {sleepDebt}h</Text>}
              </View>
            </View>

            {/* Summary */}
            <View style={styles.modalSection}>
              <Text style={styles.modalSectionTitle}>📋 Summary</Text>
              <Text style={styles.modalTextLight}>{summary}</Text>
            </View>

            {/* Risk Assessment */}
            {riskAssessment && (
              <View style={[styles.modalSection, styles.riskSection]}>
                <Text style={[styles.modalSectionTitle, styles.riskTitle]}>⚠️ Health Risk Assessment</Text>
                <Text style={styles.modalTextLight}>{riskAssessment}</Text>
              </View>
            )}

            {/* Deep Insights */}
            {insights.length > 0 && (
              <View style={styles.modalSection}>
                <Text style={styles.modalSectionTitle}>🔬 Deep Analysis</Text>
                {insights.map((insight, idx) => (
                  <Text key={idx} style={styles.modalBulletLight}>💡 {insight}</Text>
                ))}
              </View>
            )}

            {/* Recommendations */}
            {recommendations.length > 0 && (
              <View style={styles.modalSection}>
                <Text style={styles.modalSectionTitle}>🎯 Recommendations</Text>
                {recommendations.map((rec, idx) => (
                  <Text key={idx} style={styles.modalBulletLight}>✅ {rec}</Text>
                ))}
              </View>
            )}

            {/* Positive Note */}
            {positiveNote && (
              <View style={styles.positiveNoteModal}>
                <Text style={styles.positiveNoteText}>✨ {positiveNote}</Text>
              </View>
            )}

            {/* Trends Summary */}
            {trends && trends.days_analyzed && (
              <View style={styles.modalSection}>
                <Text style={styles.modalSectionTitle}>📊 7-Day Trends</Text>
                <Text style={styles.modalTextLight}>• Average Sleep: {trends.avg_sleep_hours} hours</Text>
                <Text style={styles.modalTextLight}>• Sleep Consistency: {trends.consistency_score}</Text>
                <Text style={styles.modalTextLight}>• Days Analyzed: {trends.days_analyzed}</Text>
                {trends.sleep_social_jetlag && (
                  <Text style={styles.modalTextLight}>• Social Jetlag: {trends.sleep_social_jetlag}h</Text>
                )}
              </View>
            )}

            <View style={styles.modalDisclaimer}>
              <Text style={styles.modalDisclaimerText}>
                ⚠️ These insights are for informational purposes only. Not a substitute for professional medical advice.
              </Text>
            </View>
          </ScrollView>
        </SafeAreaView>
      </Modal>
    );
  };

  const getHeaderPadding = () => {
    if (Platform.OS === 'ios') {
      return insets.top + 10;
    }
    return insets.top + 10;
  };

  if (loading && !refreshing) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6fc6a8" />
        <Text style={styles.loadingText}>Loading history...</Text>
      </View>
    );
  }

  return (
    <>
      <ScrollView
        style={styles.container}
        contentContainerStyle={[
          styles.contentContainer,
          { paddingTop: getHeaderPadding() }
        ]}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={['#6fc6a8']} />
        }
      >
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Health History</Text>
          <Text style={styles.headerSubtitle}>
            {items.length} {items.length === 1 ? 'analysis' : 'analyses'} completed
          </Text>
        </View>

        <View style={styles.filterContainer}>
          <FilterButton title="All" active={filter === 'all'} onPress={() => setFilter('all')} />
          <FilterButton title="Sleep Analysis" active={filter === 'sleep'} onPress={() => setFilter('sleep')} />
          <FilterButton title="Apnea Detection" active={filter === 'apnea'} onPress={() => setFilter('apnea')} />
        </View>

        {filteredItems.length === 0 && (
          <View style={styles.emptyContainer}>
            <Svg width="80" height="80" viewBox="0 0 24 24" fill="none">
              <Path
                d="M9 12h6m-6 4h6m2-12H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z"
                stroke="#6fc6a8"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </Svg>
            <Text style={styles.emptyTitle}>No History Yet</Text>
            <Text style={styles.emptyText}>
              {filter === 'all' ? 'Run your first analysis to see results here.' : 
               filter === 'apnea' ? 'No apnea detection results yet. Run an analysis from the Sleep Apnea screen.' :
               'No sleep analysis results yet. Get insights from the SleepAI screen.'}
            </Text>
            <TouchableOpacity
              style={styles.analyzeButton}
              onPress={() => navigation.navigate('Analysis')}
            >
              <Text style={styles.analyzeButtonText}>Start Analysis</Text>
            </TouchableOpacity>
          </View>
        )}

        {filteredItems.map((item) => {
  if (isSleepAnalysis(item)) {
    return (
      <View key={item.id}>
        {renderSleepAnalysisCard(item)}
      </View>
    );
  } else {
    return (
      <View key={item.id}>
        {renderApneaCard(item)}
      </View>
    );
  }
})}
      </ScrollView>

      <FullReportModal />
    </>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0f0f0f',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#95a5a6',
  },
  header: {
    marginBottom: 20,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#f8fbfa',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#95a5a6',
  },
  filterContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  filterButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    backgroundColor: '#1a1a1a',
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  filterButtonActive: {
    backgroundColor: '#6fc6a8',
    borderColor: '#6fc6a8',
  },
  filterButtonText: {
    fontSize: 13,
    color: '#95a5a6',
    fontWeight: '500',
  },
  filterButtonTextActive: {
    color: '#0f0f0f',
  },
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  typeBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    gap: 6,
  },
  typeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  cardDate: {
    fontSize: 12,
    color: '#95a5a6',
  },
  predictionLabel: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#f8fbfa',
    marginBottom: 16,
  },
  recoveryContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 12,
  },
  recoveryCircle: {
    width: 60,
    height: 60,
    borderRadius: 30,
    borderWidth: 3,
    justifyContent: 'center',
    alignItems: 'center',
  },
  recoveryScore: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  recoveryLabel: {
    fontSize: 9,
    color: '#95a5a6',
    marginTop: 2,
  },
  recoveryStats: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  recoveryStat: {
    alignItems: 'center',
  },
  recoveryStatValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f8fbfa',
  },
  recoveryStatLabel: {
    fontSize: 10,
    color: '#95a5a6',
    marginTop: 4,
  },
  previewText: {
    fontSize: 13,
    color: '#95a5a6',
    lineHeight: 18,
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  apneaStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 12,
  },
  apneaStat: {
    alignItems: 'center',
    flex: 1,
  },
  apneaStatValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#f8fbfa',
  },
  apneaStatLabel: {
    fontSize: 11,
    color: '#95a5a6',
    marginTop: 4,
  },
  featurePreview: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  featurePreviewText: {
    fontSize: 12,
    color: '#6fc6a8',
    textAlign: 'center',
  },
  expandedContent: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  expandedSection: {
    marginBottom: 16,
  },
  expandedTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6fc6a8',
    marginBottom: 8,
  },
  expandedText: {
    fontSize: 13,
    color: '#d6dfdd',
    lineHeight: 20,
    marginBottom: 4,
  },
  fullReportButton: {
    backgroundColor: '#2a2a2a',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 8,
  },
  fullReportButtonText: {
    fontSize: 14,
    color: '#6fc6a8',
    fontWeight: '600',
  },
  emptyContainer: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#f8fbfa',
    marginTop: 20,
    marginBottom: 8,
  },
  emptyText: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 40,
  },
  analyzeButton: {
    backgroundColor: '#6fc6a8',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  analyzeButtonText: {
    color: '#0f0f0f',
    fontSize: 16,
    fontWeight: '600',
  },
  // Modal styles
  modalSafeArea: {
    flex: 1,
    backgroundColor: '#0f0f0f',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#0f0f0f',
  },
  modalContent: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    paddingTop: 10,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a2a',
  },
  modalBackButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    marginLeft: -12,
  },
  modalClose: {
    fontSize: 17,
    color: '#6fc6a8',
    fontWeight: '600',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#f8fbfa',
  },
  modalDate: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
    marginBottom: 20,
  },
  modalScoreCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  modalScoreCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    borderWidth: 4,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  modalScoreValue: {
    fontSize: 40,
    fontWeight: 'bold',
  },
  modalScoreLabel: {
    fontSize: 12,
    color: '#95a5a6',
    marginTop: 4,
  },
  modalScoreDetails: {
    flexDirection: 'row',
    gap: 20,
    marginTop: 12,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  modalScoreDetailText: {
    fontSize: 14,
    color: '#d6dfdd',
  },
  riskBadgeContainer: {
    marginTop: 12,
  },
  riskBadge: {
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 20,
  },
  riskBadgeText: {
    fontSize: 16,
    fontWeight: '600',
  },
  modalSection: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  riskSection: {
    borderLeftWidth: 3,
    borderLeftColor: '#e74c3c',
  },
  riskTitle: {
    color: '#e74c3c',
  },
  modalSectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#6fc6a8',
    marginBottom: 14,
  },
  modalTextLight: {
    fontSize: 15,
    color: '#e0e0e0',
    lineHeight: 24,
    marginBottom: 12,
  },
  modalBulletLight: {
    fontSize: 15,
    color: '#e0e0e0',
    lineHeight: 24,
    marginBottom: 10,
    paddingLeft: 8,
  },
  positiveNoteModal: {
    backgroundColor: 'rgba(111, 198, 168, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 3,
    borderLeftColor: '#6fc6a8',
  },
  positiveNoteText: {
    fontSize: 14,
    color: '#6fc6a8',
    lineHeight: 22,
    fontStyle: 'italic',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  metricCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: '#0f0f0f',
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
  },
  metricValue: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#6fc6a8',
  },
  metricLabel: {
    fontSize: 12,
    color: '#95a5a6',
    marginTop: 6,
  },
  modalDisclaimer: {
    backgroundColor: 'rgba(111, 198, 168, 0.1)',
    padding: 16,
    borderRadius: 12,
    marginTop: 8,
    marginBottom: 20,
  },
  modalDisclaimerText: {
    fontSize: 12,
    color: '#95a5a6',
    textAlign: 'center',
    lineHeight: 18,
  },
});