// src/screens/SleepAIScreen.js
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Platform,
} from 'react-native';
import { useAuth } from '../context/AuthContext';
import { useFitbit } from '../context/FitbitContext';
import { getSleepInsights } from '../services/sleepAIService';
import Svg, { Path, Circle } from 'react-native-svg';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export default function SleepAIScreen({ navigation }) {
  const { user, profile } = useAuth();
  const { isConnected } = useFitbit();
  const insets = useSafeAreaInsets();
  const [loading, setLoading] = useState(false);
  const [insights, setInsights] = useState(null);
  const [sleepData, setSleepData] = useState(null);
  const [trends, setTrends] = useState(null);
  const [isLLM, setIsLLM] = useState(false);

  useEffect(() => {
    if (isConnected) {
      fetchInsights();
    }
  }, [isConnected]);

  const getHeaderPadding = () => {
    if (Platform.OS === 'ios') {
      return insets.top + 10;
    }
    return insets.top + 10;
  };

  const fetchInsights = async () => {
    setLoading(true);
    try {
      const result = await getSleepInsights(user.id, profile);
      
      if (result.success) {
        setSleepData(result.data.sleepData);
        setTrends(result.data.trends);
        setInsights(result.data.insights);
        setIsLLM(result.data.isLLM || false);
      } else {
        Alert.alert('Error', result.error || 'Failed to get sleep insights');
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const AIIcon = () => (
    <Svg width="60" height="60" viewBox="0 0 24 24" fill="none">
      <Path
        d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"
        stroke="#6fc6a8"
        strokeWidth="1.5"
        fill="none"
      />
      <Path
        d="M12 8v4m0 4h.01"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <Path
        d="M8 12h8"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </Svg>
  );

  const BrainIcon = () => (
    <Svg width="24" height="24" viewBox="0 0 24 24" fill="none">
      <Path
        d="M12 4a4 4 0 0 1 4 4c0 1.5-.8 2.8-2 3.5V15a2 2 0 0 1-4 0v-3.5c-1.2-.7-2-2-2-3.5a4 4 0 0 1 4-4z"
        stroke="#6fc6a8"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </Svg>
  );

  if (!isConnected) {
    return (
      <ScrollView 
        style={styles.container} 
        contentContainerStyle={[
          styles.contentContainer,
          { paddingTop: getHeaderPadding() }
        ]}
        showsVerticalScrollIndicator={true}
      >
        <View style={styles.noDataContainer}>
          <AIIcon />
          <Text style={styles.noDataTitle}>Connect Fitbit First</Text>
          <Text style={styles.noDataText}>
            Connect your Fitbit account and sync your sleep data to get AI-powered insights.
          </Text>
          <TouchableOpacity
            style={styles.connectButton}
            onPress={() => navigation.navigate('Analysis')}
          >
            <Text style={styles.connectButtonText}>Connect Fitbit</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    );
  }

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6fc6a8" />
        <Text style={styles.loadingText}>Analyzing your sleep data...</Text>
        <Text style={styles.loadingSubtext}>Our AI is generating personalized insights</Text>
      </View>
    );
  }

  if (!insights) {
    return (
      <ScrollView 
        style={styles.container} 
        contentContainerStyle={[
          styles.contentContainer,
          { paddingTop: getHeaderPadding() }
        ]}
        showsVerticalScrollIndicator={true}
      >
        <View style={styles.noDataContainer}>
          <AIIcon />
          <Text style={styles.noDataTitle}>No Sleep Data Yet</Text>
          <Text style={styles.noDataText}>
            Sync your Fitbit data first to get AI-powered sleep insights.
          </Text>
          <TouchableOpacity
            style={styles.syncButton}
            onPress={() => navigation.navigate('Analysis')}
          >
            <Text style={styles.syncButtonText}>Go to Analysis</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    );
  }

  return (
    <ScrollView 
      style={styles.container} 
      contentContainerStyle={[
        styles.contentContainer,
        { paddingTop: getHeaderPadding() }
      ]}
      showsVerticalScrollIndicator={true}
    >
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerIcon}>
          <BrainIcon />
        </View>
        <Text style={styles.headerTitle}>SleepAI Analyst</Text>
        <Text style={styles.headerSubtitle}>
          AI-powered insights based on your sleep data
        </Text>
      </View>

      {/* Last Night Summary */}
      {sleepData && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Last Night's Sleep</Text>
          <View style={styles.sleepSummary}>
            <View style={styles.sleepStat}>
              <Text style={styles.sleepStatValue}>{sleepData.duration_hours}h</Text>
              <Text style={styles.sleepStatLabel}>Duration</Text>
            </View>
            <View style={styles.sleepStat}>
              <Text style={styles.sleepStatValue}>{sleepData.quality_score}%</Text>
              <Text style={styles.sleepStatLabel}>Quality</Text>
            </View>
            <View style={styles.sleepStat}>
              <Text style={styles.sleepStatValue}>{sleepData.stages.deep.minutes}m</Text>
              <Text style={styles.sleepStatLabel}>Deep Sleep</Text>
            </View>
            <View style={styles.sleepStat}>
              <Text style={styles.sleepStatValue}>{sleepData.stages.rem.minutes}m</Text>
              <Text style={styles.sleepStatLabel}>REM Sleep</Text>
            </View>
          </View>
          <Text style={styles.sleepTime}>
            🛏️ {sleepData.bedtime} → 🌅 {sleepData.wakeup}
          </Text>
        </View>
      )}

      {/* Weekly Trends */}
      {trends && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>7-Day Trends</Text>
          <View style={styles.trendsGrid}>
            <View style={styles.trendItem}>
              <Text style={styles.trendValue}>{trends.avg_sleep_hours}h</Text>
              <Text style={styles.trendLabel}>Avg Sleep</Text>
            </View>
            <View style={styles.trendItem}>
              <Text style={styles.trendValue}>{trends.avg_deep_minutes}m</Text>
              <Text style={styles.trendLabel}>Avg Deep</Text>
            </View>
            <View style={styles.trendItem}>
              <Text style={styles.trendValue}>{trends.avg_rem_minutes}m</Text>
              <Text style={styles.trendLabel}>Avg REM</Text>
            </View>
            <View style={styles.trendItem}>
              <Text style={styles.trendValue}>{trends.avg_efficiency}%</Text>
              <Text style={styles.trendLabel}>Avg Efficiency</Text>
            </View>
          </View>
          <Text style={styles.trendNote}>
            Based on {trends.days_analyzed} nights of data
          </Text>
        </View>
      )}

      {/* AI Insights */}
      <View style={styles.insightsCard}>
        <View style={styles.aiBadge}>
          <Text style={styles.aiBadgeText}>🤖 AI Analyst</Text>
          {!isLLM && <Text style={styles.fallbackBadge}>Fallback Mode</Text>}
          {insights.recoveryScore && (
            <View style={[styles.recoveryBadge, { backgroundColor: insights.recoveryScore >= 70 ? 'rgba(46, 204, 113, 0.2)' : insights.recoveryScore >= 50 ? 'rgba(243, 156, 18, 0.2)' : 'rgba(231, 76, 60, 0.2)' }]}>
              <Text style={[styles.recoveryText, { color: insights.recoveryScore >= 70 ? '#2ecc71' : insights.recoveryScore >= 50 ? '#f39c12' : '#e74c3c' }]}>
                Recovery: {insights.recoveryScore}/100
              </Text>
            </View>
          )}
        </View>
        
        <Text style={styles.insightsText}>{insights.summary}</Text>
        
        {/* Risk Assessment */}
        {insights.riskAssessment && (
          <View style={styles.riskAssessment}>
            <Text style={styles.riskAssessmentTitle}>⚠️ Health Risk Assessment</Text>
            <Text style={styles.riskAssessmentText}>{insights.riskAssessment}</Text>
          </View>
        )}
        
        {/* Deep Insights */}
        {insights.deepInsights && insights.deepInsights.length > 0 && (
          <>
            <Text style={styles.sectionTitle}>🔬 Deep Analysis</Text>
            {insights.deepInsights.map((insight, index) => (
              <View key={index} style={styles.bulletPoint}>
                <View style={styles.bulletDot} />
                <Text style={styles.bulletText}>{insight}</Text>
              </View>
            ))}
          </>
        )}
        
        {/* Recommendations */}
        {insights.recommendations && insights.recommendations.length > 0 && (
          <>
            <Text style={styles.sectionTitle}>📋 Recommendations</Text>
            {insights.recommendations.map((rec, index) => (
              <View key={index} style={styles.bulletPoint}>
                <View style={[styles.bulletDot, styles.recommendationDot]} />
                <Text style={styles.bulletText}>{rec}</Text>
              </View>
            ))}
          </>
        )}
        
        {/* Positive Note */}
        {insights.positiveNote && (
          <View style={styles.positiveNote}>
            <Text style={styles.positiveNoteText}>✨ {insights.positiveNote}</Text>
          </View>
        )}
        
        {/* Refresh Button */}
        <TouchableOpacity
          style={styles.refreshButton}
          onPress={fetchInsights}
        >
          <Text style={styles.refreshButtonText}>⟳ Refresh Insights</Text>
        </TouchableOpacity>
        
        {/* Disclaimer */}
        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerText}>
            ⚠️ AI-generated insights are for informational purposes only. 
            Not a substitute for professional medical advice.
          </Text>
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
    // For web: ensure the container can grow to accommodate content
    minHeight: Platform.OS === 'web' ? '100%' : undefined,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0f0f0f',
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#f8fbfa',
  },
  loadingSubtext: {
    marginTop: 8,
    fontSize: 14,
    color: '#95a5a6',
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  headerIcon: {
    marginBottom: 12,
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
    textAlign: 'center',
  },
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#f8fbfa',
    marginBottom: 16,
  },
  sleepSummary: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 16,
    flexWrap: 'wrap',
    gap: 12,
  },
  sleepStat: {
    alignItems: 'center',
    minWidth: 70,
  },
  sleepStatValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#6fc6a8',
  },
  sleepStatLabel: {
    fontSize: 12,
    color: '#95a5a6',
    marginTop: 4,
  },
  sleepTime: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  trendsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 12,
    flexWrap: 'wrap',
    gap: 12,
  },
  trendItem: {
    alignItems: 'center',
    minWidth: 70,
  },
  trendValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#6fc6a8',
  },
  trendLabel: {
    fontSize: 11,
    color: '#95a5a6',
    marginTop: 4,
  },
  trendNote: {
    fontSize: 12,
    color: '#95a5a6',
    textAlign: 'center',
  },
  insightsCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  aiBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a2a',
    flexWrap: 'wrap',
    gap: 8,
  },
  aiBadgeText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6fc6a8',
  },
  fallbackBadge: {
    fontSize: 12,
    color: '#f39c12',
    backgroundColor: 'rgba(243, 156, 18, 0.1)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  insightsText: {
    fontSize: 15,
    color: '#f8fbfa',
    lineHeight: 24,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f8fbfa',
    marginTop: 16,
    marginBottom: 12,
  },
  bulletPoint: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  bulletDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#6fc6a8',
    marginTop: 8,
    marginRight: 12,
    flexShrink: 0,
  },
  recommendationDot: {
    backgroundColor: '#f39c12',
  },
  bulletText: {
    flex: 1,
    fontSize: 14,
    color: '#d6dfdd',
    lineHeight: 22,
  },
  positiveNote: {
    marginTop: 20,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  positiveNoteText: {
    fontSize: 14,
    color: '#6fc6a8',
    lineHeight: 22,
    fontStyle: 'italic',
  },
  refreshButton: {
    backgroundColor: '#2a2a2a',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 20,
  },
  refreshButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f8fbfa',
  },
  disclaimer: {
    backgroundColor: 'rgba(111, 198, 168, 0.1)',
    padding: 16,
    borderRadius: 12,
  },
  disclaimerText: {
    fontSize: 12,
    color: '#95a5a6',
    textAlign: 'center',
    lineHeight: 18,
  },
  noDataContainer: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  noDataTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#f8fbfa',
    marginTop: 24,
    marginBottom: 8,
  },
  noDataText: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 20,
  },
  connectButton: {
    backgroundColor: '#6fc6a8',
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 10,
  },
  connectButtonText: {
    color: '#0f0f0f',
    fontSize: 16,
    fontWeight: '600',
  },
  syncButton: {
    backgroundColor: '#2a2a2a',
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 10,
  },
  syncButtonText: {
    color: '#f8fbfa',
    fontSize: 16,
    fontWeight: '600',
  },
  recoveryBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  recoveryText: {
    fontSize: 12,
    fontWeight: '600',
  },
  riskAssessment: {
    backgroundColor: 'rgba(231, 76, 60, 0.1)',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    borderLeftWidth: 3,
    borderLeftColor: '#e74c3c',
  },
  riskAssessmentTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#e74c3c',
    marginBottom: 6,
  },
  riskAssessmentText: {
    fontSize: 13,
    color: '#d6dfdd',
    lineHeight: 20,
  },
});