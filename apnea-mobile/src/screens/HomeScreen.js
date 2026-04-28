// src/screens/HomeScreen.js
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
  RefreshControl,
  Platform,
  StatusBar,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { useAuth } from '../context/AuthContext';
import { useFitbit } from '../context/FitbitContext';
import { supabase } from '../services/supabase';
import Svg, { Path, Circle, Line, G, Text as SvgText, Rect } from 'react-native-svg';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

const { width } = Dimensions.get('window');
const CHART_WIDTH = width - 80;
const BAR_WIDTH = (CHART_WIDTH - 50) / 7;
const BAR_GAP = 4;
const LINE_CHART_WIDTH = width - 100;
const LINE_CHART_HEIGHT = 180;

export default function HomeScreen({ navigation }) {
  const { user } = useAuth();
  const { isConnected } = useFitbit();
  const insets = useSafeAreaInsets();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [stats, setStats] = useState({
    avgSleepHours: 0,
    avgDeepHours: 0,
    avgLightHours: 0,
    avgRemHours: 0,
    sleepEfficiency: 0,
    avgRestingHeartRate: 0,
    lastNight: null,
    weeklyData: [],
    hrvData: [],
    heartRateData: [],
  });
  const [currentWeekOffset, setCurrentWeekOffset] = useState(0);
  const [expandedBar, setExpandedBar] = useState(null);
  const [selectedDay, setSelectedDay] = useState(null);
  const [hasValidData, setHasValidData] = useState(true);
  const [maxWeeksBack, setMaxWeeksBack] = useState(52);

  useFocusEffect(
    useCallback(() => {
      if (user && isConnected) {
        fetchData();
      }
      return () => {};
    }, [user, isConnected, currentWeekOffset])
  );

  const getHeaderPadding = () => {
    if (Platform.OS === 'ios') {
      const { height: screenHeight } = Dimensions.get('window');
      if (screenHeight >= 812) {
        return insets.top + 10;
      }
      return insets.top + 10;
    }
    return (StatusBar.currentHeight || 30) + 10;
  };

  const fetchData = async () => {
    setLoading(true);
    setHasValidData(true);
    try {
      const today = new Date();
      const startDate = new Date(today);
      startDate.setDate(today.getDate() - (today.getDay() + 7 * currentWeekOffset));
      const endDate = new Date(startDate);
      endDate.setDate(startDate.getDate() + 6);

      const startStr = startDate.toISOString().split('T')[0];
      const endStr = endDate.toISOString().split('T')[0];

      const { data, error } = await supabase
        .from('fitbit_data')
        .select('*')
        .eq('user_id', user.id)
        .gte('date', startStr)
        .lte('date', endStr)
        .order('date', { ascending: true });

      if (error) throw error;

      await processSleepData(data || []);
      await processHRVData(data || []);
      await processHeartRateData(data || []);
      
      const hasAnyData = data && data.length > 0;
      if (!hasAnyData && currentWeekOffset > 0) {
        setMaxWeeksBack(currentWeekOffset);
      }
      
    } catch (error) {
      console.error('Error fetching home data:', error);
      setHasValidData(false);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  const processSleepData = async (data) => {
    let totalSleepHours = 0;
    let totalDeepHours = 0;
    let totalLightHours = 0;
    let totalRemHours = 0;
    let totalEfficiency = 0;
    let validNights = 0;
    let weeklyData = [];

    const today = new Date();
    const startDate = new Date(today);
    startDate.setDate(today.getDate() - (today.getDay() + 7 * currentWeekOffset));
    
    const weekMap = {};
    for (let i = 0; i < 7; i++) {
      const date = new Date(startDate);
      date.setDate(startDate.getDate() + i);
      const dateStr = date.toISOString().split('T')[0];
      weekMap[dateStr] = {
        date: dateStr,
        dayName: date.toLocaleDateString('en-GB', { weekday: 'short' }),
        fullDate: date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }),
        hours: 0,
        deepHours: 0,
        lightHours: 0,
        remHours: 0,
        wakeHours: 0,
        efficiency: 0,
        hasData: false,
      };
    }

    for (const record of data) {
      const sleep = record.sleep_data?.[0];
      if (sleep && sleep.minutesAsleep > 0 && weekMap[record.date]) {
        validNights++;
        const sleepHours = sleep.minutesAsleep / 60;
        const deepHours = (sleep.levels?.summary?.deep?.minutes || 0) / 60;
        const lightHours = (sleep.levels?.summary?.light?.minutes || 0) / 60;
        const remHours = (sleep.levels?.summary?.rem?.minutes || 0) / 60;
        const wakeHours = (sleep.minutesAwake || 0) / 60;
        
        totalSleepHours += sleepHours;
        totalDeepHours += deepHours;
        totalLightHours += lightHours;
        totalRemHours += remHours;
        totalEfficiency += sleep.efficiency || 0;

        weekMap[record.date] = {
          ...weekMap[record.date],
          hours: sleepHours,
          deepHours: deepHours,
          lightHours: lightHours,
          remHours: remHours,
          wakeHours: wakeHours,
          efficiency: sleep.efficiency || 0,
          hasData: true,
        };
      }
    }

    weeklyData = Object.values(weekMap);
    const lastNightData = weeklyData.find(d => d.hasData) || null;

    setStats(prev => ({
      ...prev,
      avgSleepHours: validNights > 0 ? (totalSleepHours / validNights).toFixed(1) : 0,
      avgDeepHours: validNights > 0 ? (totalDeepHours / validNights).toFixed(1) : 0,
      avgLightHours: validNights > 0 ? (totalLightHours / validNights).toFixed(1) : 0,
      avgRemHours: validNights > 0 ? (totalRemHours / validNights).toFixed(1) : 0,
      sleepEfficiency: validNights > 0 ? Math.round(totalEfficiency / validNights) : 0,
      lastNight: lastNightData,
      weeklyData: weeklyData,
      totalNights: validNights,
    }));
  };

  const processHRVData = async (data) => {
    const hrvValues = [];
    let minHRV = Infinity;
    let maxHRV = -Infinity;
    
    for (const record of data) {
      const date = new Date(record.date);
      const dayName = date.toLocaleDateString('en-GB', { weekday: 'short' });
      
      let dailyRMSSD = null;
      
      if (record.hrv_data && record.hrv_data.hrv && record.hrv_data.hrv.length > 0) {
        const hrvEntry = record.hrv_data.hrv[0];
        
        if (hrvEntry && hrvEntry.minutes && hrvEntry.minutes.length > 0) {
          let totalRMSSD = 0;
          let validMinutes = 0;
          
          for (const minute of hrvEntry.minutes) {
            if (minute.value && minute.value.rmssd) {
              totalRMSSD += minute.value.rmssd;
              validMinutes++;
            }
          }
          
          if (validMinutes > 0) {
            dailyRMSSD = totalRMSSD / validMinutes;
            if (dailyRMSSD < minHRV) minHRV = dailyRMSSD;
            if (dailyRMSSD > maxHRV) maxHRV = dailyRMSSD;
          }
        }
      }
      
      hrvValues.push({
        date: record.date,
        dayName: dayName,
        value: dailyRMSSD,
        hasData: dailyRMSSD !== null,
      });
    }
    
    if (minHRV === Infinity) minHRV = 0;
    if (maxHRV === -Infinity) maxHRV = 100;
    if (minHRV === maxHRV) maxHRV = minHRV + 10;
    
    setStats(prev => ({
      ...prev,
      hrvData: hrvValues,
      hrvMin: minHRV,
      hrvMax: maxHRV,
    }));
  };

  const processHeartRateData = async (data) => {
    const heartRateValues = [];
    let totalRestingHR = 0;
    let validHRDays = 0;
    let minHR = Infinity;
    let maxHR = -Infinity;
    
    for (const record of data) {
      const date = new Date(record.date);
      const dayName = date.toLocaleDateString('en-GB', { weekday: 'short' });
      
      let restingHR = null;
      
      if (record.heart_rate_data && record.heart_rate_data[0]?.value?.restingHeartRate) {
        restingHR = record.heart_rate_data[0].value.restingHeartRate;
        totalRestingHR += restingHR;
        validHRDays++;
        if (restingHR < minHR) minHR = restingHR;
        if (restingHR > maxHR) maxHR = restingHR;
      }
      
      heartRateValues.push({
        date: record.date,
        dayName: dayName,
        value: restingHR,
        hasData: restingHR !== null,
      });
    }
    
    if (minHR === Infinity) minHR = 60;
    if (maxHR === -Infinity) maxHR = 80;
    if (minHR === maxHR) maxHR = minHR + 10;
    
    setStats(prev => ({
      ...prev,
      heartRateData: heartRateValues,
      avgRestingHeartRate: validHRDays > 0 ? Math.round(totalRestingHR / validHRDays) : 0,
      hrMin: minHR,
      hrMax: maxHR,
    }));
  };

  const handleWeekChange = (direction) => {
    const newOffset = currentWeekOffset + direction;
    
    if (direction === -1 && newOffset < 0) return;
    if (direction === 1 && newOffset > maxWeeksBack) return;
    
    setCurrentWeekOffset(newOffset);
    setExpandedBar(null);
    setSelectedDay(null);
  };

  const handleBarPress = (index, day) => {
    if (!day.hasData) return;
    if (expandedBar === index) {
      setExpandedBar(null);
      setSelectedDay(null);
    } else {
      setExpandedBar(index);
      setSelectedDay(day);
    }
  };

  const formatWeekRange = () => {
    if (!stats.weeklyData || stats.weeklyData.length === 0) {
      return 'No data for this week';
    }
    const firstDay = stats.weeklyData[0]?.date;
    const lastDay = stats.weeklyData[6]?.date;
    if (!firstDay || !lastDay) {
      return 'Invalid date range';
    }
    
    try {
      const start = new Date(firstDay);
      const end = new Date(lastDay);
      return `${start.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })} - ${end.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}`;
    } catch (e) {
      return 'Invalid date range';
    }
  };

  const calculateSleepQuality = () => {
    if (stats.totalNights === 0) return 0;
    
    let score = 0;
    const avgHours = parseFloat(stats.avgSleepHours);
    if (avgHours >= 7) score += 30;
    else if (avgHours >= 6) score += 20;
    else if (avgHours >= 5) score += 10;
    
    const deepPct = parseFloat(stats.avgDeepHours) / parseFloat(stats.avgSleepHours);
    if (deepPct >= 0.2) score += 30;
    else if (deepPct >= 0.15) score += 20;
    else if (deepPct >= 0.1) score += 10;
    
    const remPct = parseFloat(stats.avgRemHours) / parseFloat(stats.avgSleepHours);
    if (remPct >= 0.2) score += 20;
    else if (remPct >= 0.15) score += 15;
    else if (remPct >= 0.1) score += 10;
    
    if (stats.sleepEfficiency >= 90) score += 20;
    else if (stats.sleepEfficiency >= 85) score += 15;
    else if (stats.sleepEfficiency >= 80) score += 10;
    
    return Math.min(100, score);
  };

  const handleBarPressWithFeedback = (index, day) => {
    if (!day || !day.hasData) return;
    handleBarPress(index, day);
  };

  // Interactive Line Chart Component with tooltips
  const InteractiveLineChart = ({ title, data, unit, color, note, minValue, maxValue }) => {
    const [tooltip, setTooltip] = useState(null);
    const hasData = data && data.some(d => d && d.hasData);
    
    if (!hasData) {
      return (
        <View style={styles.noDataSmallContainer}>
          <Text style={styles.noDataSmallText}>No {title ? title.toLowerCase() : 'sleep'} data available for this week</Text>
        </View>
      );
    }
    
    const values = data.filter(d => d && d.hasData).map(d => d.value).filter(v => v !== null && v !== undefined);
    if (values.length === 0) {
      return (
        <View style={styles.noDataSmallContainer}>
          <Text style={styles.noDataSmallText}>No valid {title ? title.toLowerCase() : 'sleep'} data available</Text>
        </View>
      );
    }
    
    const actualMin = minValue !== undefined ? minValue : Math.min(...values);
    const actualMax = maxValue !== undefined ? maxValue : Math.max(...values);
    const range = actualMax - actualMin === 0 ? 10 : actualMax - actualMin;
    
    const getYPosition = (value) => {
      if (range === 0) return LINE_CHART_HEIGHT - 30;
      return LINE_CHART_HEIGHT - 30 - ((value - actualMin) / range) * (LINE_CHART_HEIGHT - 50);
    };
    
    const points = [];
    const linePoints = [];
    
    data.forEach((item, index) => {
      if (!item) return;
      const x = 40 + (index * (LINE_CHART_WIDTH - 45) / 6);
      const y = item.hasData ? getYPosition(item.value) : LINE_CHART_HEIGHT - 30;
      points.push({ x, y, hasData: item.hasData, value: item.value, dayName: item.dayName || '', date: item.date || '' });
      
      if (item.hasData) {
        linePoints.push({ x, y });
      }
    });
    
    const linePath = linePoints.reduce((path, point, idx) => {
      return path + (idx === 0 ? `M ${point.x} ${point.y}` : ` L ${point.x} ${point.y}`);
    }, '');
    
    const handlePointPress = (point) => {
      if (point && point.hasData) {
        setTooltip(point);
        setTimeout(() => setTooltip(null), 3000);
      }
    };
    
    const getPointHandlers = (point) => {
      if (Platform.OS === 'web') {
        return {
          onMouseEnter: () => handlePointPress(point),
          onMouseLeave: () => setTooltip(null),
        };
      } else {
        return {
          onPressIn: () => handlePointPress(point),
        };
      }
    };
    
    const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
    
    const getTooltipPosition = (point) => {
      let tooltipX = point.x;
      let tooltipY = point.y - 45;
      
      if (tooltipX < 50) tooltipX = 50;
      if (tooltipX > LINE_CHART_WIDTH - 50) tooltipX = LINE_CHART_WIDTH - 50;
      if (tooltipY < 10) tooltipY = point.y + 15;
      
      return { x: tooltipX, y: tooltipY };
    };
    
    return (
      <View style={styles.chartCard}>
        <View style={styles.chartHeader}>
          <Text style={styles.chartTitle}>{title || 'Chart'}</Text>
          <Text style={[styles.chartValue, { color: color || '#6fc6a8' }]}>{Math.round(avgValue)} {unit || ''}</Text>
        </View>
        
        <View style={styles.svgContainer}>
          <Svg width={LINE_CHART_WIDTH} height={LINE_CHART_HEIGHT}>
            <SvgText x="30" y={getYPosition(actualMax)} fontSize="10" fill="#95a5a6" textAnchor="end">
              {Math.round(actualMax)}
            </SvgText>
            <SvgText x="30" y={getYPosition(actualMin + range * 0.75)} fontSize="10" fill="#95a5a6" textAnchor="end">
              {Math.round(actualMin + range * 0.75)}
            </SvgText>
            <SvgText x="30" y={getYPosition(actualMin + range * 0.5)} fontSize="10" fill="#95a5a6" textAnchor="end">
              {Math.round(actualMin + range * 0.5)}
            </SvgText>
            <SvgText x="30" y={getYPosition(actualMin + range * 0.25)} fontSize="10" fill="#95a5a6" textAnchor="end">
              {Math.round(actualMin + range * 0.25)}
            </SvgText>
            <SvgText x="30" y={getYPosition(actualMin)} fontSize="10" fill="#95a5a6" textAnchor="end">
              {Math.round(actualMin)}
            </SvgText>
            
            <Line x1="35" y1={getYPosition(actualMax)} x2={LINE_CHART_WIDTH - 10} y2={getYPosition(actualMax)} stroke="#2a2a2a" strokeWidth="0.5" strokeDasharray="4" />
            <Line x1="35" y1={getYPosition(actualMin + range * 0.5)} x2={LINE_CHART_WIDTH - 10} y2={getYPosition(actualMin + range * 0.5)} stroke="#2a2a2a" strokeWidth="0.5" strokeDasharray="4" />
            <Line x1="35" y1={getYPosition(actualMin)} x2={LINE_CHART_WIDTH - 10} y2={getYPosition(actualMin)} stroke="#2a2a2a" strokeWidth="0.5" strokeDasharray="4" />
            
            {linePath ? <Path d={linePath} stroke={color || '#6fc6a8'} strokeWidth="2" fill="none" /> : null}
            
            {linePath && linePoints.length > 0 ? (
              <Path
                d={`${linePath} L ${linePoints[linePoints.length - 1].x} ${LINE_CHART_HEIGHT - 30} L ${linePoints[0].x} ${LINE_CHART_HEIGHT - 30} Z`}
                fill={`${color || '#6fc6a8'}20`}
              />
            ) : null}
            
            {points.map((point, index) => (
              <G key={index} {...getPointHandlers(point)} style={Platform.OS === 'web' ? { cursor: 'pointer' } : {}}>
                {point.hasData ? (
                  <>
                    <Circle
                      cx={point.x}
                      cy={point.y}
                      r="8"
                      fill={`${color || '#6fc6a8'}30`}
                    />
                    <Circle
                      cx={point.x}
                      cy={point.y}
                      r="5"
                      fill={color || '#6fc6a8'}
                    />
                  </>
                ) : (
                  <Circle cx={point.x} cy={point.y} r="3" fill="#4a4a4a" />
                )}
              </G>
            ))}
            
            {tooltip ? (
              <G>
                <Rect
                  x={getTooltipPosition(tooltip).x - 40}
                  y={getTooltipPosition(tooltip).y}
                  width="80"
                  height="40"
                  rx="6"
                  fill="#1a1a1a"
                  stroke={color || '#6fc6a8'}
                  strokeWidth="1"
                />
                <SvgText x={getTooltipPosition(tooltip).x} y={getTooltipPosition(tooltip).y + 15} fontSize="12" fill={color || '#6fc6a8'} fontWeight="bold" textAnchor="middle">
                  {Math.round(tooltip.value)} {unit || ''}
                </SvgText>
                <SvgText x={getTooltipPosition(tooltip).x} y={getTooltipPosition(tooltip).y + 28} fontSize="10" fill="#95a5a6" textAnchor="middle">
                  {tooltip.dayName || ''}
                </SvgText>
              </G>
            ) : null}
          </Svg>
        </View>
        
        <View style={styles.xAxisLabels}>
          {data.map((item, index) => (
            <Text key={index} style={[styles.xAxisLabel, !item || !item.hasData ? styles.xAxisLabelMuted : null]}>
              {item && item.dayName ? item.dayName : ''}
            </Text>
          ))}
        </View>
        
        <View style={styles.chartNote}>
          <Text style={styles.chartNoteText}>{note || ''}</Text>
          <Text style={styles.chartNoteSubtext}>💡 Tap any data point to see the exact value</Text>
        </View>
      </View>
    );
  };

  // Interactive Stacked Bar Chart Component
  const StackedBarChart = () => {
  if (!stats.weeklyData || stats.weeklyData.length === 0) {
    return (
      <View style={styles.noDataChartContainer}>
        <Text style={styles.noDataChartText}>No sleep data for this week</Text>
        <Text style={styles.noDataChartSubtext}>Try navigating to another week</Text>
      </View>
    );
  }
  
  const hasAnyData = stats.weeklyData.some(d => d && d.hasData);
  const maxHours = hasAnyData 
    ? Math.max(...stats.weeklyData.filter(d => d && d.hasData).map(d => d.hours || 0), 8)
    : 8;
  const chartHeight = 140;
  
  if (!hasAnyData) {
    return (
      <View style={styles.noDataChartContainer}>
        <Text style={styles.noDataChartText}>No sleep data for this week</Text>
        <Text style={styles.noDataChartSubtext}>Try navigating to another week</Text>
      </View>
    );
  }
  
  return (
    <View style={styles.chartContainer}>
      <View style={styles.chartYAxis}>
        <Text style={styles.yAxisLabel}>8h</Text>
        <Text style={styles.yAxisLabel}>6h</Text>
        <Text style={styles.yAxisLabel}>4h</Text>
        <Text style={styles.yAxisLabel}>2h</Text>
        <Text style={styles.yAxisLabel}>0h</Text>
      </View>
      
      <View style={styles.chartBars}>
        {stats.weeklyData.map((day, index) => {
          // Handle days with no data
          if (!day || !day.hasData) {
            return (
              <View key={index} style={styles.barWrapper}>
                <View style={[styles.barStack, { height: 4, backgroundColor: '#2a2a2a', justifyContent: 'flex-end' }]}>
                  <View style={[styles.barSegment, { height: 4, backgroundColor: '#3a3a3a' }]} />
                </View>
                <Text style={[styles.barLabel, styles.barLabelMuted]}>{day?.dayName || ''}</Text>
              </View>
            );
          }
          
          // Calculate heights as percentages of the max height
          const totalHours = day.hours || 0;
          const totalHeight = Math.max((totalHours / maxHours) * chartHeight, 4);
          
          // Calculate each stage's height as a percentage of the total bar height
          // This ensures proper stacking without position calculations
          const deepPct = Math.min((day.deepHours || 0) / totalHours, 1);
          const lightPct = Math.min((day.lightHours || 0) / totalHours, 1);
          const remPct = Math.min((day.remHours || 0) / totalHours, 1);
          const wakePct = Math.min((day.wakeHours || 0) / totalHours, 1);
          
          // Stack from BOTTOM to TOP: Deep -> Light -> REM -> Wake
          const deepHeight = totalHeight * deepPct;
          const lightHeight = totalHeight * lightPct;
          const remHeight = totalHeight * remPct;
          const wakeHeight = totalHeight * wakePct;
          
          // Calculate positions from bottom
          const deepBottom = 0;
          const lightBottom = deepHeight;
          const remBottom = deepHeight + lightHeight;
          const wakeBottom = deepHeight + lightHeight + remHeight;
          
          return (
            <TouchableOpacity
              key={index}
              style={styles.barWrapper}
              onPress={() => handleBarPressWithFeedback(index, day)}
              activeOpacity={0.7}
            >
              <View style={[styles.barStack, { height: totalHeight, backgroundColor: '#2a2a2a' }]}>
                {/* Deep Sleep - Bottom */}
                {deepHeight > 0 && (
                  <View 
                    style={[
                      styles.barSegment, 
                      { 
                        height: deepHeight, 
                        backgroundColor: '#3a7c5c',
                        bottom: deepBottom,
                      }
                    ]} 
                  />
                )}
                {/* Light Sleep */}
                {lightHeight > 0 && (
                  <View 
                    style={[
                      styles.barSegment, 
                      { 
                        height: lightHeight, 
                        backgroundColor: '#5a9e7a',
                        bottom: lightBottom,
                      }
                    ]} 
                  />
                )}
                {/* REM Sleep */}
                {remHeight > 0 && (
                  <View 
                    style={[
                      styles.barSegment, 
                      { 
                        height: remHeight, 
                        backgroundColor: '#7ab89a',
                        bottom: remBottom,
                      }
                    ]} 
                  />
                )}
                {/* Wake - Top */}
                {wakeHeight > 0 && (
                  <View 
                    style={[
                      styles.barSegment, 
                      { 
                        height: wakeHeight, 
                        backgroundColor: '#8a6e5a',
                        bottom: wakeBottom,
                      }
                    ]} 
                  />
                )}
              </View>
              <Text style={styles.barLabel}>{day.dayName || ''}</Text>
              {totalHours > 0 && !isNaN(totalHours) ? (
                <Text style={styles.barHourLabel}>{totalHours.toFixed(1)}h</Text>
              ) : null}
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
};

  const ExpandedDayDetails = () => {
    if (!selectedDay || !selectedDay.hasData) return null;
    
    const totalHours = (selectedDay.deepHours || 0) + (selectedDay.lightHours || 0) + (selectedDay.remHours || 0) + (selectedDay.wakeHours || 0);
    const deepPercent = totalHours > 0 ? ((selectedDay.deepHours || 0) / totalHours * 100).toFixed(1) : '0';
    const lightPercent = totalHours > 0 ? ((selectedDay.lightHours || 0) / totalHours * 100).toFixed(1) : '0';
    const remPercent = totalHours > 0 ? ((selectedDay.remHours || 0) / totalHours * 100).toFixed(1) : '0';
    const wakePercent = totalHours > 0 ? ((selectedDay.wakeHours || 0) / totalHours * 100).toFixed(1) : '0';
    
    const deepWidth = totalHours > 0 ? ((selectedDay.deepHours || 0) / totalHours) * 100 : 0;
    const lightWidth = totalHours > 0 ? ((selectedDay.lightHours || 0) / totalHours) * 100 : 0;
    const remWidth = totalHours > 0 ? ((selectedDay.remHours || 0) / totalHours) * 100 : 0;
    const wakeWidth = totalHours > 0 ? ((selectedDay.wakeHours || 0) / totalHours) * 100 : 0;
    
    const hrvForDay = stats.hrvData.find(h => h && h.date === selectedDay.date);
    const hrvValue = hrvForDay && hrvForDay.hasData ? hrvForDay.value : null;
    
    const hrForDay = stats.heartRateData.find(h => h && h.date === selectedDay.date);
    const hrValue = hrForDay && hrForDay.hasData ? hrForDay.value : null;
    
    return (
      <View style={styles.expandedContainer}>
        <View style={styles.expandedHeader}>
          <Text style={styles.expandedDate}>{selectedDay.fullDate || ''}</Text>
          <Text style={styles.expandedDay}>{selectedDay.dayName || ''}</Text>
        </View>
        
        <View style={styles.stageBarContainer}>
          <View style={styles.stageBar}>
            <View style={[styles.stageSegment, { width: `${deepWidth}%`, backgroundColor: '#3a7c5c' }]} />
            <View style={[styles.stageSegment, { width: `${lightWidth}%`, backgroundColor: '#5a9e7a' }]} />
            <View style={[styles.stageSegment, { width: `${remWidth}%`, backgroundColor: '#7ab89a' }]} />
            <View style={[styles.stageSegment, { width: `${wakeWidth}%`, backgroundColor: '#8a6e5a' }]} />
          </View>
        </View>
        
        <View style={styles.breakdownContainer}>
          <View style={styles.breakdownItem}>
            <View style={[styles.breakdownDot, { backgroundColor: '#3a7c5c' }]} />
            <Text style={styles.breakdownLabel}>Deep Sleep</Text>
            <Text style={styles.breakdownValue}>{(selectedDay.deepHours || 0).toFixed(1)}h</Text>
            <Text style={styles.breakdownPercent}>{deepPercent}%</Text>
          </View>
          <View style={styles.breakdownItem}>
            <View style={[styles.breakdownDot, { backgroundColor: '#5a9e7a' }]} />
            <Text style={styles.breakdownLabel}>Light Sleep</Text>
            <Text style={styles.breakdownValue}>{(selectedDay.lightHours || 0).toFixed(1)}h</Text>
            <Text style={styles.breakdownPercent}>{lightPercent}%</Text>
          </View>
          <View style={styles.breakdownItem}>
            <View style={[styles.breakdownDot, { backgroundColor: '#7ab89a' }]} />
            <Text style={styles.breakdownLabel}>REM Sleep</Text>
            <Text style={styles.breakdownValue}>{(selectedDay.remHours || 0).toFixed(1)}h</Text>
            <Text style={styles.breakdownPercent}>{remPercent}%</Text>
          </View>
          <View style={styles.breakdownItem}>
            <View style={[styles.breakdownDot, { backgroundColor: '#8a6e5a' }]} />
            <Text style={styles.breakdownLabel}>Awake</Text>
            <Text style={styles.breakdownValue}>{(selectedDay.wakeHours || 0).toFixed(1)}h</Text>
            <Text style={styles.breakdownPercent}>{wakePercent}%</Text>
          </View>
        </View>
        
        {(hrvValue || hrValue) ? (
          <View style={styles.vitalsContainer}>
            {hrvValue ? (
              <View style={styles.vitalItem}>
                <Text style={styles.vitalLabel}>HRV</Text>
                <Text style={styles.vitalValue}>{Math.round(hrvValue)} ms</Text>
              </View>
            ) : null}
            {hrValue ? (
              <View style={styles.vitalItem}>
                <Text style={styles.vitalLabel}>Resting HR</Text>
                <Text style={styles.vitalValue}>{Math.round(hrValue)} bpm</Text>
              </View>
            ) : null}
          </View>
        ) : null}
        
        <View style={styles.qualityIndicator}>
          <Text style={styles.qualityIndicatorLabel}>Sleep Efficiency</Text>
          <View style={styles.efficiencyBarContainer}>
            <View style={[styles.efficiencyBar, { width: `${selectedDay.efficiency || 0}%` }]} />
          </View>
          <Text style={styles.qualityIndicatorValue}>{selectedDay.efficiency || 0}%</Text>
        </View>
      </View>
    );
  };

  const sleepQuality = calculateSleepQuality();

  if (loading && !refreshing) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6fc6a8" />
        <Text style={styles.loadingText}>Loading your sleep insights...</Text>
      </View>
    );
  }

  if (!isConnected || (stats.totalNights === 0 && stats.weeklyData.filter(d => d && d.hasData).length === 0 && currentWeekOffset === 0)) {
    return (
      <ScrollView 
        style={styles.container} 
        contentContainerStyle={[
          styles.noDataContainer,
          { paddingTop: getHeaderPadding() }
        ]}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={['#6fc6a8']} />
        }
      >
        <Svg width="100" height="100" viewBox="0 0 24 24" fill="none">
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
        </Svg>
        <Text style={styles.noDataTitle}>No Sleep Data Yet</Text>
        <Text style={styles.noDataText}>
          {!isConnected 
            ? 'Connect your Fitbit account and sync your sleep data to see insights and trends.'
            : 'Sync your Fitbit data to see your sleep insights and trends.'}
        </Text>
        <TouchableOpacity
          style={styles.connectButton}
          onPress={() => navigation.navigate('Analysis')}
        >
          <Text style={styles.connectButtonText}>
            {!isConnected ? 'Connect Fitbit' : 'Go to Analysis'}
          </Text>
        </TouchableOpacity>
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
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={['#6fc6a8']} />
      }
      showsVerticalScrollIndicator={false}
    >
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Sleep Insights</Text>
        <Text style={styles.headerSubtitle}>
          Welcome back, {user?.email?.split('@')[0] || 'User'}
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Overall Sleep Quality</Text>
        <View style={styles.qualityContainer}>
          <View style={styles.qualityScore}>
            <Text style={styles.qualityScoreValue}>{sleepQuality}</Text>
            <Text style={styles.qualityScoreLabel}>/100</Text>
          </View>
          <View style={styles.qualityStats}>
            <View style={styles.qualityStat}>
              <Text style={styles.qualityValue}>{stats.avgSleepHours}h</Text>
              <Text style={styles.qualityLabel}>Avg Sleep</Text>
            </View>
            <View style={styles.qualityStat}>
              <Text style={styles.qualityValue}>{stats.sleepEfficiency}%</Text>
              <Text style={styles.qualityLabel}>Efficiency</Text>
            </View>
            <View style={styles.qualityStat}>
              <Text style={styles.qualityValue}>{stats.totalNights}</Text>
              <Text style={styles.qualityLabel}>Nights</Text>
            </View>
          </View>
        </View>
      </View>

      <View style={styles.card}>
        <View style={styles.cardHeaderWithNav}>
          <Text style={styles.cardTitle}>Weekly Sleep Trend</Text>
          <View style={styles.centerNavButtons}>
            <TouchableOpacity 
              style={[styles.navButton, currentWeekOffset >= maxWeeksBack && styles.navButtonDisabled]} 
              onPress={() => handleWeekChange(1)}
              disabled={currentWeekOffset >= maxWeeksBack}
            >
              <Text style={styles.navButtonText}>←</Text>
            </TouchableOpacity>
            <Text style={styles.weekRange}>{formatWeekRange()}</Text>
            <TouchableOpacity
              style={[styles.navButton, currentWeekOffset === 0 && styles.navButtonDisabled]}
              onPress={() => handleWeekChange(-1)}
              disabled={currentWeekOffset === 0}
            >
              <Text style={styles.navButtonText}>→</Text>
            </TouchableOpacity>
          </View>
        </View>
        
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#3a7c5c' }]} />
            <Text style={styles.legendText}>Deep</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#5a9e7a' }]} />
            <Text style={styles.legendText}>Light</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#7ab89a' }]} />
            <Text style={styles.legendText}>REM</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#8a6e5a' }]} />
            <Text style={styles.legendText}>Wake</Text>
          </View>
        </View>
        
        <StackedBarChart />
        
        {expandedBar !== null ? <ExpandedDayDetails /> : null}
      </View>

      <InteractiveLineChart 
        title="Heart Rate Variability (HRV)"
        data={stats.hrvData}
        unit="ms"
        color="#6fc6a8"
        minValue={stats.hrvMin}
        maxValue={stats.hrvMax}
        note="Higher HRV indicates better recovery and cardiovascular health"
      />

      <InteractiveLineChart 
        title="Resting Heart Rate"
        data={stats.heartRateData}
        unit="bpm"
        color="#5a9e7a"
        minValue={stats.hrMin}
        maxValue={stats.hrMax}
        note="Lower resting heart rate typically indicates better cardiovascular fitness"
      />

      <View style={styles.card}>
        <Text style={styles.cardTitle}>7-Day Averages</Text>
        <View style={styles.averagesGrid}>
          <View style={styles.averageItem}>
            <Text style={styles.averageValue}>{stats.avgSleepHours}h</Text>
            <Text style={styles.averageLabel}>Total Sleep</Text>
          </View>
          <View style={styles.averageItem}>
            <Text style={styles.averageValue}>{stats.avgDeepHours}h</Text>
            <Text style={styles.averageLabel}>Deep Sleep</Text>
          </View>
          <View style={styles.averageItem}>
            <Text style={styles.averageValue}>{stats.avgLightHours}h</Text>
            <Text style={styles.averageLabel}>Light Sleep</Text>
          </View>
          <View style={styles.averageItem}>
            <Text style={styles.averageValue}>{stats.avgRemHours}h</Text>
            <Text style={styles.averageLabel}>REM Sleep</Text>
          </View>
          <View style={styles.averageItem}>
            <Text style={styles.averageValue}>{stats.avgRestingHeartRate}</Text>
            <Text style={styles.averageLabel}>Resting HR</Text>
          </View>
        </View>
      </View>

      <View style={styles.disclaimer}>
        <Text style={styles.disclaimerText}>
          💡 Tap any bar to see detailed sleep stage breakdown. Tap any point on graphs to see exact values.
        </Text>
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
  noDataContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
    paddingBottom: 40,
    minHeight: 500,
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
    marginBottom: 24,
    marginTop: 0,
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
    marginTop: 0,
    marginBottom: 4,
  },
  cardHeaderWithNav: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    flexWrap: 'wrap',
    gap: 12,
  },
  centerNavButtons: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    backgroundColor: '#2a2a2a',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  weekRange: {
    fontSize: 12,
    color: '#95a5a6',
    minWidth: 140,
    textAlign: 'center',
  },
  navButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#3a3a3a',
    borderRadius: 12,
    minWidth: 48,
    alignItems: 'center',
  },
  navButtonDisabled: {
    opacity: 0.4,
  },
  navButtonText: {
    fontSize: 18,
    color: '#f8fbfa',
    fontWeight: '600',
  },
  qualityContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  qualityScore: {
    alignItems: 'center',
    marginRight: 24,
  },
  qualityScoreValue: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#6fc6a8',
  },
  qualityScoreLabel: {
    fontSize: 14,
    color: '#95a5a6',
    marginTop: -8,
  },
  qualityStats: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  qualityStat: {
    alignItems: 'center',
  },
  qualityValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#6fc6a8',
  },
  qualityLabel: {
    fontSize: 12,
    color: '#95a5a6',
    marginTop: 4,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a2a',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  legendText: {
    fontSize: 12,
    color: '#95a5a6',
  },
  chartContainer: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  chartYAxis: {
    width: 35,
    justifyContent: 'space-between',
    height: 140,
    paddingVertical: 4,
  },
  yAxisLabel: {
    fontSize: 9,
    color: '#95a5a6',
    textAlign: 'right',
  },
  chartBars: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
  },
  barWrapper: {
    alignItems: 'center',
    width: BAR_WIDTH,
  },
  barStack: {
    width: BAR_WIDTH - BAR_GAP * 2,
    backgroundColor: '#2a2a2a',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
    position: 'relative',
  },
  barSegment: {
    position: 'absolute',
    left: 0,
    right: 0,
    borderRadius: 2,
  },
  barLabel: {
    fontSize: 11,
    color: '#95a5a6',
  },
  barLabelMuted: {
    color: '#4a4a4a',
  },
  barHourLabel: {
  fontSize: 9,
  color: '#6fc6a8',
  marginTop: 4,
  textAlign: 'center',
},
  chartCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: 16,
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f8fbfa',
  },
  chartValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  svgContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  xAxisLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
    paddingHorizontal: 35,
  },
  xAxisLabel: {
    fontSize: 10,
    color: '#95a5a6',
    textAlign: 'center',
    width: (LINE_CHART_WIDTH - 45) / 7,
  },
  xAxisLabelMuted: {
    color: '#4a4a4a',
  },
  chartNote: {
    marginTop: 12,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  chartNoteText: {
    fontSize: 10,
    color: '#95a5a6',
    textAlign: 'center',
  },
  chartNoteSubtext: {
    fontSize: 9,
    color: '#6fc6a8',
    textAlign: 'center',
    marginTop: 4,
  },
  expandedContainer: {
    marginTop: 20,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#2a2a2a',
  },
  expandedHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  expandedDate: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f8fbfa',
  },
  expandedDay: {
    fontSize: 14,
    color: '#6fc6a8',
  },
  stageBarContainer: {
    marginBottom: 20,
  },
  stageBar: {
    flexDirection: 'row',
    height: 24,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#2a2a2a',
  },
  stageSegment: {
    height: '100%',
  },
  breakdownContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    marginBottom: 20,
  },
  breakdownItem: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '45%',
    gap: 8,
  },
  breakdownDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  breakdownLabel: {
    fontSize: 12,
    color: '#95a5a6',
    flex: 1,
  },
  breakdownValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#f8fbfa',
  },
  breakdownPercent: {
    fontSize: 12,
    color: '#6fc6a8',
  },
  vitalsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
    paddingVertical: 12,
    backgroundColor: '#0f0f0f',
    borderRadius: 12,
  },
  vitalItem: {
    alignItems: 'center',
  },
  vitalLabel: {
    fontSize: 11,
    color: '#95a5a6',
    marginBottom: 4,
  },
  vitalValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f8fbfa',
  },
  qualityIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  qualityIndicatorLabel: {
    fontSize: 12,
    color: '#95a5a6',
  },
  efficiencyBarContainer: {
    flex: 1,
    height: 6,
    backgroundColor: '#2a2a2a',
    borderRadius: 3,
    overflow: 'hidden',
  },
  efficiencyBar: {
    height: '100%',
    backgroundColor: '#6fc6a8',
    borderRadius: 3,
  },
  qualityIndicatorValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#f8fbfa',
  },
  averagesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  averageItem: {
    width: '48%',
    backgroundColor: '#0f0f0f',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginBottom: 12,
  },
  averageValue: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#6fc6a8',
  },
  averageLabel: {
    fontSize: 12,
    color: '#95a5a6',
    marginTop: 6,
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
    lineHeight: 20,
  },
  noDataSmallContainer: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 40,
    marginBottom: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  noDataSmallText: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
  },
  noDataChartContainer: {
    height: 180,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0f0f0f',
    borderRadius: 12,
    marginBottom: 16,
  },
  noDataChartText: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
    marginBottom: 8,
  },
  noDataChartSubtext: {
    fontSize: 12,
    color: '#6fc6a8',
    textAlign: 'center',
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
});