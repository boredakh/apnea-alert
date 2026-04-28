// src/services/exportService.js
import { Platform, Share, Alert, Clipboard } from 'react-native';
import { supabase } from './supabase';

// Helper function to safely parse JSON strings
const safeParseJSON = (data) => {
  if (!data) return null;
  if (typeof data === 'object') return data;
  try {
    return JSON.parse(data);
  } catch (e) {
    console.error('JSON parse error:', e);
    return null;
  }
};

// Format sleep data for human readability
const formatSleepData = (sleepDataRaw) => {
  const sleepData = safeParseJSON(sleepDataRaw);
  if (!sleepData || sleepData.length === 0) return null;
  
  const mainSleep = sleepData.find(s => s.isMainSleep === true) || sleepData[0];
  if (!mainSleep) return null;
  
  return {
    date: mainSleep.dateOfSleep,
    duration_hours: (mainSleep.minutesAsleep / 60).toFixed(1),
    duration_minutes: mainSleep.minutesAsleep,
    quality_score: mainSleep.efficiency,
    stages: {
      deep_minutes: mainSleep.levels?.summary?.deep?.minutes || 0,
      deep_hours: ((mainSleep.levels?.summary?.deep?.minutes || 0) / 60).toFixed(1),
      light_minutes: mainSleep.levels?.summary?.light?.minutes || 0,
      light_hours: ((mainSleep.levels?.summary?.light?.minutes || 0) / 60).toFixed(1),
      rem_minutes: mainSleep.levels?.summary?.rem?.minutes || 0,
      rem_hours: ((mainSleep.levels?.summary?.rem?.minutes || 0) / 60).toFixed(1),
      awake_minutes: mainSleep.minutesAwake || 0,
      awake_hours: ((mainSleep.minutesAwake || 0) / 60).toFixed(1),
    },
    time_in_bed_minutes: mainSleep.timeInBed,
    bedtime: mainSleep.startTime,
    wakeup: mainSleep.endTime,
  };
};

// Format heart rate data for human readability
const formatHeartRateData = (heartRateDataRaw) => {
  const heartRateData = safeParseJSON(heartRateDataRaw);
  if (!heartRateData || heartRateData.length === 0) return null;
  
  const hr = heartRateData[0]?.value;
  return {
    resting_heart_rate: hr?.restingHeartRate || null,
    heart_rate_zones: hr?.heartRateZones?.map(zone => ({
      name: zone.name,
      minutes: zone.minutes,
      calories: zone.caloriesOut?.toFixed(0) || 0,
      range: `${zone.min}-${zone.max} bpm`
    })) || [],
  };
};

// Format HRV data for human readability
const formatHRVData = (hrvDataRaw) => {
  const hrvData = safeParseJSON(hrvDataRaw);
  if (!hrvData?.hrv?.[0]?.minutes) return null;
  
  const minutes = hrvData.hrv[0].minutes;
  const validReadings = minutes.filter(m => m.value?.rmssd);
  
  if (validReadings.length === 0) return null;
  
  const rmssdValues = validReadings.map(m => m.value.rmssd);
  const avgRMSSD = rmssdValues.reduce((a, b) => a + b, 0) / rmssdValues.length;
  
  return {
    average_rmssd: Math.round(avgRMSSD),
    readings_count: validReadings.length,
    interpretation: avgRMSSD > 50 ? 'Good' : avgRMSSD > 30 ? 'Fair' : 'Low',
  };
};

// Fetch user data for export
export const fetchUserDataForExport = async (userId, dateRange = null) => {
  console.log('📊 Fetching export data for user:', userId);
  
  let query = supabase
    .from('fitbit_data')
    .select('*')
    .eq('user_id', userId)
    .order('date', { ascending: false });
  
  if (dateRange) {
    if (dateRange.startDate) query = query.gte('date', dateRange.startDate);
    if (dateRange.endDate) query = query.lte('date', dateRange.endDate);
  }
  
  const { data, error } = await query;
  
  if (error) {
    console.error('❌ Error fetching fitbit data:', error);
    throw error;
  }
  
  if (!data || data.length === 0) {
    console.log('⚠️ No fitbit data found for user');
    return [];
  }
  
  console.log(`✅ Found ${data.length} records to export`);
  
  const formattedData = data.map(record => {
    const formatted = {
      date: record.date,
      sleep: formatSleepData(record.sleep_data),
      heart_rate: formatHeartRateData(record.heart_rate_data),
      hrv: formatHRVData(record.hrv_data),
      has_sleep_data: !!record.sleep_data,
    };
    return formatted;
  });
  
  const filteredData = formattedData.filter(d => d.has_sleep_data);
  console.log(`📊 After filtering: ${filteredData.length} records with sleep data`);
  
  return filteredData;
};

// Generate beautifully formatted human-readable text
const generateHumanReadableText = (data, userName, dateRange) => {
  const totalNights = data.length;
  const avgSleep = totalNights > 0 ? data.reduce((sum, d) => sum + (parseFloat(d.sleep?.duration_hours) || 0), 0) / totalNights : 0;
  const avgQuality = totalNights > 0 ? data.reduce((sum, d) => sum + (d.sleep?.quality_score || 0), 0) / totalNights : 0;
  const avgHRV = totalNights > 0 ? data.reduce((sum, d) => sum + (d.hrv?.average_rmssd || 0), 0) / totalNights : 0;
  
  let text = `========================================
     APNEAALERT SLEEP REPORT
========================================

Report Type: HUMAN READABLE TEXT
Generated for: ${userName || 'User'}
Date Range: ${dateRange || 'All available data'}
Generated on: ${new Date().toLocaleString()}
Total Nights Analyzed: ${totalNights}

========================================
           SUMMARY STATISTICS
========================================

Average Sleep Duration:    ${avgSleep.toFixed(1)} hours
Average Sleep Quality:     ${Math.round(avgQuality)}%
Average HRV (RMSSD):       ${Math.round(avgHRV)} ms
${avgHRV > 50 ? '✅ HRV Status: Good (Healthy nervous system)' : avgHRV > 30 ? '⚠️ HRV Status: Fair (Monitor your sleep)' : '❌ HRV Status: Low (Consider lifestyle changes)'}

========================================
         DETAILED SLEEP DATA
========================================

`;
  
  data.forEach((day, index) => {
    const sleepQuality = day.sleep?.quality_score || 'N/A';
    const sleepQualityEmoji = sleepQuality >= 85 ? '😊' : sleepQuality >= 70 ? '😐' : '😴';
    
    text += `📅 Date: ${day.date}
────────────────────────────────────────
${sleepQualityEmoji} Sleep Quality: ${sleepQuality}%
⏰ Total Sleep: ${day.sleep?.duration_hours || '0'} hours (${day.sleep?.duration_minutes || 0} minutes)
🛌 Time in Bed: ${day.sleep?.time_in_bed_minutes || 0} minutes

Sleep Stages:
  • Deep Sleep:  ${day.sleep?.stages.deep_hours || '0'} hours (${day.sleep?.stages.deep_minutes || 0} min)
  • Light Sleep: ${day.sleep?.stages.light_hours || '0'} hours (${day.sleep?.stages.light_minutes || 0} min)
  • REM Sleep:   ${day.sleep?.stages.rem_hours || '0'} hours (${day.sleep?.stages.rem_minutes || 0} min)
  • Awake:       ${day.sleep?.stages.awake_hours || '0'} hours (${day.sleep?.stages.awake_minutes || 0} min)

`;
    
    if (day.heart_rate?.resting_heart_rate || day.hrv?.average_rmssd) {
      text += `Health Metrics:
`;
      if (day.heart_rate?.resting_heart_rate) {
        const hrEmoji = day.heart_rate.resting_heart_rate < 70 ? '💚' : day.heart_rate.resting_heart_rate < 85 ? '💛' : '❤️';
        text += `  ${hrEmoji} Resting Heart Rate: ${day.heart_rate.resting_heart_rate} bpm
`;
      }
      if (day.hrv?.average_rmssd) {
        const hrvEmoji = day.hrv.average_rmssd > 50 ? '💪' : day.hrv.average_rmssd > 30 ? '👍' : '⚠️';
        text += `  ${hrvEmoji} HRV (RMSSD): ${day.hrv.average_rmssd} ms
`;
        text += `  📊 HRV Status: ${day.hrv.interpretation}
`;
      }
    }
    
    text += `
Bedtime: ${day.sleep?.bedtime ? new Date(day.sleep.bedtime).toLocaleTimeString() : 'N/A'}
Wakeup:  ${day.sleep?.wakeup ? new Date(day.sleep.wakeup).toLocaleTimeString() : 'N/A'}

`;
    
    if (index < data.length - 1) {
      text += `────────────────────────────────────────
`;
    }
  });
  
  text += `
========================================
              LEGEND
========================================

Sleep Quality:
  😊 85-100% - Excellent
  😐 70-84%  - Fair
  😴 <70%    - Needs improvement

Sleep Stages:
  • Deep Sleep: Physical restoration
  • Light Sleep: Mental processing
  • REM Sleep: Memory consolidation
  • Awake: Time spent awake in bed

HRV (Heart Rate Variability):
  💪 >50 ms  - Optimal (Good recovery)
  👍 30-50 ms - Fair (Monitor trends)
  ⚠️ <30 ms  - Low (Consider lifestyle changes)

Resting Heart Rate:
  💚 <70 bpm - Excellent
  💛 70-85 bpm - Fair
  ❤️ >85 bpm - Monitor

========================================
           HEALTH DISCLAIMER
========================================

This report is for informational purposes only.
It is not medical advice. Consult your healthcare
provider for medical decisions.

ApneaAlert - Tracking your sleep health journey
========================================`;

  return text;
};

// Generate CSV
const generateCSV = (data) => {
  const headers = [
    'Date',
    'Sleep Duration (hours)',
    'Sleep Quality (%)',
    'Deep Sleep (hours)',
    'Light Sleep (hours)',
    'REM Sleep (hours)',
    'Awake (hours)',
    'Resting Heart Rate (bpm)',
    'Avg HRV (ms)',
    'HRV Status',
  ];
  
  const escapeCSV = (value) => {
    if (value === undefined || value === null) return '';
    const stringValue = String(value);
    if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
      return `"${stringValue.replace(/"/g, '""')}"`;
    }
    return stringValue;
  };
  
  const rows = data.map(day => [
    escapeCSV(day.date),
    escapeCSV(day.sleep?.duration_hours),
    escapeCSV(day.sleep?.quality_score),
    escapeCSV(day.sleep?.stages.deep_hours),
    escapeCSV(day.sleep?.stages.light_hours),
    escapeCSV(day.sleep?.stages.rem_hours),
    escapeCSV(day.sleep?.stages.awake_hours),
    escapeCSV(day.heart_rate?.resting_heart_rate),
    escapeCSV(day.hrv?.average_rmssd),
    escapeCSV(day.hrv?.interpretation),
  ]);
  
  return [headers, ...rows].map(row => row.join(',')).join('\n');
};

// Generate JSON
const generateJSON = (data) => {
  return JSON.stringify(data, null, 2);
};

// Generate HTML report
const generateHTML = (data, userName, dateRange) => {
  const totalNights = data.length;
  const avgSleep = totalNights > 0 ? data.reduce((sum, d) => sum + (parseFloat(d.sleep?.duration_hours) || 0), 0) / totalNights : 0;
  const avgQuality = totalNights > 0 ? data.reduce((sum, d) => sum + (d.sleep?.quality_score || 0), 0) / totalNights : 0;
  const avgHRV = totalNights > 0 ? data.reduce((sum, d) => sum + (d.hrv?.average_rmssd || 0), 0) / totalNights : 0;
  
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>ApneaAlert - Sleep Report</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: #0f0f0f;
      color: #f8fbfa;
      padding: 40px 20px;
    }
    .container { max-width: 800px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #2a2a2a; }
    .header h1 { font-size: 32px; color: #6fc6a8; margin-bottom: 8px; }
    .header p { color: #95a5a6; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin-bottom: 40px; }
    .summary-card { background: #1a1a1a; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #2a2a2a; }
    .summary-card h3 { font-size: 14px; color: #95a5a6; margin-bottom: 8px; }
    .summary-card .value { font-size: 32px; font-weight: bold; color: #6fc6a8; }
    .summary-card .unit { font-size: 14px; color: #95a5a6; }
    .day-card { background: #1a1a1a; border-radius: 12px; padding: 20px; margin-bottom: 16px; border: 1px solid #2a2a2a; }
    .day-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2a2a2a; }
    .day-date { font-size: 18px; font-weight: 600; color: #6fc6a8; }
    .day-quality { background: rgba(111, 198, 168, 0.15); padding: 4px 12px; border-radius: 20px; font-size: 14px; }
    .sleep-stages { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
    .stage { flex: 1; background: #0f0f0f; border-radius: 8px; padding: 12px; text-align: center; }
    .stage-name { font-size: 12px; color: #95a5a6; margin-bottom: 4px; }
    .stage-value { font-size: 20px; font-weight: 600; color: #f8fbfa; }
    .stage-unit { font-size: 10px; color: #95a5a6; }
    .vitals { display: flex; gap: 16px; margin-top: 16px; padding-top: 16px; border-top: 1px solid #2a2a2a; }
    .vital { flex: 1; text-align: center; }
    .vital-label { font-size: 12px; color: #95a5a6; margin-bottom: 4px; }
    .vital-value { font-size: 18px; font-weight: 600; color: #6fc6a8; }
    .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #2a2a2a; color: #95a5a6; font-size: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>ApneaAlert Sleep Report</h1>
      <p>Generated for ${userName || 'User'}</p>
      <p>${dateRange || 'All available data'} | ${totalNights} nights analyzed</p>
    </div>
    
    <div class="summary">
      <div class="summary-card">
        <h3>Nights Analyzed</h3>
        <div class="value">${totalNights}</div>
      </div>
      <div class="summary-card">
        <h3>Average Sleep</h3>
        <div class="value">${avgSleep.toFixed(1)}<span class="unit">h</span></div>
      </div>
      <div class="summary-card">
        <h3>Average Quality</h3>
        <div class="value">${Math.round(avgQuality)}<span class="unit">%</span></div>
      </div>
      <div class="summary-card">
        <h3>Average HRV</h3>
        <div class="value">${Math.round(avgHRV)}<span class="unit">ms</span></div>
      </div>
    </div>
    
    ${data.map(day => `
      <div class="day-card">
        <div class="day-header">
          <span class="day-date">${day.date}</span>
          <span class="day-quality">Quality: ${day.sleep?.quality_score || 'N/A'}%</span>
        </div>
        <div class="sleep-stages">
          <div class="stage">
            <div class="stage-name">Total Sleep</div>
            <div class="stage-value">${day.sleep?.duration_hours || '0'}<span class="stage-unit">h</span></div>
          </div>
          <div class="stage">
            <div class="stage-name">Deep Sleep</div>
            <div class="stage-value">${day.sleep?.stages.deep_hours || '0'}<span class="stage-unit">h</span></div>
          </div>
          <div class="stage">
            <div class="stage-name">Light Sleep</div>
            <div class="stage-value">${day.sleep?.stages.light_hours || '0'}<span class="stage-unit">h</span></div>
          </div>
          <div class="stage">
            <div class="stage-name">REM Sleep</div>
            <div class="stage-value">${day.sleep?.stages.rem_hours || '0'}<span class="stage-unit">h</span></div>
          </div>
        </div>
        ${(day.heart_rate?.resting_heart_rate || day.hrv?.average_rmssd) ? `
          <div class="vitals">
            ${day.heart_rate?.resting_heart_rate ? `
              <div class="vital">
                <div class="vital-label">Resting HR</div>
                <div class="vital-value">${day.heart_rate.resting_heart_rate} bpm</div>
              </div>
            ` : ''}
            ${day.hrv?.average_rmssd ? `
              <div class="vital">
                <div class="vital-label">HRV (RMSSD)</div>
                <div class="vital-value">${day.hrv.average_rmssd} ms</div>
              </div>
            ` : ''}
          </div>
        ` : ''}
      </div>
    `).join('')}
    
    <div class="footer">
      <p>This report was generated by ApneaAlert. Data is for informational purposes only.</p>
      <p>Generated on ${new Date().toLocaleString()}</p>
    </div>
  </div>
</body>
</html>`;
};

// Main export function - WITH ALERT LIKE THE WORKING VERSION
export const exportData = async (format, data, userName, dateRange) => {
  if (!data || data.length === 0) {
    throw new Error('No data available to export');
  }
  
  let content = '';
  let filename = `ApneaAlert_Sleep_Report_${new Date().toISOString().split('T')[0]}`;
  
  console.log(`📁 Exporting data as ${format.toUpperCase()}...`);
  
  switch (format) {
    case 'txt':
      content = generateHumanReadableText(data, userName, dateRange);
      filename += '.txt';
      break;
    case 'csv':
      content = generateCSV(data);
      filename += '.csv';
      break;
    case 'json':
      content = generateJSON(data);
      filename += '.json';
      break;
    case 'html':
      content = generateHTML(data, userName, dateRange);
      filename += '.html';
      break;
    default:
      throw new Error('Unsupported format');
  }
  
  // For web: download directly
  if (Platform.OS === 'web') {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 100);
    return;
  }
  
  // For iOS and Android: Show Alert with options (THIS IS WHAT WORKED BEFORE)
  return new Promise((resolve, reject) => {
    Alert.alert(
      'Export Data',
      `Your ${format.toUpperCase()} report is ready. How would you like to save it?`,
      [
        {
          text: 'Copy to Clipboard',
          onPress: async () => {
            try {
              await Clipboard.setString(content);
              Alert.alert('Copied!', 'The data has been copied to your clipboard. You can now paste it into Notes, Email, or any other app to save it.');
              resolve(true);
            } catch (error) {
              reject(error);
            }
          }
        },
        {
          text: 'Save & Share',
          onPress: async () => {
            try {
              await Share.share({
                title: filename,
                message: content,
              });
              resolve(true);
            } catch (error) {
              reject(error);
            }
          }
        },
        {
          text: 'Cancel',
          style: 'cancel',
          onPress: () => reject(new Error('User cancelled'))
        }
      ],
      { cancelable: true }
    );
  });
};