// src/services/sleepAIService.js
import { supabase } from './supabase';

// Format sleep data for analysis
const formatSleepDataForAI = (sleepData, heartRateData, hrvData) => {
  if (!sleepData || sleepData.length === 0) return null;
  
  const sleep = sleepData[0];
  
  if (!sleep.minutesAsleep || sleep.minutesAsleep === 0) return null;
  
  const hr = heartRateData?.[0]?.value;
  
  const deepMinutes = sleep.levels?.summary?.deep?.minutes || 0;
  const lightMinutes = sleep.levels?.summary?.light?.minutes || 0;
  const remMinutes = sleep.levels?.summary?.rem?.minutes || 0;
  const awakeMinutes = sleep.minutesAwake || 0;
  const totalMinutes = sleep.minutesAsleep;
  
  const deepPercent = totalMinutes > 0 ? (deepMinutes / totalMinutes * 100).toFixed(1) : 0;
  const remPercent = totalMinutes > 0 ? (remMinutes / totalMinutes * 100).toFixed(1) : 0;
  const lightPercent = totalMinutes > 0 ? (lightMinutes / totalMinutes * 100).toFixed(1) : 0;
  const awakePercent = totalMinutes > 0 ? (awakeMinutes / totalMinutes * 100).toFixed(1) : 0;
  
  // Extract HRV data for each minute
  let hrvDataPoints = [];
  let avgHRV = null;
  let hrvVariability = null;
  if (hrvData?.hrv?.[0]?.minutes) {
    const minutes = hrvData.hrv[0].minutes;
    const validReadings = minutes.filter(m => m.value?.rmssd);
    if (validReadings.length > 0) {
      const rmssdValues = validReadings.map(m => m.value.rmssd);
      avgHRV = Math.round(rmssdValues.reduce((a, b) => a + b, 0) / rmssdValues.length);
      // Calculate HRV variability (standard deviation approximation)
      const variance = rmssdValues.reduce((sum, val) => sum + Math.pow(val - avgHRV, 2), 0) / rmssdValues.length;
      hrvVariability = Math.round(Math.sqrt(variance));
      hrvDataPoints = rmssdValues;
    }
  }
  
  const formatTime = (isoString) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    let hours = date.getHours();
    const minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12;
    hours = hours ? hours : 12;
    return `${hours}:${minutes.toString().padStart(2, '0')} ${ampm}`;
  };
  
  // Calculate bedtime hour for circadian analysis
  const bedtimeHour = sleep.startTime ? new Date(sleep.startTime).getHours() : null;
  const wakeupHour = sleep.endTime ? new Date(sleep.endTime).getHours() : null;
  
  return {
    date: sleep.dateOfSleep,
    bedtime: formatTime(sleep.startTime),
    wakeup: formatTime(sleep.endTime),
    bedtime_hour: bedtimeHour,
    wakeup_hour: wakeupHour,
    duration_hours: (sleep.minutesAsleep / 60).toFixed(1),
    duration_minutes: sleep.minutesAsleep,
    quality_score: sleep.efficiency || 0,
    time_in_bed_minutes: sleep.timeInBed,
    sleep_efficiency: ((sleep.minutesAsleep / sleep.timeInBed) * 100).toFixed(1),
    stages: {
      deep: { minutes: deepMinutes, percent: deepPercent },
      light: { minutes: lightMinutes, percent: lightPercent },
      rem: { minutes: remMinutes, percent: remPercent },
      awake: { minutes: awakeMinutes, percent: awakePercent },
    },
    resting_heart_rate: hr?.restingHeartRate || null,
    hrv: avgHRV,
    hrv_variability: hrvVariability,
    hrv_data_points: hrvDataPoints,
  };
};

// Get the most recent sleep date
const getMostRecentSleepDate = async (userId) => {
  const { data, error } = await supabase
    .from('fitbit_data')
    .select('date, sleep_data')
    .eq('user_id', userId)
    .not('sleep_data', 'is', null)
    .order('date', { ascending: false });
  
  if (error) throw error;
  
  for (const record of data) {
    const sleep = record.sleep_data?.[0];
    if (sleep && sleep.minutesAsleep > 0) {
      return record.date;
    }
  }
  
  return null;
};

// Get comprehensive historical data
const getHistoricalAnalysis = async (userId, days = 30) => {
  const { data, error } = await supabase
    .from('fitbit_data')
    .select('*')
    .eq('user_id', userId)
    .not('sleep_data', 'is', null)
    .order('date', { ascending: false })
    .limit(days);
  
  if (error) throw error;
  
  const dailyMetrics = [];
  let totalSleep = 0;
  let totalDeep = 0;
  let totalREM = 0;
  let totalLight = 0;
  let totalEfficiency = 0;
  let totalHRV = [];
  let bedtimes = [];
  let validDays = 0;
  let weekendDays = 0;
  let weekdayDays = 0;
  let weekendSleep = 0;
  let weekdaySleep = 0;
  
  for (const record of data) {
    const sleep = record.sleep_data?.[0];
    if (sleep && sleep.minutesAsleep > 0) {
      validDays++;
      const date = new Date(record.date);
      const isWeekend = date.getDay() === 0 || date.getDay() === 6;
      const sleepHours = sleep.minutesAsleep / 60;
      
      totalSleep += sleepHours;
      totalDeep += sleep.levels?.summary?.deep?.minutes || 0;
      totalREM += sleep.levels?.summary?.rem?.minutes || 0;
      totalLight += sleep.levels?.summary?.light?.minutes || 0;
      totalEfficiency += sleep.efficiency || 0;
      
      if (isWeekend) {
        weekendDays++;
        weekendSleep += sleepHours;
      } else {
        weekdayDays++;
        weekdaySleep += sleepHours;
      }
      
      if (sleep.startTime) {
        const hour = new Date(sleep.startTime).getHours();
        bedtimes.push(hour);
      }
      
      dailyMetrics.push({
        date: record.date,
        duration: sleepHours,
        deep: sleep.levels?.summary?.deep?.minutes || 0,
        rem: sleep.levels?.summary?.rem?.minutes || 0,
        efficiency: sleep.efficiency || 0,
        bedtime_hour: sleep.startTime ? new Date(sleep.startTime).getHours() : null,
        isWeekend,
      });
      
      if (record.hrv_data?.hrv?.[0]?.minutes) {
        const minutes = record.hrv_data.hrv[0].minutes;
        const validReadings = minutes.filter(m => m.value?.rmssd);
        if (validReadings.length > 0) {
          const avg = validReadings.reduce((sum, m) => sum + m.value.rmssd, 0) / validReadings.length;
          totalHRV.push(avg);
        }
      }
    }
  }
  
  if (validDays === 0) return null;
  
  // Calculate sleep timing consistency (standard deviation of bedtimes)
  const avgBedtime = bedtimes.reduce((a, b) => a + b, 0) / bedtimes.length;
  const bedtimeStdDev = Math.sqrt(bedtimes.reduce((sum, hour) => sum + Math.pow(hour - avgBedtime, 2), 0) / bedtimes.length);
  
  // Calculate sleep duration distribution
  const allDurations = dailyMetrics.map(m => m.duration);
  const avgDuration = totalSleep / validDays;
  const durationStdDev = Math.sqrt(allDurations.reduce((sum, d) => sum + Math.pow(d - avgDuration, 2), 0) / allDurations.length);
  
  // Detect outliers
  const outliers = dailyMetrics.filter(m => 
    Math.abs(m.duration - avgDuration) > 2 * durationStdDev || 
    (m.deep < 15 && m.rem < 15)
  );
  
  return {
    days_analyzed: validDays,
    avg_sleep_hours: (totalSleep / validDays).toFixed(1),
    avg_deep_minutes: Math.round(totalDeep / validDays),
    avg_rem_minutes: Math.round(totalREM / validDays),
    avg_light_minutes: Math.round(totalLight / validDays),
    avg_deep_percent: totalDeep > 0 ? ((totalDeep / (totalSleep * 60)) * 100).toFixed(1) : 0,
    avg_rem_percent: totalREM > 0 ? ((totalREM / (totalSleep * 60)) * 100).toFixed(1) : 0,
    avg_light_percent: totalLight > 0 ? ((totalLight / (totalSleep * 60)) * 100).toFixed(1) : 0,
    avg_efficiency: Math.round(totalEfficiency / validDays),
    avg_hrv: totalHRV.length > 0 ? Math.round(totalHRV.reduce((a, b) => a + b, 0) / totalHRV.length) : null,
    consistency_score: bedtimeStdDev < 1 ? 'Excellent' : bedtimeStdDev < 2 ? 'Good' : bedtimeStdDev < 3 ? 'Fair' : 'Poor',
    bedtime_consistency_hours: bedtimeStdDev.toFixed(1),
    weekday_sleep: weekdayDays > 0 ? (weekdaySleep / weekdayDays).toFixed(1) : null,
    weekend_sleep: weekendDays > 0 ? (weekendSleep / weekendDays).toFixed(1) : null,
    sleep_social_jetlag: (weekendDays > 0 && weekdayDays > 0) ? Math.abs((weekendSleep / weekendDays) - (weekdaySleep / weekdayDays)).toFixed(1) : null,
    outlier_days: outliers.length,
    total_days: validDays,
    recent_trend: dailyMetrics.slice(0, 7).map(m => m.duration),
  };
};

// Age-specific sleep recommendations (based on NSF guidelines)
const getAgeSpecificInsights = (age, gender) => {
  const ageGroups = {
    teen: { min: 8, max: 10, range: '14-17', deepTarget: 20, remTarget: 22 },
    youngAdult: { min: 7, max: 9, range: '18-25', deepTarget: 18, remTarget: 22 },
    adult: { min: 7, max: 9, range: '26-64', deepTarget: 18, remTarget: 22 },
    senior: { min: 7, max: 8, range: '65+', deepTarget: 15, remTarget: 20 },
  };
  
  let group;
  if (!age) return { min: 7, max: 9, range: '18-64', deepTarget: 18, remTarget: 22 };
  if (age < 18) group = ageGroups.teen;
  else if (age <= 25) group = ageGroups.youngAdult;
  else if (age <= 64) group = ageGroups.adult;
  else group = ageGroups.senior;
  
  return {
    ...group,
    age_actual: age,
    gender,
    recommended_hours: `${group.min}-${group.max} hours`,
    deep_expectation: `Aim for ${group.deepTarget}% of total sleep in deep sleep`,
    rem_expectation: `Aim for ${group.remTarget}% of total sleep in REM sleep`,
  };
};

// BMI calculation and analysis
const calculateBMI = (heightCm, weightKg) => {
  if (!heightCm || !weightKg) return null;
  const heightM = heightCm / 100;
  const bmi = weightKg / (heightM * heightM);
  let category = '';
  if (bmi < 18.5) category = 'Underweight';
  else if (bmi < 25) category = 'Normal weight';
  else if (bmi < 30) category = 'Overweight';
  else category = 'Obese';
  return { bmi: bmi.toFixed(1), category };
};

// HRV analysis
const analyzeHRV = (hrvValue, hrvVariability, age) => {
  if (!hrvValue) return null;
  
  let interpretation = '';
  if (hrvValue > 50) interpretation = 'Excellent';
  else if (hrvValue > 40) interpretation = 'Good';
  else if (hrvValue > 30) interpretation = 'Fair';
  else interpretation = 'Low';
  
  let ageAdjusted = '';
  if (age) {
    if (age < 30 && hrvValue < 40) ageAdjusted = 'Below average for your age group';
    else if (age < 30 && hrvValue > 55) ageAdjusted = 'Excellent for your age group';
    else if (age > 50 && hrvValue > 35) ageAdjusted = 'Above average for your age group';
    else if (age > 50 && hrvValue < 25) ageAdjusted = 'Below average for your age group';
  }
  
  return {
    value: hrvValue,
    interpretation,
    variability: hrvVariability,
    age_adjusted: ageAdjusted,
  };
};

// Outlier detection
const detectOutliers = (sleepData, trends, userProfile) => {
  const insights = [];
  const avgDuration = parseFloat(trends.avg_sleep_hours);
  const currentDuration = parseFloat(sleepData.duration_hours);
  const durationDiff = Math.abs(currentDuration - avgDuration);
  
  if (durationDiff > 2) {
    if (currentDuration > avgDuration) {
      insights.push(`📊 This is a recovery sleep night (${currentDuration}h vs your average ${avgDuration}h). Your body needed extra rest - listen to it!`);
    } else {
      insights.push(`⚠️ Significantly shorter sleep than usual (${currentDuration}h vs your average ${avgDuration}h). Consider what might have caused this disruption.`);
    }
  }
  
  const avgDeep = trends.avg_deep_minutes;
  const currentDeep = sleepData.stages.deep.minutes;
  if (Math.abs(currentDeep - avgDeep) > avgDeep * 0.5 && avgDeep > 0) {
    if (currentDeep > avgDeep) {
      insights.push(`💪 Exceptional deep sleep! (${currentDeep}min vs your average ${avgDeep}min). Your physical recovery is prioritized tonight.`);
    } else {
      insights.push(`⚠️ Low deep sleep tonight (${currentDeep}min vs your average ${avgDeep}min). Stress, caffeine, or late exercise may be factors.`);
    }
  }
  
  const avgREM = trends.avg_rem_minutes;
  const currentREM = sleepData.stages.rem.minutes;
  if (Math.abs(currentREM - avgREM) > avgREM * 0.5 && avgREM > 0) {
    if (currentREM > avgREM) {
      insights.push(`🧠 High REM sleep (${currentREM}min vs your average ${avgREM}min). Your brain is processing memories and emotions intensely.`);
    } else {
      insights.push(`⚠️ Low REM sleep tonight. Try maintaining a consistent wake-up time to support REM cycles.`);
    }
  }
  
  const bedtimeDiff = Math.abs(sleepData.bedtime_hour - (trends.recent_bedtime_avg || 23));
  if (bedtimeDiff > 2) {
    insights.push(`⏰ Unusual bedtime (${sleepData.bedtime} vs your typical schedule). Circadian disruption can affect sleep quality.`);
  }
  
  if (sleepData.hrv && trends.avg_hrv) {
    const hrvDiff = Math.abs(sleepData.hrv - trends.avg_hrv);
    if (hrvDiff > 15) {
      if (sleepData.hrv > trends.avg_hrv) {
        insights.push(`💚 Elevated HRV (${sleepData.hrv}ms vs your average ${trends.avg_hrv}ms). Great recovery and low stress!`);
      } else {
        insights.push(`❤️ Reduced HRV (${sleepData.hrv}ms vs your average ${trends.avg_hrv}ms). Your nervous system may be under stress.`);
      }
    }
  }
  
  return insights;
};

// Circadian rhythm analysis
const analyzeCircadianRhythm = (sleepData, trends) => {
  const insights = [];
  const bedtimeHour = sleepData.bedtime_hour;
  
  if (bedtimeHour >= 24 || bedtimeHour <= 2) {
    insights.push(`🌙 Very late bedtime (${bedtimeHour === 0 ? 12 : bedtimeHour % 12}:${sleepData.bedtime.includes(':') ? sleepData.bedtime.split(':')[1] : '00'} ${bedtimeHour >= 12 ? 'AM' : 'AM'}). Late sleep onset can disrupt natural cortisol rhythms.`);
  } else if (bedtimeHour >= 22 && bedtimeHour <= 23) {
    insights.push(`✅ Optimal bedtime window (${sleepData.bedtime}) aligns with your circadian rhythm for best sleep quality.`);
  } else if (bedtimeHour <= 21) {
    insights.push(`🌅 Early bedtime detected. While sufficient sleep is great, extremely early bedtimes may indicate exhaustion.`);
  }
  
  if (trends.sleep_social_jetlag && parseFloat(trends.sleep_social_jetlag) > 1.5) {
    insights.push(`⚠️ Social jetlag detected: You sleep ${trends.sleep_social_jetlag}h longer on weekends. This can cause "Monday morning fatigue." Try keeping weekend wake times within 1 hour of weekdays.`);
  } else if (trends.sleep_social_jetlag && parseFloat(trends.sleep_social_jetlag) > 0.5) {
    insights.push(`📅 Small weekend sleep shift (${trends.sleep_social_jetlag}h). Consider narrowing this gap for better consistency.`);
  }
  
  if (trends.bedtime_consistency_hours && parseFloat(trends.bedtime_consistency_hours) < 1) {
    insights.push(`🎯 Exceptional bedtime consistency! Your body's internal clock is well-regulated.`);
  }
  
  return insights;
};

// Deep sleep physiology insights
const analyzeDeepSleep = (deepMinutes, deepPercent, age, totalSleepHours) => {
  const insights = [];
  const ageGuidelines = getAgeSpecificInsights(age);
  const targetPercent = ageGuidelines.deepTarget;
  const isLow = deepPercent < targetPercent - 5;
  const isHigh = deepPercent > targetPercent + 10;
  
  if (deepMinutes === 0) {
    insights.push(`⚠️ No deep sleep detected. This is unusual - deep sleep should comprise ${targetPercent}% of total sleep. Possible causes: fragmented sleep, recent alcohol, or device tracking issues.`);
  } else if (isLow) {
    insights.push(`⚠️ Low deep sleep (${deepPercent}% of total sleep, target ${targetPercent}%). Deep sleep is when your body repairs tissue, builds bone/muscle, and strengthens immunity. To increase: reduce evening caffeine, cool your bedroom to 65°F/18°C, and exercise earlier in the day.`);
  } else if (isHigh) {
    insights.push(`💪 Excellent deep sleep (${deepPercent}% of total sleep)! Your body is getting quality physical recovery. This is great for athletic performance and immune function.`);
  } else {
    insights.push(`✅ Healthy deep sleep (${deepPercent}% of total sleep). You're getting the physical restoration your body needs.`);
  }
  
  // Deep sleep timing
  insights.push(`💤 Deep sleep duration: ${deepMinutes} minutes. Optimal deep sleep typically occurs in the first half of the night.`);
  
  return insights;
};

// REM sleep analysis
const analyzeREMSleep = (remMinutes, remPercent, age, totalSleepHours) => {
  const insights = [];
  const ageGuidelines = getAgeSpecificInsights(age);
  const targetPercent = ageGuidelines.remTarget;
  const isLow = remPercent < targetPercent - 5;
  const isHigh = remPercent > targetPercent + 10;
  
  if (remMinutes === 0) {
    insights.push(`⚠️ No REM sleep detected. REM sleep is crucial for memory, learning, and emotional health. This may indicate a recording issue or severely disrupted sleep.`);
  } else if (isLow) {
    insights.push(`⚠️ Low REM sleep (${remPercent}% of total sleep, target ${targetPercent}%). REM sleep consolidates memories and processes emotions. To increase: maintain consistent wake times, get morning sunlight, and avoid alcohol before bed.`);
  } else if (isHigh) {
    insights.push(`🧠 Excellent REM sleep (${remPercent}% of total sleep)! Great for memory consolidation and creative thinking.`);
  } else {
    insights.push(`✅ Healthy REM sleep (${remPercent}% of total sleep). Your brain is getting proper cognitive restoration.`);
  }
  
  insights.push(`💭 REM sleep duration: ${remMinutes} minutes. REM periods get longer throughout the night, which is why adequate total sleep is important.`);
  
  return insights;
};

// Heart rate and cardiovascular insights
const analyzeHeartRate = (restingHR, age, sleepData) => {
  const insights = [];
  if (!restingHR) return insights;
  
  let interpretation = '';
  if (restingHR < 60) interpretation = 'Excellent (athletic range)';
  else if (restingHR < 70) interpretation = 'Good';
  else if (restingHR < 80) interpretation = 'Fair';
  else interpretation = 'Elevated';
  
  insights.push(`❤️ Resting heart rate: ${restingHR} bpm (${interpretation}). Lower nighttime HR indicates better cardiovascular recovery.`);
  
  if (restingHR > 75 && age && age < 50) {
    insights.push(`⚠️ Elevated resting heart rate may indicate inadequate recovery, dehydration, or stress. Consider relaxation techniques before bed.`);
  }
  
  if (restingHR < 55 && age && age > 40) {
    insights.push(`💚 Excellent resting heart rate! This suggests good cardiovascular fitness and recovery.`);
  }
  
  return insights;
};

// Build comprehensive prompt for Groq
const buildComprehensivePrompt = (sleepData, trends, userProfile, historical) => {
  const bmi = userProfile?.height_cm && userProfile?.weight_kg ? 
    calculateBMI(userProfile.height_cm, userProfile.weight_kg) : null;
  const hrvAnalysis = analyzeHRV(sleepData.hrv, sleepData.hrv_variability, userProfile?.age);
  const outlierInsights = detectOutliers(sleepData, trends, userProfile);
  const circadianInsights = analyzeCircadianRhythm(sleepData, trends);
  const deepInsights = analyzeDeepSleep(sleepData.stages.deep.minutes, parseFloat(sleepData.stages.deep.percent), userProfile?.age, parseFloat(sleepData.duration_hours));
  const remInsights = analyzeREMSleep(sleepData.stages.rem.minutes, parseFloat(sleepData.stages.rem.percent), userProfile?.age, parseFloat(sleepData.duration_hours));
  const heartInsights = analyzeHeartRate(sleepData.resting_heart_rate, userProfile?.age, sleepData);
  
  const ageGuidelines = getAgeSpecificInsights(userProfile?.age, userProfile?.gender);
  
  return `You are a clinical sleep physiologist. Analyze this sleep data and provide a comprehensive, professional sleep report.

USER PROFILE:
- Age: ${userProfile?.age || 'Not provided'} (${userProfile?.age ? ageGuidelines.range : ''})
- Gender: ${userProfile?.gender || 'Not provided'}
- Height/Weight: ${userProfile?.height_cm || '?'}cm / ${userProfile?.weight_kg || '?'}kg ${bmi ? `(BMI: ${bmi.bmi}, ${bmi.category})` : ''}

LAST NIGHT'S SLEEP:
- Date: ${sleepData.date}
- Bedtime: ${sleepData.bedtime} (${sleepData.bedtime_hour}:00 hour)
- Wakeup: ${sleepData.wakeup}
- Duration: ${sleepData.duration_hours} hours (${sleepData.duration_minutes} minutes)
- Sleep Efficiency: ${sleepData.sleep_efficiency}%
- Quality Score: ${sleepData.quality_score}/100

STAGES:
- Deep Sleep: ${sleepData.stages.deep.minutes} min (${sleepData.stages.deep.percent}%)
- Light Sleep: ${sleepData.stages.light.minutes} min (${sleepData.stages.light.percent}%)
- REM Sleep: ${sleepData.stages.rem.minutes} min (${sleepData.stages.rem.percent}%)
- Awake: ${sleepData.stages.awake.minutes} min (${sleepData.stages.awake.percent}%)

PHYSIOLOGICAL DATA:
- Resting Heart Rate: ${sleepData.resting_heart_rate || 'Not available'} bpm
- HRV (RMSSD): ${sleepData.hrv || 'Not available'} ms
- HRV Variability: ${sleepData.hrv_variability || 'N/A'} ms

7-DAY TRENDS (from ${trends.days_analyzed} days):
- Average sleep: ${trends.avg_sleep_hours} hours
- Deep sleep: ${trends.avg_deep_minutes} min (${trends.avg_deep_percent}%)
- REM sleep: ${trends.avg_rem_minutes} min (${trends.avg_rem_percent}%)
- Sleep Efficiency: ${trends.avg_efficiency}%
- Bedtime Consistency: ${trends.consistency_score} (${trends.bedtime_consistency_hours}h variation)
- Average HRV: ${trends.avg_hrv || 'N/A'} ms

CONTEXTUAL INSIGHTS (calculated from your data):
${outlierInsights.map(i => `- ${i}`).join('\n')}
${circadianInsights.map(i => `- ${i}`).join('\n')}
${deepInsights.map(i => `- ${i}`).join('\n')}
${remInsights.map(i => `- ${i}`).join('\n')}
${heartInsights.map(i => `- ${i}`).join('\n')}
${hrvAnalysis ? `- HRV Analysis: ${hrvAnalysis.value}ms (${hrvAnalysis.interpretation})${hrvAnalysis.age_adjusted ? ` - ${hrvAnalysis.age_adjusted}` : ''}` : ''}

Based on this data, provide a JSON response with:
{
  "summary": "Comprehensive 2-3 sentence summary of their sleep health",
  "deepInsights": [
    "insight about sleep physiology",
    "insight about trends and patterns",
    "insight about HRV/autonomic nervous system",
    "insight about circadian timing",
    "insight about recovery quality"
  ],
  "recommendations": [
    "primary actionable recommendation",
    "secondary recommendation",
    "lifestyle modification",
    "sleep hygiene tip"
  ],
  "positiveNote": "One encouraging, personalized note",
  "riskAssessment": "String describing any health risks identified",
  "recoveryScore": "Number 0-100 based on: sleep duration, stage quality, HRV, consistency, efficiency",
  "circadianScore": "Number 0-100 based on bedtime consistency and timing",
  "sleepDebt": "Hours of sleep debt (if any, otherwise 0)"
}`;
};

// Call Groq API
const callGroqAPI = async (prompt) => {
  const apiKey = process.env.EXPO_PUBLIC_GROQ_API_KEY;
  
  if (!apiKey) {
    throw new Error('No Groq API key configured');
  }
  
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'llama-3.1-8b-instant',
      messages: [
        {
          role: 'system',
          content: 'You are a clinical sleep physiologist. Provide detailed, evidence-based sleep analysis. Respond with valid JSON only.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 0.7,
      max_tokens: 1500,
    }),
  });
  
  const data = await response.json();
  
  if (!response.ok) {
    throw new Error(data.error?.message || 'Groq API request failed');
  }
  
  const content = data.choices[0].message.content;
  let cleanContent = content;
  if (content.includes('```json')) cleanContent = content.split('```json')[1].split('```')[0];
  else if (content.includes('```')) cleanContent = content.split('```')[1].split('```')[0];
  
  try {
    return JSON.parse(cleanContent.trim());
  } catch {
    return null;
  }
};

// Comprehensive fallback insights (always works)
const getComprehensiveFallback = (sleepData, trends, userProfile, historical) => {
  const outlierInsights = detectOutliers(sleepData, trends, userProfile);
  const circadianInsights = analyzeCircadianRhythm(sleepData, trends);
  const deepInsights = analyzeDeepSleep(sleepData.stages.deep.minutes, parseFloat(sleepData.stages.deep.percent), userProfile?.age, parseFloat(sleepData.duration_hours));
  const remInsights = analyzeREMSleep(sleepData.stages.rem.minutes, parseFloat(sleepData.stages.rem.percent), userProfile?.age, parseFloat(sleepData.duration_hours));
  const heartInsights = analyzeHeartRate(sleepData.resting_heart_rate, userProfile?.age, sleepData);
  const hrvAnalysis = analyzeHRV(sleepData.hrv, sleepData.hrv_variability, userProfile?.age);
  
  const allInsights = [...outlierInsights, ...circadianInsights, ...deepInsights, ...remInsights, ...heartInsights];
  if (hrvAnalysis) allInsights.push(`📊 HRV: ${hrvAnalysis.value}ms (${hrvAnalysis.interpretation}). ${hrvAnalysis.age_adjusted || ''}`);
  
  // Calculate recovery score
  let recoveryScore = 50;
  if (parseFloat(sleepData.duration_hours) >= 7) recoveryScore += 15;
  else if (parseFloat(sleepData.duration_hours) >= 6) recoveryScore += 5;
  if (parseFloat(sleepData.stages.deep.percent) >= 18) recoveryScore += 10;
  else if (parseFloat(sleepData.stages.deep.percent) >= 12) recoveryScore += 5;
  if (parseFloat(sleepData.stages.rem.percent) >= 20) recoveryScore += 10;
  else if (parseFloat(sleepData.stages.rem.percent) >= 15) recoveryScore += 5;
  if (sleepData.quality_score >= 85) recoveryScore += 10;
  if (trends.avg_efficiency >= 85) recoveryScore += 5;
  if (sleepData.hrv && sleepData.hrv > 40) recoveryScore += 10;
  else if (sleepData.hrv && sleepData.hrv > 30) recoveryScore += 5;
  if (trends.consistency_score === 'Excellent') recoveryScore += 10;
  else if (trends.consistency_score === 'Good') recoveryScore += 5;
  recoveryScore = Math.min(100, Math.max(0, recoveryScore));
  
  // Calculate circadian score
  let circadianScore = 50;
  if (trends.bedtime_consistency_hours && parseFloat(trends.bedtime_consistency_hours) < 1) circadianScore += 30;
  else if (trends.bedtime_consistency_hours && parseFloat(trends.bedtime_consistency_hours) < 2) circadianScore += 15;
  if (sleepData.bedtime_hour >= 22 && sleepData.bedtime_hour <= 23) circadianScore += 10;
  if (trends.sleep_social_jetlag && parseFloat(trends.sleep_social_jetlag) < 1) circadianScore += 10;
  circadianScore = Math.min(100, Math.max(0, circadianScore));
  
  // Calculate sleep debt
  const recommendedHours = 8;
  const avgSleep = parseFloat(trends.avg_sleep_hours);
  const sleepDebt = avgSleep < recommendedHours ? Math.round((recommendedHours - avgSleep) * historical.days_analyzed) : 0;
  
  const summary = `Based on your sleep data from ${sleepData.date}, you slept for ${sleepData.duration_hours} hours with ${sleepData.quality_score}% efficiency. ` +
    `Deep sleep: ${sleepData.stages.deep.minutes}min (${sleepData.stages.deep.percent}%), REM: ${sleepData.stages.rem.minutes}min (${sleepData.stages.rem.percent}%). ` +
    `Over ${trends.days_analyzed} days, your average sleep is ${trends.avg_sleep_hours}h. ${trends.consistency_score === 'Excellent' ? 'You have excellent bedtime consistency!' : ''}`;
  
  let riskAssessment = `Based on your ${userProfile?.age || ''} year old profile: `;
  if (parseFloat(sleepData.duration_hours) < 6) {
    riskAssessment += `Chronic short sleep (<6h) increases cardiovascular disease risk by 20% and diabetes risk by 30%. `;
  } else if (parseFloat(sleepData.duration_hours) < 7) {
    riskAssessment += `Moderate sleep deficit may impact metabolic health. `;
  } else {
    riskAssessment += `Your sleep duration is within the healthy range for your age. `;
  }
  if (parseFloat(sleepData.stages.deep.percent) < 12) {
    riskAssessment += `Low deep sleep may impair immune function and recovery. `;
  }
  if (sleepData.resting_heart_rate && sleepData.resting_heart_rate > 80) {
    riskAssessment += `Elevated nighttime heart rate suggests cardiovascular strain. `;
  }
  if (riskAssessment.length < 50) riskAssessment += `No major risk factors detected from this night's data.`;
  
  let positiveNote = `✨ You're building valuable health awareness by tracking your sleep. `;
  if (recoveryScore >= 70) {
    positiveNote += `Your recovery metrics are excellent! Keep up your great sleep habits.`;
  } else if (recoveryScore >= 50) {
    positiveNote += `You're on the right track. Small improvements will make a big difference.`;
  } else {
    positiveNote += `Tonight is a new opportunity for better sleep. Start with a consistent bedtime.`;
  }
  
  // Generate recommendations
  const recommendations = [];
  if (sleepDebt > 14) recommendations.push(`⏰ You have accumulated ${sleepDebt} hours of sleep debt over ${historical.days_analyzed} days. Consider weekend catch-up sleep or adjusting your schedule.`);
  if (parseFloat(sleepData.stages.deep.percent) < 15) recommendations.push(`🏋️ To boost deep sleep: exercise before 3 PM, keep bedroom at 65-68°F, and avoid alcohol 3 hours before bed.`);
  if (parseFloat(sleepData.stages.rem.percent) < 20) recommendations.push(`🧘 For better REM: maintain consistent wake times (even weekends) and get 15 min morning sunlight.`);
  if (sleepData.hrv && sleepData.hrv < 30) recommendations.push(`🫁 To improve HRV: practice box breathing (4-4-4-4) before bed, reduce caffeine after 2 PM, and stay hydrated.`);
  if (trends.sleep_social_jetlag && parseFloat(trends.sleep_social_jetlag) > 1) recommendations.push(`🌅 Reduce social jetlag by waking within 1 hour of your weekday time on weekends.`);
  if (recommendations.length === 0) recommendations.push(`🌟 You're doing great! Maintain your sleep hygiene and consider mindfulness practice before bed.`);
  
  return {
    summary,
    deepInsights: allInsights.slice(0, 5),
    recommendations: recommendations.slice(0, 4),
    positiveNote,
    riskAssessment,
    recoveryScore,
    circadianScore,
    sleepDebt,
  };
};

// Save analysis
const saveSleepAnalysis = async (userId, analysisData, sleepData, trends) => {
  try {
    const predictionData = {
      user_id: userId,
      features: {
        duration_hours: parseFloat(sleepData.duration_hours),
        quality_score: sleepData.quality_score,
        deep_percent: parseFloat(sleepData.stages.deep.percent),
        rem_percent: parseFloat(sleepData.stages.rem.percent),
        hrv: sleepData.hrv,
        avg_sleep_7day: parseFloat(trends.avg_sleep_hours),
        avg_deep_7day: trends.avg_deep_minutes,
        avg_rem_7day: trends.avg_rem_minutes,
        consistency_score: trends.consistency_score,
        recovery_score: analysisData.recoveryScore,
        circadian_score: analysisData.circadianScore,
        sleep_debt: analysisData.sleepDebt,
      },
      prediction_label: analysisData.recoveryScore >= 70 ? 'Excellent Recovery' : analysisData.recoveryScore >= 50 ? 'Moderate Recovery' : 'Poor Recovery',
      risk_level: analysisData.recoveryScore >= 70 ? 'Low' : analysisData.recoveryScore >= 50 ? 'Moderate' : 'High',
      source: 'sleep_ai_analysis',
      fitbit_date: sleepData.date,
      metadata: {
        summary: analysisData.summary,
        insights: analysisData.deepInsights,
        recommendations: analysisData.recommendations,
        positiveNote: analysisData.positiveNote,
        riskAssessment: analysisData.riskAssessment,
        recoveryScore: analysisData.recoveryScore,
        circadianScore: analysisData.circadianScore,
        sleepDebt: analysisData.sleepDebt,
        trends: trends,
      },
    };
    
    const { error } = await supabase.from('predictions').insert([predictionData]);
    if (error) throw error;
    console.log('✅ Sleep analysis saved');
    return true;
  } catch (error) {
    console.error('Error saving sleep analysis:', error);
    return false;
  }
};

// Main export function
export const getSleepInsights = async (userId, userProfile) => {
  try {
    console.log('📊 Fetching sleep data for deep analysis...');
    
    const mostRecentSleepDate = await getMostRecentSleepDate(userId);
    if (!mostRecentSleepDate) throw new Error('No sleep data found. Please sync your Fitbit data first.');
    
    const { data, error } = await supabase
      .from('fitbit_data')
      .select('*')
      .eq('user_id', userId)
      .eq('date', mostRecentSleepDate)
      .single();
    
    if (error || !data) throw new Error('Could not fetch sleep data');
    
    const sleepData = formatSleepDataForAI(data.sleep_data, data.heart_rate_data, data.hrv_data);
    if (!sleepData) throw new Error('Could not process sleep data');
    
    const trends = await getHistoricalAnalysis(userId, 30);
    if (!trends) throw new Error('Could not calculate trends');
    
    let insights;
    let isLLM = false;
    
    const groqApiKey = process.env.EXPO_PUBLIC_GROQ_API_KEY;
    if (groqApiKey) {
      try {
        const prompt = buildComprehensivePrompt(sleepData, trends, userProfile, trends);
        const llmResponse = await callGroqAPI(prompt);
        if (llmResponse) {
          insights = llmResponse;
          isLLM = true;
          console.log('✅ AI insights generated via Groq');
        } else {
          throw new Error('Invalid LLM response');
        }
      } catch (llmError) {
        console.warn('Groq API failed, using comprehensive fallback:', llmError.message);
        insights = getComprehensiveFallback(sleepData, trends, userProfile, trends);
      }
    } else {
      insights = getComprehensiveFallback(sleepData, trends, userProfile, trends);
    }
    
    await saveSleepAnalysis(userId, insights, sleepData, trends);
    
    return {
      success: true,
      data: {
        sleepData,
        trends,
        insights,
        isLLM,
      },
    };
  } catch (error) {
    console.error('SleepAI error:', error);
    return { success: false, error: error.message };
  }
};