// src/services/supabaseDirect.js
import { supabase } from './supabase';

// Direct fetch with timeout for profiles
export const fetchProfileWithTimeout = async (userId, timeoutMs = 5000) => {
  console.log('📡 Direct profile fetch for:', userId);
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    console.log('⚠️ Direct fetch timeout, aborting...');
    controller.abort();
  }, timeoutMs);

  try {
    // Use REST API directly instead of Supabase client
    const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;
    const supabaseKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
    
    const response = await fetch(
      `${supabaseUrl}/rest/v1/profiles?id=eq.${userId}&select=*`,
      {
        method: 'GET',
        headers: {
          'apikey': supabaseKey,
          'Authorization': `Bearer ${supabaseKey}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        signal: controller.signal,
      }
    );

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('📡 Direct fetch response:', data);
    
    return { data: data[0] || null, error: null };
  } catch (error) {
    clearTimeout(timeoutId);
    console.error('❌ Direct fetch error:', error);
    return { data: null, error };
  }
};