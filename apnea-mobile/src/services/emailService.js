// src/services/emailService.js
import { supabase } from './supabase';

// This will call a Supabase Edge Function to send emails
export const sendVerificationEmail = async (email, code) => {
  try {
    // Option 1: Use Supabase Edge Function (recommended)
    const { error } = await supabase.functions.invoke('send-verification-email', {
      body: { email, code }
    });
    
    if (error) throw error;
    return { success: true };
    
    // Option 2: For now, just log the code (for development)
    console.log(`📧 Verification code for ${email}: ${code}`);
    return { success: true };
  } catch (error) {
    console.error('Error sending email:', error);
    return { success: false, error };
  }
};