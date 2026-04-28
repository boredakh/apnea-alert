// C:\Users\s4303218\Documents\codebase\apnea-alert\apnea-mobile\src\context\AuthContext.js
import React, { createContext, useState, useEffect, useContext } from 'react'
import { supabase } from '../services/supabase'
import { Alert } from 'react-native'
import { fetchProfileWithTimeout } from '../services/supabaseDirect';

const AuthContext = createContext({})

export const useAuth = () => useContext(AuthContext)

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(true)

  // Add this at the top of your AuthContext, outside the component
  let pendingProfileFetch = null;

  // Send verification email via Resend Edge Function
  const sendVerificationEmail = async (email, code, name = '') => {
    try {
      console.log(`🔥🔥🔥 sendVerificationEmail WAS CALLED!`, { email, code });
      console.log(`📧 Sending verification email to ${email} with code ${code}`);
      
      const { data, error } = await supabase.functions.invoke('send-verification', {
        body: { email, code, name }
      });

      if (error) {
        console.error('❌ Failed to send email:', error);
        return false;
      }

      console.log('✅ Verification email sent:', data);
      return true;
    } catch (error) {
      console.error('❌ Email sending error:', error);
      return false;
    }
  };

  // Inside your component, update fetchProfile:
  const fetchProfile = async (userId) => {
    if (pendingProfileFetch) {
      console.log('⚠️ Profile fetch already in progress, skipping...');
      return pendingProfileFetch;
    }
    
    console.log('📡 fetchProfile START for user:', userId);
    
    pendingProfileFetch = (async () => {
      try {
        // Use direct fetch with timeout
        const { data, error } = await fetchProfileWithTimeout(userId, 5000);
        
        if (error) {
          console.error('❌ Profile fetch error:', error);
          throw error;
        }

        if (data) {
          console.log('✅ Profile data received:', data);
          setProfile(data);
        } else {
          console.log('⚠️ No profile found, creating one...');
          
          // Create profile
          const { data: newProfile, error: createError } = await supabase
            .from('profiles')
            .insert([{ 
              id: userId, 
              email: user?.email || 'unknown',
              email_verified: false,
              onboarding_complete: false 
            }])
            .select()
            .single();
            
          if (createError) throw createError;
          
          console.log('✅ Profile created:', newProfile);
          setProfile(newProfile);
        }
        
      } catch (error) {
        console.error('❌ Error in fetchProfile:', error);
        
        // On error, sign out
        console.log('🚪 Signing out due to profile fetch error');
        await supabase.auth.signOut();
        setUser(null);
        setProfile(null);
      } finally {
        setLoading(false);
        pendingProfileFetch = null;
      }
    })();
    
    return pendingProfileFetch;
  };

  useEffect(() => {
    // Check for active session
    supabase.auth.getSession().then(async ({ data: { session } }) => {
      if (session?.user) {
        setUser(session.user);
        await fetchProfile(session.user.id);
      } else {
        setLoading(false); // No session, stop loading
      }
    });

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(async (_event, session) => {
      console.log('🔄 Auth state changed:', _event);
      setUser(session?.user ?? null);
      
      if (session?.user) {
        await fetchProfile(session.user.id);
      } else {
        setProfile(null);
        setLoading(false);
      }
    });

    return () => subscription.unsubscribe();
  }, []);
  

  // Create verification code after signup
  const createVerificationCode = async (userId) => {
    try {
      // Generate 6-digit code
      const code = Math.floor(100000 + Math.random() * 900000).toString();
      const expiresAt = new Date();
      expiresAt.setHours(expiresAt.getHours() + 24); // 24 hour expiry

      console.log(`📝 Creating verification code for user: ${userId}`);
      console.log(`🔢 Code: ${code}, Expires: ${expiresAt.toISOString()}`);

      // Store in database
      const { data, error } = await supabase
        .from('verification_codes')
        .insert([
          {
            user_id: userId,
            code: code,
            expires_at: expiresAt.toISOString(),
            attempts: 0
          }
        ])
        .select();

      if (error) {
        console.error('❌ Error creating verification code:', error);
        return { code: null, error };
      }

      console.log('✅ Verification code saved successfully:', data);
      return { code, error: null };
    } catch (error) {
      console.error('❌ Unexpected error creating verification code:', error);
      return { code: null, error };
    }
  };

  // Verify the 6-digit code
  const verifyCode = async (userId, code) => {
    try {
      console.log(`🔍 Verifying code for user: ${userId}, code: ${code}`);
      
      // First, get the verification code record
      const { data, error } = await supabase
        .from('verification_codes')
        .select('*')
        .eq('user_id', userId)
        .eq('code', code);

      if (error) {
        console.error('Database error:', error);
        return { success: false, error: 'Database error occurred' };
      }

      if (!data || data.length === 0) {
        console.log('❌ No verification code found');
        return { success: false, error: 'Invalid verification code' };
      }

      console.log('✅ Verification code found:', data[0]);
      const verificationRecord = data[0];

      // Check if expired
      if (new Date(verificationRecord.expires_at) < new Date()) {
        return { success: false, error: 'Verification code has expired' };
      }

      // Check attempts
      if (verificationRecord.attempts >= 5) {
        return { success: false, error: 'Too many failed attempts. Request a new code.' };
      }

      // Increment attempts
      await supabase
        .from('verification_codes')
        .update({ attempts: verificationRecord.attempts + 1 })
        .eq('user_id', userId)
        .eq('code', code);

      // Update profile to mark email as verified
      const { error: profileError, data: profileData } = await supabase
        .from('profiles')
        .update({ email_verified: true })
        .eq('id', userId)
        .select();

      if (profileError) {
        console.error('❌ Failed to update profile:', profileError);
        return { success: false, error: profileError.message };
      }

      // Delete the verification code
      await supabase
        .from('verification_codes')
        .delete()
        .eq('user_id', userId);

      console.log('✅ Email verified successfully');

      // IMPORTANT: Refresh the user's profile in context
      await fetchProfile(userId);
      
      // Also refresh the user object to ensure everything is in sync
      const { data: { user: refreshedUser } } = await supabase.auth.getUser();
      setUser(refreshedUser);

      return { success: true, error: null };
      
    } catch (error) {
      console.error('Error in verifyCode:', error);
      return { success: false, error: error.message };
    }
  };

  // Resend verification code - FIXED VERSION
  const resendVerificationCode = async (userId) => {
    try {
      console.log(`📝 Resending verification code for user: ${userId}`);
      
      // Get user email
      const { data: { user } } = await supabase.auth.getUser();
      
      if (!user?.email) {
        console.error('❌ No email found for user');
        return { success: false, error: 'No email found' };
      }
      
      // Delete any existing codes for this user
      const { error: deleteError } = await supabase
        .from('verification_codes')
        .delete()
        .eq('user_id', userId);

      if (deleteError) {
        console.error('Error deleting old codes:', deleteError);
      }

      // Create new code
      const { code, error } = await createVerificationCode(userId);
      if (error) throw error;

      // Send email with new code
      const emailSent = await sendVerificationEmail(user.email, code);
      
      if (emailSent) {
        console.log(`✅ New verification code sent to ${user.email}: ${code}`);
        return { success: true, error: null };
      } else {
        console.error(`❌ Failed to send email to ${user.email}`);
        return { success: false, error: 'Failed to send email' };
      }
    } catch (error) {
      console.error('Error resending code:', error);
      return { success: false, error };
    }
  };

  // Check if email is verified
  const isEmailVerified = async () => {
    if (!user) return false;
    
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('email_verified')
        .eq('id', user.id)
        .single();

      if (error) throw error;
      return data?.email_verified || false;
    } catch (error) {
      console.error('Error checking verification status:', error);
      return false;
    }
  };

  // Check if onboarding is complete
  const isOnboardingComplete = async () => {
    if (!user) return false;
    
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('onboarding_complete')
        .eq('id', user.id)
        .single();

      if (error) throw error;
      return data?.onboarding_complete || false;
    } catch (error) {
      console.error('Error checking onboarding status:', error);
      return false;
    }
  };

  // Get user's current flow state
  const getUserFlowState = async () => {
    if (!user) return 'login';
    
    try {
      // First try to get from local profile if available
      if (profile) {
        console.log('📊 Using local profile for flow state:', { 
          email_verified: profile.email_verified, 
          onboarding_complete: profile.onboarding_complete 
        });
        
        if (!profile.email_verified) return 'verification';
        if (!profile.onboarding_complete) return 'onboarding';
        return 'home';
      }
      
      // If no local profile, fetch from database
      console.log('📡 Fetching profile for flow state');
      const { data, error } = await supabase
        .from('profiles')
        .select('email_verified, onboarding_complete')
        .eq('id', user.id)
        .single();

      if (error) {
        console.error('Error fetching profile:', error);
        // If profile doesn't exist, they need verification
        return 'verification';
      }
      
      console.log('📊 getUserFlowState - data:', data);
      
      if (!data.email_verified) return 'verification';
      if (!data.onboarding_complete) return 'onboarding';
      return 'home';
    } catch (error) {
      console.error('Error getting user flow state:', error);
      return 'login';
    }
  };

  // Mark onboarding as complete
  const completeOnboarding = async () => {
    if (!user) return { error: 'No user logged in' };
    
    try {
      const { error } = await supabase
        .from('profiles')
        .update({ onboarding_complete: true })
        .eq('id', user.id);

      if (error) throw error;
      
      // Update local profile state
      setProfile(prev => ({ ...prev, onboarding_complete: true }));
      
      return { error: null };
    } catch (error) {
      console.error('Error completing onboarding:', error);
      return { error };
    }
  };

  const refreshProfile = async () => {
    if (!user) return null;
    console.log('🔄 Refreshing profile for user:', user.id);
    await fetchProfile(user.id);
    return profile;
  };

  // Sign up function with verification code - FIXED VERSION
  const signUp = async (email, password, profileData = {}) => {
    try {
      setLoading(true);
      console.log(`📝 Attempting to sign up user: ${email}`);

      // First, check if there's an existing unverified user with this email
      const { data: existingProfiles, error: profileError } = await supabase
        .from('profiles')
        .select('id, email_verified')
        .eq('email', email);

      if (profileError) {
        console.error('Error checking existing profile:', profileError);
      }

      // If there's an existing unverified profile, just resend code
      if (existingProfiles && existingProfiles.length > 0) {
        const existingProfile = existingProfiles[0];
        
        if (!existingProfile.email_verified) {
          console.log('📝 Found existing unverified user');
          
          // Delete old verification codes
          await supabase
            .from('verification_codes')
            .delete()
            .eq('user_id', existingProfile.id);

          // Create new verification code
          const { code, error: codeError } = await createVerificationCode(existingProfile.id);
          if (codeError) throw codeError;

          // Send email with code
          await sendVerificationEmail(email, code, profileData.full_name);

          console.log(`🔐 New verification code for ${email}: ${code}`);

          return {
            data: {
              user: { id: existingProfile.id, email },
              userId: existingProfile.id,
              email: email
            },
            error: null
          };
        }
      }

      // No existing unverified user, proceed with normal signup
      const options = {};
      if (Object.keys(profileData).length > 0) {
        options.data = {
          full_name: profileData.full_name,
          age: profileData.age,
          gender: profileData.gender,
          height_cm: profileData.height,
          weight_kg: profileData.weight,
        };
      }

      const { data, error } = await supabase.auth.signUp({ 
        email, 
        password,
        options
      });

      if (error) {
        if (error.status === 429) {
          throw new Error('Too many signup attempts. Please wait a few minutes and try again.');
        }
        throw error;
      }

      console.log('✅ Signup successful:', data.user?.id);

      // Create verification code for the new user
      if (data.user) {
        const { code, error: codeError } = await createVerificationCode(data.user.id);
        if (codeError) throw codeError;
        
        // Send email with the code
        await sendVerificationEmail(email, code, profileData.full_name);
        
        console.log(`🔐 Verification code for ${email}: ${code}`);
        
        return { 
          data: { 
            ...data, 
            userId: data.user.id,
            email: data.user.email 
          }, 
          error: null 
        };
      }

      return { data, error: null };
    } catch (error) {
      console.error('❌ Signup error:', error);
      return { data: null, error };
    } finally {
      setLoading(false);
    }
  };

  // Sign in function - UPDATED
  const signIn = async (email, password) => {
    try {
      setLoading(true);
      console.log('🔑 Attempting sign in for:', email);
      
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      
      if (error) throw error;
      
      console.log('✅ Sign in successful, user:', data.user?.id);
      
      setUser(data.user);
      
      return { data, error: null };
    } catch (error) {
      console.error('❌ Sign in error:', error);
      setLoading(false);
      return { data: null, error };
    }
  };

  // Sign out function - FIXED VERSION
  const signOut = async () => {
    console.log('🚪 signOut function called');
    try {
      setLoading(true);
      console.log('📝 Clearing local state...');
      
      // Clear local state FIRST
      setUser(null);
      setProfile(null);
      console.log('✅ Local state cleared');
      
      // Then attempt Supabase sign-out
      console.log('📞 Calling supabase.auth.signOut()...');
      const { error } = await supabase.auth.signOut();
      
      if (error) {
        console.log('⚠️ Supabase sign-out error (ignoring):', error.message);
      } else {
        console.log('✅ Supabase sign-out successful');
      }
      
      console.log('🎉 signOut completed');
      
    } catch (error) {
      console.error('❌ Fatal error in signOut:', error);
      // Still ensure local state is cleared
      setUser(null);
      setProfile(null);
    } finally {
      console.log('🏁 signOut finally block - setting loading false');
      setLoading(false);
    }
  };

  const forceLogout = async () => {
    console.log('🔨 forceLogout called');
    
    // Clear local state first
    setUser(null);
    setProfile(null);
    
    // Try multiple methods to sign out
    try {
      // Method 1: Supabase signOut
      await supabase.auth.signOut();
    } catch (e) {
      console.log('Method 1 failed:', e.message);
    }
    
    try {
      // Method 2: Clear session manually
      await supabase.auth.setSession(null);
    } catch (e) {
      console.log('Method 2 failed:', e.message);
    }
    
    console.log('✅ forceLogout completed');
  };

  // Reset password function
  const resetPassword = async (email) => {
    try {
      setLoading(true);
      console.log('🔑 Requesting password reset for:', email);
      
      const { data, error } = await supabase.functions.invoke('reset-password', {
        body: { email }
      });

      if (error) {
        console.error('❌ Edge Function error:', error);
        throw error;
      }

      console.log('✅ Edge Function response:', data);
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to send reset email');
      }

      return { error: null };
    } catch (error) {
      console.error('❌ Reset password error:', error);
      return { error };
    } finally {
      setLoading(false);
    }
  };

  // Update profile function
  const updateProfile = async (updates) => {
    try {
      const { error } = await supabase
        .from('profiles')
        .update(updates)
        .eq('id', user.id);

      if (error) throw error;
      await fetchProfile(user.id);
      return { error: null };
    } catch (error) {
      return { error };
    }
  };

  // Check if user needs to complete onboarding (legacy method, use isOnboardingComplete instead)
  const needsOnboarding = () => {
    if (!profile) return true;
    // Check if essential profile fields are missing
    return !profile.full_name || !profile.age || !profile.gender;
  };

  const value = {
  signUp,
  signIn,
  signOut,
  forceLogout,
  resetPassword,
  updateProfile,
  verifyCode,
  resendVerificationCode,
  createVerificationCode,
  isEmailVerified,
  isOnboardingComplete,
  getUserFlowState,
  completeOnboarding,
  needsOnboarding,
  refreshProfile,  // ADD THIS LINE
  user,
  profile,
  loading,
};

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};