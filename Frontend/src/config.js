/**
 * Configuration settings for the application
 */

// API configuration
export const API_CONFIG = {
  // Base URL for the API
  BASE_URL: 'http://openai-rag:8010',
  
  // Base URL for the custom models API
  CUSTOM_MODELS_URL: 'http://custom-rag:8030',
  
  // No timeout for API requests - wait indefinitely
  TIMEOUT: 0,
  
  // No retries needed as we wait indefinitely
  RETRY: {
    MAX_RETRIES: 0,
    RETRY_DELAY: 0,
    MAX_BACKOFF: 0
  },
  
  // Connection check intervals
  CONNECTION_CHECK: {
    INTERVAL: 0, // No connection checking
    HEALTH_ENDPOINT: '/health',
    TIMEOUT: 0 // No timeout for health checks
  },
  
  // Fallback behavior
  FALLBACK: {
    ENABLED: true, // Enable fallback mechanisms
    OFFLINE_MODE: false, // If true, enables limited offline functionality
    CACHE_RESPONSES: true // Cache API responses for offline use
  },
  
  // Debug mode - set to true to enable detailed console logging
  DEBUG: true
};

// Feature flags
export const FEATURES = {
  ENABLE_METHODOLOGY_ANALYSIS: true,
  ENABLE_FOLLOW_UP_QUESTIONS: true,
  ENABLE_DARK_MODE: true,
  ENABLE_API_STATUS_MONITORING: true,
  ENABLE_ERROR_REPORTING: true,
  ENABLE_SESSION_RECOVERY: true
};

// Summary configuration
export const SUMMARY_CONFIG = {
  MIN_LENGTH: 50,
  MAX_LENGTH: 500,
  DEFAULT_LENGTH: 250,
  DEFAULT_TYPE: 'comprehensive',
};

// Error handling configuration
export const ERROR_CONFIG = {
  SHOW_TECHNICAL_DETAILS: false, // Show technical error details to users
  LOG_TO_CONSOLE: true, // Log errors to console
  RETRY_ON_ERROR: true, // Automatically retry on certain errors
  DEFAULT_ERROR_MESSAGE: 'An error occurred while processing your request. Please try again later.'
};

// Performance monitoring
export const PERFORMANCE_CONFIG = {
  COLLECT_METRICS: true, // Collect performance metrics
  METRICS_SAMPLE_RATE: 1.0, // Percentage of requests to collect metrics for (0-1)
  SLOW_REQUEST_THRESHOLD: 5000 // Threshold in ms to mark a request as slow
};
