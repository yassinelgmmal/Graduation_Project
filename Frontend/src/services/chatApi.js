/**
 * Enhanced API service for the chat functionality with 
 * simplified error handling and retry mechanisms
 */
import { askQuestion as baseAskQuestion, getFollowUpQuestions as baseGetFollowUpQuestions } from './api';
import { getApiErrorExplanation } from './apiTesting';
import { API_CONFIG } from '../config';

// Always assume we're connected - no connection checking
export const getConnectionState = () => {
  return { 
    isConnected: true, 
    lastCheckTimestamp: Date.now(),
    recoveryAttempts: 0,
    isRecovering: false
  };
};

/**
 * Dummy function - always returns true
 * @returns {Promise<boolean>} - Always returns true
 */
export const recoverApiConnection = async () => {
  return true;
};

/**
 * Get the processing option from sessionStorage
 * @returns {string} - The processing option ('custom-models' or 'external-api-full')
 */
export const getProcessingOption = () => {
  return sessionStorage.getItem('processingOption') || 'external-api-full';
};

/**
 * Wrapper for askQuestion with retries
 * @param {string} query - The question to ask
 * @param {string} documentId - Optional document ID to restrict the question to
 * @param {number} topK - Number of chunks to retrieve (default: 5)
 * @returns {Promise<{answer: string, sources: Array, query: string}>}
 */
export const askQuestionWithRetry = async (query, documentId = null, topK = 5) => {
  const maxRetries = API_CONFIG.RETRY.MAX_RETRIES;
  let retries = 0;
  let lastError = null;
  const processingOption = getProcessingOption();
  
  while (retries <= maxRetries) {
    try {
      const result = await baseAskQuestion(query, documentId, topK, processingOption);
      return result;
    } catch (error) {
      lastError = error;
      
      // Don't retry timeouts or client errors
      if (error.name === 'AbortError' || 
          error.message.includes('400') || 
          error.message.includes('401') || 
          error.message.includes('403')) {
        throw error;
      }
      
      // If it's the last retry, throw the error
      if (retries === maxRetries) {
        throw error;
      }
      
      // Wait before retrying with exponential backoff
      const delay = API_CONFIG.RETRY.RETRY_DELAY * Math.pow(2, retries);
      await new Promise(resolve => setTimeout(resolve, delay));
      
      retries++;
      console.log(`Retrying askQuestion (attempt ${retries}/${maxRetries})...`);
    }
  }
  
  // This should never be reached, but just in case
  throw lastError || new Error('Failed to get response after retries');
};

/**
 * Wrapper for getFollowUpQuestions with retries
 * @param {string} query - The original question
 * @param {string} documentId - Optional document ID to restrict the question to
 * @returns {Promise<Object>}
 */
export const getFollowUpQuestionsWithRetry = async (query, documentId = null) => {
  const maxRetries = API_CONFIG.RETRY.MAX_RETRIES;
  let retries = 0;
  let lastError = null;
  const processingOption = getProcessingOption();
  
  while (retries <= maxRetries) {
    try {
      const result = await baseGetFollowUpQuestions(query, documentId, processingOption);
      return result;
    } catch (error) {
      lastError = error;
      
      // Don't retry timeouts or client errors
      if (error.name === 'AbortError' || 
          error.message.includes('400') || 
          error.message.includes('401') || 
          error.message.includes('403')) {
        throw error;
      }
      
      // If it's the last retry, throw the error
      if (retries === maxRetries) {
        throw error;
      }
      
      // Wait before retrying with exponential backoff
      const delay = API_CONFIG.RETRY.RETRY_DELAY * Math.pow(2, retries);
      await new Promise(resolve => setTimeout(resolve, delay));
      
      retries++;
      console.log(`Retrying getFollowUpQuestions (attempt ${retries}/${maxRetries})...`);
    }
  }
  
  // This should never be reached, but just in case
  throw lastError || new Error('Failed to get follow-up questions after retries');
};

/**
 * Handle API errors in a user-friendly way
 * @param {Error} error - The error object
 * @returns {Object} - User-friendly error message with suggestions
 */
export const handleApiError = (error) => {
  if (!error) return {
    message: 'An unknown error occurred',
    suggestion: 'Please try again or contact support if the problem persists.',
    recoverable: true
  };
  
  // Get detailed error explanation
  const explanation = getApiErrorExplanation(error);
  
  // Connection-related errors
  if (error.name === 'AbortError' || error.message.includes('timeout')) {
    return {
      message: explanation.message,
      suggestion: 'Try asking a simpler question or try again later when the server is less busy.',
      cause: explanation.cause,
      solution: explanation.solution,
      recoverable: true
    };
  }
  
  if (error.message.includes('Failed to fetch') || error.message.includes('Network error')) {
    return {
      message: explanation.message,
      suggestion: 'Check your internet connection and confirm the API server is running.',
      cause: explanation.cause,
      solution: explanation.solution,
      recoverable: true
    };
  }
  
  if (error.message.includes('CORS')) {
    return {
      message: explanation.message,
      suggestion: 'This is a configuration issue that needs to be fixed by the API server administrator.',
      cause: explanation.cause,
      solution: explanation.solution,
      recoverable: false
    };
  }
  
  // Default response
  return {
    message: explanation.message,
    suggestion: 'Please try again or contact support if the problem persists.',
    cause: explanation.cause,
    solution: explanation.solution,
    recoverable: true
  };
};
