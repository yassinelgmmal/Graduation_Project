/**
 * API service for communicating with the Multimodal RAG API
 */
import { API_CONFIG } from '../config';

// Set API base URL
let API_BASE_URL = API_CONFIG.BASE_URL;

/**
 * Get the appropriate API URL based on the processing option
 * @param {string} processingOption - The selected processing option ('custom-models' or 'external-api-full')
 * @returns {string} - The API base URL
 */
const getApiUrl = (processingOption) => {
  if (processingOption === 'custom-models') {
    return API_CONFIG.CUSTOM_MODELS_URL;
  }
  return API_CONFIG.BASE_URL;
};

/**
 * Debug log function that only logs when debug mode is enabled
 */
const debugLog = (message, data = null) => {
  if (API_CONFIG.DEBUG) {
    if (data) {
      console.log(`[API DEBUG] ${message}`, data);
    } else {
      console.log(`[API DEBUG] ${message}`);
    }
  }
};

/**
 * Utility function to execute an async operation without a timeout
 * @param {Function} operation - The operation to execute
 * @returns {Promise<any>} - Result of the operation
 */
const retryOperation = async (operation) => {
  return await operation();
};

/**
 * Check the API server availability (simplified to always return the base URL)
 * @returns {Promise<string>} - The API base URL
 */
export const checkServer = async () => {
  debugLog(`Using configured API server URL: ${API_BASE_URL}`);
  return API_BASE_URL;
};

/**
 * Upload and ingest a PDF document
 * @param {File} file - The PDF file to upload
 * @param {string} processingOption - The selected processing option ('custom-models' or 'external-api-full')
 * @returns {Promise<{document_id: string, filename: string, status: string, chunks_processed: number}>}
 */
export const ingestDocument = async (file, processingOption = 'external-api-full') => {
  // Get the appropriate API URL based on the processing option
  const apiUrl = processingOption === 'custom-models' 
    ? getApiUrl(processingOption) 
    : await checkServer();
  
  const formData = new FormData();
  formData.append('file', file);

  return retryOperation(async () => {
    try {
      // Different endpoints based on processing option
      const endpoint = processingOption === 'custom-models'
        ? '/process-paper/'
        : '/api/v1/qa/ingest';
      
      debugLog(`Uploading document to ${apiUrl}${endpoint}`);
      
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error (${response.status}): ${errorText}`);
      }
      
      const data = await response.json();
      debugLog('Document ingested successfully', data);
      
      // Handle different response formats
      if (processingOption === 'custom-models') {
        // Transform the custom-models API response to match the external-api-full format
        return {
          document_id: data.paper_id || data.id,
          filename: data.filename || file.name,
          status: data.status || 'processed',
          chunks_processed: data.chunks_processed || 0
        };
      }
      
      return data;
    } catch (error) {
      debugLog('Error ingesting document', error.message);
      
      // Check for specific errors
      if (error.name === 'AbortError') {
        throw new Error(`Request timed out. The server might be busy or the file may be too large.`);
      }
      
      // Check if it's a network error
      if (error.message === 'Failed to fetch') {
        throw new Error(`Network error: Unable to connect to the API server at ${apiUrl}. Please ensure the server is running.`);
      }
      
      // Check for empty file error
      if (error.message.includes('No file provided')) {
        throw new Error('No file was provided for upload. Please select a PDF file.');
      }
      
      // Check for file format errors
      if (error.message.includes('Unsupported file format')) {
        throw new Error('Unsupported file format. Only PDF files are accepted.');
      }
      
      // Check for file size errors
      if (error.message.includes('File size exceeds')) {
        throw new Error('File size exceeds the maximum limit of 20MB.');
      }
      
      throw error;
    }
  });
};

/**
 * Ask a question about the ingested document
 * @param {string} query - The question to ask
 * @param {string} documentId - Optional document ID to restrict the question to
 * @param {number} topK - Number of chunks to retrieve (default: 5)
 * @param {string} processingOption - The selected processing option ('custom-models' or 'external-api-full')
 * @returns {Promise<{answer: string, sources: Array, query: string}>}
 */
export const askQuestion = async (query, documentId = null, topK = 5, processingOption = 'external-api-full') => {
  // Get the appropriate API URL based on the processing option
  const apiUrl = processingOption === 'custom-models' 
    ? getApiUrl(processingOption) 
    : await checkServer();
  
  let payload;
  let endpoint;
  
  if (processingOption === 'custom-models') {
    // Format for custom-models API
    payload = {
      query: query
    };
    
    if (documentId) {
      payload.paper_id = documentId;
    }
    
    endpoint = '/query/';
  } else {
    // Format for external-api-full API
    payload = {
      query,
      top_k: topK
    };
    
    if (documentId) {
      payload.document_id = documentId;
    }
    
    endpoint = '/api/v1/qa/ask';
  }
  
  return retryOperation(async () => {
    try {
      debugLog(`Sending question to ${apiUrl}${endpoint}`, payload);
      
      // Different content types based on API
      const headers = {
        'Content-Type': processingOption === 'custom-models' 
          ? 'application/x-www-form-urlencoded' 
          : 'application/json'
      };
      
      // Convert payload to form data for custom-models API
      let body;
      if (processingOption === 'custom-models') {
        const formData = new URLSearchParams();
        for (const key in payload) {
          formData.append(key, payload[key]);
        }
        body = formData;
      } else {
        body = JSON.stringify(payload);
      }
      
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        headers,
        body
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error (${response.status}): ${errorText}`);
      }
      
      const data = await response.json();
      debugLog('Question answered successfully', data);
      
      // Transform response to a consistent format
      if (processingOption === 'custom-models') {
        return {
          answer: data.generated_response || data.response || data.answer || '',
          sources: data.relevant_chunks || data.chunks || data.sources || [],
          query: query
        };
      }
      
      return data;
    } catch (error) {
      debugLog('Error asking question', error.message);
      
      if (error.name === 'AbortError') {
        throw new Error(`Request timed out. The server might be busy or your question might be too complex.`);
      }
      
      if (error.message === 'Failed to fetch') {
        throw new Error(`Network error: Unable to connect to the API server at ${apiUrl}. Please ensure the server is running.`);
      }
      
      throw error;
    }
  });
};

/**
 * Get follow-up questions for a given question
 * @param {string} query - The original question
 * @param {string} documentId - Optional document ID to restrict follow-up questions to
 * @param {string} processingOption - The selected processing option ('custom-models' or 'external-api-full')
 * @returns {Promise<{questions: string[]}>}
 */
export const getFollowUpQuestions = async (query, documentId = null, processingOption = 'external-api-full') => {
  // Get the appropriate API URL based on the processing option
  const apiUrl = processingOption === 'custom-models' 
    ? getApiUrl(processingOption) 
    : await checkServer();
  
  let payload;
  let endpoint;
  
  if (processingOption === 'custom-models') {
    // Format for custom-models API
    payload = {
      query: query
    };
    
    // Document ID is required for the custom-models API
    if (documentId) {
      payload.paper_id = documentId;
    } else {
      throw new Error('Document ID is required for follow-up questions with custom models');
    }
    
    endpoint = '/suggest-questions/';
  } else {
    // Format for external-api-full API
    payload = {
      query
    };
    
    if (documentId) {
      payload.document_id = documentId;
    }
    
    endpoint = '/api/v1/qa/follow-up';
  }
  
  return retryOperation(async () => {
    try {
      debugLog(`Getting follow-up questions from ${apiUrl}${endpoint}`, payload);
      
      // Different content types based on API
      const headers = {
        'Content-Type': processingOption === 'custom-models' 
          ? 'application/x-www-form-urlencoded' 
          : 'application/json'
      };
      
      // Convert payload to form data for custom-models API
      let body;
      if (processingOption === 'custom-models') {
        const formData = new URLSearchParams();
        for (const key in payload) {
          formData.append(key, payload[key]);
        }
        body = formData;
      } else {
        body = JSON.stringify(payload);
      }
      
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        headers,
        body
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error (${response.status}): ${errorText}`);
      }
      
      const data = await response.json();
      debugLog('Follow-up questions generated successfully', data);
      
      // Transform response to a consistent format
      if (processingOption === 'custom-models') {
        return {
          questions: data.questions || data.suggested_questions || []
        };
      }
      
      return data;
    } catch (error) {
      debugLog('Error generating follow-up questions', error.message);
      
      if (error.name === 'AbortError') {
        throw new Error(`Request timed out. The server might be busy.`);
      }
      
      if (error.message === 'Failed to fetch') {
        throw new Error(`Network error: Unable to connect to the API server at ${apiUrl}. Please ensure the server is running.`);
      }
      
      throw error;
    }
  });
};

/**
 * Generate a summary of the ingested document
 * @param {string} documentId - The document ID to summarize
 * @param {string} summaryType - Type of summary ('comprehensive' or 'executive')
 * @param {number} length - Desired length of the summary in words
 * @param {string} processingOption - The selected processing option ('custom-models' or 'external-api-full')
 * @returns {Promise<{summary: string, document_id: string, summary_type: string, length: number, paperTopic: string}>}
 */
export const getSummary = async (documentId, summaryType = 'comprehensive', length = 250, processingOption = 'external-api-full') => {
  // Get the appropriate API URL based on the processing option
  const apiUrl = processingOption === 'custom-models' 
    ? getApiUrl(processingOption) 
    : await checkServer();
  
  let payload;
  let endpoint;
  
  if (processingOption === 'custom-models') {
    // For custom-models, we get the paper data directly using the paper_id
    // This API doesn't have a separate summary endpoint, so we'll get the paper data
    // which already includes the summary
    endpoint = `/papers/${documentId}`;
    
    return retryOperation(async () => {
      try {
        debugLog(`Getting paper data from ${apiUrl}${endpoint}`);
        
        const response = await fetch(`${apiUrl}${endpoint}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`API error (${response.status}): ${errorText}`);
        }
        
        const data = await response.json();
        debugLog('Paper data retrieved successfully', data);
        
        // Store paper topic in sessionStorage
        if (data.classifications) {
          sessionStorage.setItem('paperTopic', data.classifications);
        }
        
        // Transform the response to match the expected format
        return {
          summary: data.summary || '',
          document_id: documentId,
          summary_type: summaryType,
          length: data.summary ? data.summary.split(' ').length : 0,
          title: data.title || '',
          authors: data.authors || [],
          paperTopic: data.classifications || 'Academic Research'
        };
      } catch (error) {
        debugLog('Error getting paper data', error.message);
        
        if (error.name === 'AbortError') {
          throw new Error(`Request timed out. The server might be busy.`);
        }
        
        if (error.message === 'Failed to fetch') {
          throw new Error(`Network error: Unable to connect to the API server at ${apiUrl}. Please ensure the server is running.`);
        }
        
        throw error;
      }
    });
  } else {
    // Format for external-api-full API
    payload = {
      doc_id: documentId,
      summary_type: summaryType,
      max_length: length
    };
    
    endpoint = '/api/v1/summarize';
  }
  
  return retryOperation(async () => {
    try {
      debugLog(`Generating summary from ${apiUrl}${endpoint}`, payload);
      
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error (${response.status}): ${errorText}`);
      }
      
      const data = await response.json();
      debugLog('Summary generated successfully', data);
      return data;
    } catch (error) {
      debugLog('Error generating summary', error.message);
      
      if (error.name === 'AbortError') {
        throw new Error(`Request timed out. Summarizing large documents can take time.`);
      }
      
      if (error.message === 'Failed to fetch') {
        throw new Error(`Network error: Unable to connect to the API server at ${apiUrl}. Please ensure the server is running.`);
      }
      
      throw error;
    }
  });
};
