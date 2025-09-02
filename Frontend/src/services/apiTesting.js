/**
 * API Testing and Error Handling Utilities
 */

/**
 * Extract a user-friendly error explanation from API errors
 * @param {Error} error - The error object
 * @returns {Object} - Object containing error message, cause, and solution
 */
export const getApiErrorExplanation = (error) => {
  if (!error) {
    return {
      message: 'An unknown error occurred',
      cause: 'The error object is undefined',
      solution: 'Please try again or contact support if the problem persists'
    };
  }

  // Network errors
  if (error.message === 'Failed to fetch' || error.message.includes('Network error')) {
    return {
      message: 'Network error: Unable to connect to the API server',
      cause: 'The API server might be down or your internet connection might be unstable',
      solution: 'Check your internet connection and confirm the API server is running'
    };
  }

  // Timeout errors
  if (error.name === 'AbortError' || error.message.includes('timeout')) {
    return {
      message: 'Request timed out',
      cause: 'The server is taking too long to respond, possibly due to high load or a complex request',
      solution: 'Try again later or try a simpler request'
    };
  }

  // CORS errors
  if (error.message.includes('CORS')) {
    return {
      message: 'Cross-Origin Resource Sharing (CORS) error',
      cause: 'The API server is not configured to accept requests from this domain',
      solution: 'This is a server configuration issue that needs to be fixed by the API administrator'
    };
  }

  // Authentication errors
  if (error.message.includes('401') || error.message.includes('unauthorized')) {
    return {
      message: 'Authentication error',
      cause: 'Your session may have expired or you lack the necessary permissions',
      solution: 'Try refreshing the page or logging in again'
    };
  }

  // File-related errors
  if (error.message.includes('No file provided')) {
    return {
      message: 'No file was provided for upload',
      cause: 'The file selection was empty or the file was removed before upload',
      solution: 'Please select a PDF file to upload'
    };
  }

  if (error.message.includes('Unsupported file format')) {
    return {
      message: 'Unsupported file format',
      cause: 'The file you selected is not in an accepted format',
      solution: 'Please upload a PDF file'
    };
  }

  if (error.message.includes('File size exceeds')) {
    return {
      message: 'File size exceeds the maximum limit',
      cause: 'The file you are trying to upload is too large',
      solution: 'Please upload a file smaller than 20MB or compress the current file'
    };
  }

  // Extract error message from API response
  let apiErrorMessage = '';
  if (error.message.includes('API error')) {
    const match = error.message.match(/API error \(\d+\): (.*)/);
    if (match && match[1]) {
      apiErrorMessage = match[1];
    }
  }

  // Default case
  return {
    message: apiErrorMessage || error.message || 'An error occurred',
    cause: 'An unexpected error occurred while communicating with the API',
    solution: 'Please try again or contact support if the problem persists'
  };
};
