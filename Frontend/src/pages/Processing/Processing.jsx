import React, { useEffect, useContext, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { SessionContext } from '../../context/SessionContext';
import { getSummary } from '../../services/api';
import './Processing.css';

const Processing = () => {
  const { processingOption, setSummary, documentId, setDocumentId } = useContext(SessionContext);
  const [processingStatus, setProcessingStatus] = useState('Uploading your paper...');
  const [error, setError] = useState('');
  const [hasStartedSummary, setHasStartedSummary] = useState(false);
  const [processingPhase, setProcessingPhase] = useState('upload');
  const navigate = useNavigate();
  
  // Get model type description based on processing option
  const getModelDescription = () => {
    if (processingOption === 'custom-models') {
      return "Processing using our specialized in-house models";
    } else if (processingOption === 'external-api-full') {
      return "Processing using external large language models";
    }
    return "Processing your document";
  };

  // Separate the upload status check from the summary fetching
  const checkUploadStatus = useCallback(() => {
    const status = sessionStorage.getItem('uploadStatus');
    const storedId = sessionStorage.getItem('documentId');
    
    if (status === 'failed') {
      const errorData = JSON.parse(sessionStorage.getItem('uploadError') || '{}');
      setError(errorData.message || 'Upload failed');
      setProcessingStatus('Error occurred');
      setProcessingPhase('error');
      setTimeout(() => navigate('/upload'), 3000);
      return true;
    }
    
    if (status === 'completed' && storedId && !documentId) {
      setDocumentId(storedId);
      setProcessingPhase('summarize');
      return true;
    }
    
    return false;
  }, [navigate, setDocumentId, documentId]);

  const fetchSummary = useCallback(async () => {
    if (!documentId || hasStartedSummary) return;

    setHasStartedSummary(true);
    setProcessingPhase('summarize');
    
    try {
      const summaryLength = parseInt(sessionStorage.getItem('summaryLength') || '250');
      
      // Update status message based on processing option
      if (processingOption === 'custom-models') {
        setProcessingStatus('Generating AI-powered summary using our advanced models...');
      } else if (processingOption === 'external-api-full') {
        setProcessingStatus('Analyzing paper with expert LLM models...');
      }
      
      // Pass the processing option to getSummary
      const response = await getSummary(documentId, 'comprehensive', summaryLength, processingOption);
      
      // Set the summary from the response
      setSummary(response.summary);
      
      // If we get a paper topic directly from the custom-models API, store it
      if (processingOption === 'custom-models' && response.paperTopic) {
        sessionStorage.setItem('paperTopic', response.paperTopic);
      }
      
      sessionStorage.removeItem('uploadStatus');
      sessionStorage.removeItem('documentId');
      
      navigate('/results');
    } catch (error) {
      console.error('Error processing paper:', error);
      setError('An error occurred while processing your paper. Please try again.');
      setProcessingStatus('Error occurred');
      sessionStorage.removeItem('uploadStatus');
      sessionStorage.removeItem('documentId');
      setTimeout(() => navigate('/upload'), 3000);
    }
  }, [documentId, processingOption, setSummary, navigate, hasStartedSummary]);

  // Handle the initial check for processing option
  useEffect(() => {
    if (!processingOption) {
      navigate('/upload');
    }
  }, [processingOption, navigate]);

  // Handle checking upload status
  useEffect(() => {
    if (documentId) return; // Don't check if we already have a document ID

    const intervalId = setInterval(() => {
      if (checkUploadStatus()) {
        clearInterval(intervalId);
      }
    }, 1000);

    return () => clearInterval(intervalId);
  }, [documentId, checkUploadStatus]);

  // Handle fetching summary when document ID is available
  useEffect(() => {
    if (documentId && !hasStartedSummary) {
      fetchSummary();
    }
  }, [documentId, fetchSummary, hasStartedSummary]);

  return (
    <section className="processing-section d-flex flex-column justify-content-center align-items-center py-5">
      <div className="spinner-border text-primary" role="status" aria-label="Loading summary">
        <span className="visually-hidden">Loading...</span>
      </div>
      
      <h3 className="mt-4" style={{ color: 'var(--primary-color)' }}>{processingStatus}</h3>
      
      <div className="model-info mt-2 text-center">
        <span className="badge bg-info mb-2">{getModelDescription()}</span>
        <p className="small text-muted">
          {processingPhase === 'upload' 
            ? 'Uploading and parsing your document...' 
            : processingPhase === 'summarize' 
              ? 'Analyzing content and generating summary...' 
              : 'Finalizing results...'}
        </p>
      </div>
      
      {error && (
        <div className="alert alert-danger mt-3" role="alert">
          {error}
        </div>
      )}
    </section>
  );
};

export default Processing;
