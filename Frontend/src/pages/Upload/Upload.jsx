// filepath: f:\University\GP Final\Frontend\src\pages\Upload\Upload.jsx
import React, { useState, useContext, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import OptionCard from '../../components/OptionCard/OptionCard';
import { SessionContext } from '../../context/SessionContext';
import { ingestDocument } from '../../services/api';
import { getApiErrorExplanation } from '../../services/apiTesting';
import { API_CONFIG } from '../../config';
import './Upload.css';

const options = [
  {
    key: 'custom-models',
    title: 'AI-Powered Summary',
    description: 'Get a comprehensive summary using our specialized in-house models, optimized for scientific papers with accurate handling of text, tables, and figures. This option offers faster processing and specialized scientific paper analysis.',
  },
  {
    key: 'external-api-full',
    title: 'Expert Analysis',
    description: 'Use external large language models (LLMs) to analyze and summarize your paper. This option provides deeper contextual insights and nuanced understanding but may take longer to process. Recommended for complex papers requiring detailed analysis.',
  },
];

const Upload = () => {
  const { processingOption, setProcessingOption, setDocumentId } = useContext(SessionContext);
  const [selectedOption, setSelectedOption] = useState(processingOption || '');
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [errorDetails, setErrorDetails] = useState(null);
  const [summaryLength, setSummaryLength] = useState('');
  const [lengthError, setLengthError] = useState('');
  const [tempLength, setTempLength] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleOptionSelect = (key) => {
    setSelectedOption(key);
    setProcessingOption(key);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    const maxSize = 20 * 1024 * 1024; // 20MB in bytes

    if (!selectedFile) {
      setFile(null);
      return;
    }

    if (selectedFile.type !== 'application/pdf') {
      setError('Only PDF files are supported.');
      setErrorDetails({
        cause: 'The file you selected is not a PDF.',
        solution: 'Please convert your file to PDF format before uploading.'
      });
      e.target.value = ''; // Clear the file input
      setFile(null);
      return;
    }

    if (selectedFile.size > maxSize) {
      setError('File size exceeds 20MB limit.');
      setErrorDetails({
        cause: 'The file you are trying to upload is larger than the maximum allowed size.',
        solution: 'Try to compress the PDF file or use a smaller document.'
      });
      e.target.value = ''; // Clear the file input
      setFile(null);
      return;
    }

    setFile(selectedFile);
    setError('');
    setErrorDetails(null);
  };

  const validateLength = (value) => {
    if (value === '') {
      setLengthError('');
      return true;
    }

    const numValue = parseInt(value);
    
    if (isNaN(numValue)) {
      setLengthError('Please enter a valid number');
      return false;
    }

    if (numValue > 500) {
      setLengthError('Maximum summary length is 500 words');
      return false;
    }
    
    if (numValue < 50) {
      setLengthError('Minimum summary length is 50 words');
      return false;
    }

    setLengthError('');
    return true;
  };

  const handleSummaryLengthChange = (e) => {
    const value = e.target.value;
    setTempLength(value);
    
    if (value === '') {
      setSummaryLength('');
      setLengthError('');
    } else {
      const numValue = parseInt(value);
      if (!isNaN(numValue)) {
        setSummaryLength(numValue);
      }
    }
  };

  const handleSummaryLengthBlur = () => {
    if (tempLength === '') {
      setLengthError('');
      return;
    }

    const numValue = parseInt(tempLength);
    if (isNaN(numValue)) {
      setLengthError('Please enter a valid number');
      return;
    }

    if (numValue > 500) {
      setLengthError('Maximum summary length is 500 words');
      setTempLength('500');
      setSummaryLength(500);
    } else if (numValue < 50) {
      setLengthError('Minimum summary length is 50 words');
      setTempLength('50');
      setSummaryLength(50);
    } else {
      setLengthError('');
      setTempLength(numValue.toString());
      setSummaryLength(numValue);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedOption) {
      setError('Please select a summarization option.');
      return;
    }
    if (!file) {
      setError('Please upload a PDF file.');
      return;
    }
    if (tempLength && !validateLength(tempLength)) {
      return;
    }

    setIsLoading(true);
    setError('');
    setErrorDetails(null);

    try {
      // Clear all chat-related data from sessionStorage
      sessionStorage.removeItem('chatHistory');
      sessionStorage.removeItem('chatMessages');
      sessionStorage.removeItem('currentChat');
      sessionStorage.removeItem('chatContext');
      sessionStorage.removeItem('chatState');
      // Proactively clear any previous documentId from both sessionStorage and localStorage
      sessionStorage.removeItem('documentId');
      localStorage.removeItem('documentId');
      // Store summary length and processing option in sessionStorage
      if (summaryLength) {
        sessionStorage.setItem('summaryLength', summaryLength.toString());
      }
      sessionStorage.setItem('processingOption', selectedOption);
      sessionStorage.setItem('uploadStatus', 'pending');
      // Navigate to processing page immediately
      navigate('/processing');
      // Start the file upload after navigation - pass the processing option
      const response = await ingestDocument(file, selectedOption);
      // Save the document ID and update upload status
      sessionStorage.setItem('documentId', response.document_id);
      localStorage.setItem('documentId', response.document_id);
      sessionStorage.setItem('uploadStatus', 'completed');
      setDocumentId(response.document_id);
    } catch (error) {
      console.error('Error uploading file:', error);
      const errorInfo = getApiErrorExplanation(error);
      sessionStorage.setItem('uploadStatus', 'failed');
      sessionStorage.setItem('uploadError', JSON.stringify({
        message: errorInfo.message,
        cause: errorInfo.cause,
        solution: errorInfo.solution
      }));
      setError(errorInfo.message);
      setErrorDetails({
        cause: errorInfo.cause,
        solution: errorInfo.solution
      });
      navigate('/upload');
    } finally {
      setIsLoading(false);
    }
  };

  // Reset error state when component unmounts
  useEffect(() => {
    return () => {
      setError('');
      setErrorDetails(null);
    };
  }, []);

  return (
    <section className="upload-section container py-4">
      <h2 className="mb-4" style={{ color: 'var(--primary-color)' }}>Upload Your Scientific Paper</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4 d-flex justify-content-center gap-4">
          {options.map((opt) => (
            <OptionCard
              key={opt.key}
              title={opt.title}
              description={opt.description}
              selected={selectedOption === opt.key}
              onSelect={() => handleOptionSelect(opt.key)}
            />
          ))}
        </div>
        
        <div className="mb-3">
          <label htmlFor="fileUpload" className="form-label">Choose PDF file:</label>
          <input
            id="fileUpload"
            type="file"
            accept="application/pdf"
            className="form-control"
            onChange={handleFileChange}
            aria-describedby="fileHelp"
          />
          <div id="fileHelp" className="form-text">
            Max size: 20MB. Supported format: PDF
          </div>
        </div>
        
        <div className="mb-3">
          <label htmlFor="summaryLength" className="form-label">Summary Length (words) - Optional:</label>
          <input
            id="summaryLength"
            type="number"
            min="50"
            max="500"
            value={tempLength}
            onChange={handleSummaryLengthChange}
            onBlur={handleSummaryLengthBlur}
            className={`form-control ${lengthError ? 'is-invalid' : ''}`}
            aria-describedby="lengthHelp"
            placeholder="Enter length (50-500 words) or leave empty for default"
          />
          <div id="lengthHelp" className="form-text">
            Optional: Enter desired summary length (50-500 words). Leave empty for default length.
          </div>
          {lengthError && (
            <div className="invalid-feedback">
              {lengthError}
            </div>
          )}
        </div>
        
        {error && (
          <div className="alert alert-danger" role="alert">
            <div className="d-flex align-items-center">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" className="bi bi-exclamation-triangle-fill flex-shrink-0 me-2" viewBox="0 0 16 16" role="img" aria-label="Warning:">
                <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/>
              </svg>
              <div>
                <strong>Error:</strong> {error}
              </div>
            </div>
            
            {errorDetails && (
              <div className="mt-2 ps-4">
                <div className="error-details">
                  <div className="mb-1">
                    <strong>Cause:</strong> {errorDetails.cause}
                  </div>
                  <div>
                    <strong>Solution:</strong> {errorDetails.solution}
                  </div>
                </div>
                
                {error.includes('Network error') && (
                  <div className="mt-2 small">
                    <strong>API URL:</strong> {API_CONFIG.BASE_URL}
                    <div>
                      Make sure the API server is running and accessible at the configured URL.
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        
        <div className="mb-3 d-flex justify-content-between align-items-center">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!file || !selectedOption || isLoading}
            style={{ backgroundColor: 'var(--btn-bg)', borderColor: 'var(--btn-bg)' }}
            onMouseOver={(e) => (e.currentTarget.style.backgroundColor = 'var(--btn-hover-bg)')}
            onMouseOut={(e) => (e.currentTarget.style.backgroundColor = 'var(--btn-bg)')}
          >
            {isLoading ? (
              <>
                <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Uploading...
              </>
            ) : (
              'Start Summarization'
            )}
          </button>
        </div>
      </form>
    </section>
  );
};

export default Upload;
