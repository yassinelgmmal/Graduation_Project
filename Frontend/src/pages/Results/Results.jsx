import React, { useState, useEffect, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { SessionContext } from '../../context/SessionContext';
import { FaFileAlt, FaRobot } from 'react-icons/fa';
import './Results.css';

const Results = () => {
  const { summary, documentId } = useContext(SessionContext);
  const navigate = useNavigate();
  const [paperTopic, setPaperTopic] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    if (!summary) {
      navigate('/upload');
    }
  }, [summary, navigate]);
  useEffect(() => {
    const fetchPaperTopic = async () => {
      if (!summary) return;
      setIsLoading(true);
      try {
        // Get processing option from sessionStorage
        const processingOption = sessionStorage.getItem('processingOption');
        
        // Check if we already have a paper topic from the custom-models API
        const storedTopic = sessionStorage.getItem('paperTopic');
        if (storedTopic && storedTopic !== 'Unknown' && processingOption === 'custom-models') {
          setPaperTopic(storedTopic);
          setIsLoading(false);
          return;
        }
        
        let topic = null;
        // Use the provided /predict endpoint for external-api-full
        try {
          const response = await fetch('http://20.75.49.202:8010/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: summary })
          });
          if (response.ok) {
            const data = await response.json();
            if (data && data.predicted_label) topic = data.predicted_label;
            else if (data && data.topic) topic = data.topic;
          }
        } catch (topicError) {
          console.warn('Error using /predict endpoint:', topicError);
        }
        // Fallback: use previous logic if /predict fails
        if (!topic) {
          const topicPatterns = [
            /(?:paper|article|research|study) (?:on|about|discusses|examines|investigates|explores) (.+?)[.]/i,
            /(?:in the field of|in) (.+?)[,.](?=\s)/i,
            /(?:focuses on|addresses) (.+?)[.]/i
          ];
          for (const pattern of topicPatterns) {
            const match = summary.match(pattern);
            if (match && match[1] && match[1].length > 3 && match[1].length < 100) {
              topic = match[1].trim();
              break;
            }
          }
        }
        if (!topic) {
          topic = 'Academic Research';
        }
        setPaperTopic(topic);
        sessionStorage.setItem('paperTopic', topic);
      } catch (error) {
        console.error('Error determining paper topic:', error);
        const fallbackTopic = 'Scientific Paper';
        setPaperTopic(fallbackTopic);
        sessionStorage.setItem('paperTopic', fallbackTopic);
      } finally {
        setIsLoading(false);
      }
    };
    if (summary) {
      fetchPaperTopic();
    }
  }, [summary, documentId]);

  if (!summary) {
    return null;
  }

  return (
    <div className="results-container">
      <div className="results-header">
        <h1>Analysis Results</h1>
      </div>

      <div className="results-content">
        <div className="results-section paper-topic-section">
          <div className="section-header">
            <FaFileAlt className="section-icon" />
            <h2>Paper Topic</h2>
          </div>          <div className="paper-topic-content">
            {isLoading ? (
              <p>Paper is classified in field of: "Loading..."</p>
            ) : paperTopic ? (
              <p>Paper is classified in field of: "{paperTopic}"</p>
            ) : (
              <p>Paper is classified in field of: "Unknown"</p>
            )}
          </div>
        </div>

        <div className="results-section summary-section">
          <div className="section-header">
            <FaRobot className="section-icon" />
            <h2>AI Summary</h2>
          </div>
          <div className="summary-content">
            <p>{summary}</p>
          </div>
        </div>

        <div className="action-buttons">
      <button
            className="chat-button"
            onClick={() => navigate('/chat')}
      >
            Chat About Summary
      </button>
        </div>
      </div>
    </div>
  );
};

export default Results;
