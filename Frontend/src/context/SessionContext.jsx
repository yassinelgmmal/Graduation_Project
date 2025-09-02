import React, { createContext, useState, useEffect } from 'react';

export const SessionContext = createContext();

export const SessionProvider = ({ children }) => {
  const [summary, setSummary] = useState(() => JSON.parse(localStorage.getItem('summary')) || '');
  const [chatHistory, setChatHistory] = useState(() => JSON.parse(localStorage.getItem('chatHistory')) || []);
  const [processingOption, setProcessingOption] = useState(() => localStorage.getItem('processingOption') || '');
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  const [documentId, setDocumentId] = useState(() => localStorage.getItem('documentId') || '');
  const [sessionError, setSessionError] = useState(null);  const [sessionInfo, setSessionInfo] = useState(() => {
    const savedInfo = localStorage.getItem('sessionInfo');
    return savedInfo ? JSON.parse(savedInfo) : {
      lastActiveTimestamp: Date.now()
    };
  });  
    // Update session info when document ID changes
  useEffect(() => {
    if (documentId) {
      const now = Date.now();
      setSessionInfo(prevInfo => {
        const updatedInfo = {
          lastActiveTimestamp: now
        };
        localStorage.setItem('sessionInfo', JSON.stringify(updatedInfo));
        return updatedInfo;
      });
    }
  }, [documentId]);

  // Persist session data to localStorage
  useEffect(() => {
    localStorage.setItem('summary', JSON.stringify(summary));
  }, [summary]);

  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
  }, [chatHistory]);

  useEffect(() => {
    localStorage.setItem('processingOption', processingOption);
  }, [processingOption]);

  useEffect(() => {
    localStorage.setItem('theme', theme);
    if (theme === 'dark') {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }, [theme]);

  useEffect(() => {
    localStorage.setItem('documentId', documentId);
  }, [documentId]);
    // More robust session reset
  const resetSession = () => {
    setSummary('');
    setChatHistory([]);
    setProcessingOption('');
    setDocumentId('');
    setSessionError(null);
    
    // Update timestamp
    setSessionInfo({
      lastActiveTimestamp: Date.now()
    });
    
    // Clear session storage items
    sessionStorage.removeItem('chatMessages');
    sessionStorage.removeItem('summaryLength');
    sessionStorage.removeItem('paperTopic');
    sessionStorage.removeItem('chatContext');
    sessionStorage.removeItem('chatState');
    
    // Clear localStorage items
    localStorage.removeItem('summary');
    localStorage.removeItem('chatHistory');
    localStorage.removeItem('processingOption');
    localStorage.removeItem('documentId');
  };
    /**
   * Recover previous session by document ID
   * @param {string} docId - The document ID to recover
   * @returns {boolean} - Whether recovery was successful
   */
  const recoverSession = (docId) => {
    try {
      if (!docId) {
        setSessionError('No document ID provided');
        return false;
      }
      
      setDocumentId(docId);
      
      // Update session info
      const updatedInfo = {
        ...sessionInfo,
        lastActiveTimestamp: Date.now()
      };
      setSessionInfo(updatedInfo);
      localStorage.setItem('sessionInfo', JSON.stringify(updatedInfo));
      
      return true;
    } catch (error) {
      setSessionError(`Failed to recover session: ${error.message}`);
      return false;
    }
  };

  return (
    <SessionContext.Provider value={{
      summary,
      setSummary,
      chatHistory,
      setChatHistory,
      processingOption,
      setProcessingOption,
      theme,
      setTheme,
      documentId,
      setDocumentId,
      sessionError,
      setSessionError,
      sessionInfo,
      resetSession,
      recoverSession
    }}>
      {children}
    </SessionContext.Provider>
  );
};


