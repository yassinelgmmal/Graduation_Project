import React, { useState, useContext, useEffect, useRef } from 'react';
import { SessionContext } from '../../context/SessionContext';
import { FaUserCircle, FaRobot, FaTrash, FaLightbulb } from 'react-icons/fa';
import { askQuestionWithRetry, handleApiError, getFollowUpQuestionsWithRetry } from '../../services/chatApi';
import './Chat.css';

const Chat = () => {
  const { summary, documentId } = useContext(SessionContext);
  const [messages, setMessages] = useState(() => {
    const sessionMessages = sessionStorage.getItem('chatMessages');
    if (sessionMessages) {
      const parsedMessages = JSON.parse(sessionMessages);
      return parsedMessages.map(msg => ({ 
        ...msg, 
        isComplete: true, 
        showCursor: false 
      }));
    }
    if (summary) {
      const initialMessage = {
        role: 'bot',
        content: `Paper Topic: ${sessionStorage.getItem('paperTopic') || ''}\n\nSummary:\n${summary || ''}`,
        isComplete: true,
        showCursor: false
      };
      sessionStorage.setItem('chatMessages', JSON.stringify([initialMessage]));
      return [initialMessage];
    }
    return [];
  });
  const [input, setInput] = useState('');
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [typingMessageIndex, setTypingMessageIndex] = useState(null);
  const messagesEndRef = useRef(null);
  const paperTopic = sessionStorage.getItem('paperTopic');
  const inputRef = useRef(null);
  const typingIntervalRef = useRef(null);
  const [paperTopicApi, setPaperTopicApi] = useState('');

  // Fetch paper topic from external endpoint
  const fetchPaperTopic = async (summaryText) => {
    try {
      // Get processing option from sessionStorage
      const processingOption = sessionStorage.getItem('processingOption');
      
      // Check if we already have a paper topic from the custom-models API
      const storedTopic = sessionStorage.getItem('paperTopic');
      if (storedTopic && storedTopic !== 'Unknown' && processingOption === 'custom-models') {
        setPaperTopicApi(storedTopic);
        return;
      }
      
      // Only fetch from predict endpoint if using external-api-full
      const response = await fetch('http://20.75.49.202:8010/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: summaryText })
      });
      if (!response.ok) throw new Error('Failed to fetch topic');
      const data = await response.json();
      if (data && data.predicted_label) setPaperTopicApi(data.predicted_label);
      else if (data && data.topic) setPaperTopicApi(data.topic);
      else setPaperTopicApi('Unknown');
    } catch (err) {
      setPaperTopicApi('Unknown');
    }
  };

  // Fetch topic when summary is available
  useEffect(() => {
    if (summary) {
      fetchPaperTopic(summary);
    }
  }, [summary]);

  // Fetch follow-up questions from API after bot finishes typing
  const fetchFollowUpQuestions = async (lastUserQuestion) => {
    try {
      const result = await getFollowUpQuestionsWithRetry(lastUserQuestion, documentId);
      // Support both 'questions' and 'follow_up_questions' keys
      const questions = Array.isArray(result.follow_up_questions)
        ? result.follow_up_questions
        : (Array.isArray(result.questions) ? result.questions : []);
      setSuggestedQuestions(questions);
    } catch (error) {
      setSuggestedQuestions([]);
      console.error('Error fetching follow-up questions:', error);
    }
  };

  // Save messages to sessionStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      sessionStorage.setItem('chatMessages', JSON.stringify(messages));
    }
  }, [messages]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize chat with summary if it's a new paper
  useEffect(() => {
    const hasExistingChat = sessionStorage.getItem('chatMessages');
    if (summary && !hasExistingChat) {
      const initialMessage = {
        role: 'bot',
        content: `Paper Topic: ${paperTopic || ''}\n\nSummary:\n${summary || ''}`,
        isComplete: true,
        showCursor: false
      };
      setMessages([initialMessage]);
    }
  }, [summary, paperTopic]);

  // Cleanup typing effect on unmount
  useEffect(() => {
    return () => {
      if (typingIntervalRef.current) {
        clearTimeout(typingIntervalRef.current);
      }
    };
  }, []);

  // Function to simulate typing effect
  const typeMessage = (fullText, messageIndex, lastUserQuestion) => {
    let currentIndex = 0;
    setTypingMessageIndex(messageIndex);
    const typeNextCharacter = () => {
      if (currentIndex < fullText.length) {
        const nextChar = fullText[currentIndex];
        setMessages(prev => {
          const newMessages = [...prev];
          if (newMessages[messageIndex]) {
            newMessages[messageIndex].content = fullText.substring(0, currentIndex + 1);
            newMessages[messageIndex].showCursor = true;
          }
          return newMessages;
        });
        currentIndex++;
        let delay = 15;
        if (nextChar === ' ') delay = 25;
        else if (nextChar === '.' || nextChar === '!' || nextChar === '?') delay = 75;
        else if (nextChar === ',' || nextChar === ';') delay = 40;
        else if (nextChar === '\n') delay = 80;
        typingIntervalRef.current = setTimeout(typeNextCharacter, delay);
      } else {
        setMessages(prev => {
          const newMessages = [...prev];
          if (newMessages[messageIndex]) {
            newMessages[messageIndex].isComplete = true;
            newMessages[messageIndex].showCursor = false;
          }
          return newMessages;
        });
        setTypingMessageIndex(null);
        setIsLoading(false);
        setTimeout(() => {
          inputRef.current?.focus();
        }, 100);
        // Fetch follow-up questions only after bot finishes typing
        if (lastUserQuestion) fetchFollowUpQuestions(lastUserQuestion);
      }
    };
    typeNextCharacter();
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const userMessage = { role: 'user', content: input.trim(), isComplete: true };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);
    try {
      // Add a placeholder for the bot's response
      setMessages(prev => [...prev, { role: 'bot', content: '', isLoading: true, isComplete: false, showCursor: false }]);
      // Fetch the answer
      const response = await askQuestionWithRetry(userMessage.content, documentId);
      // Start typing effect for the response
      const messageIndex = messages.length + 1; // +1 because we added user message
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          role: 'bot',
          content: '',
          isComplete: false,
          isLoading: false,
          showCursor: false
        };
        return newMessages;
      });
      // Start typing the response
      const responseText = response.answer || 'I could not generate an answer for your question.';
      setSuggestedQuestions([]); // Hide suggestions until bot finishes
      typeMessage(responseText, messageIndex, userMessage.content); // Pass last user question
    } catch (err) {
      console.error('Error sending message:', err);
      // Remove the loading placeholder
      setMessages(prev => prev.filter(msg => !msg.isLoading));
      // Handle the error
      const errorInfo = handleApiError(err);
      setError(errorInfo);
      // Add an error message from the bot
      setMessages(prev => [
        ...prev,
        {
          role: 'bot',
          content: `I encountered an error: \n\n${errorInfo?.message || 'Unknown error'}`,
          isError: true,
          isComplete: true,
          showCursor: false
        }
      ]);
      setIsLoading(false);
      // Focus the input field after error
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
      setSuggestedQuestions([]);
    }
  };

  const clearChat = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      // Clear any ongoing typing
      if (typingIntervalRef.current) {
        clearTimeout(typingIntervalRef.current);
      }
      setTypingMessageIndex(null);
      
      setMessages(summary ? [{
        role: 'bot',
        content: `Paper Topic: ${paperTopic || ''}\n\nSummary:\n${summary || ''}`,
        isComplete: true,
        showCursor: false
      }] : []);
      sessionStorage.removeItem('chatMessages');
      setError(null);
    }
    setSuggestedQuestions([]);
  };

  const handleQuestionClick = (question) => {
    setInput(question);
  };

  // Restore the suggested questions button
  const handleSuggestedQuestionsButton = async () => {
    if (messages.length === 0) return;
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
    if (!lastUserMsg) return;
    await fetchFollowUpQuestions(lastUserMsg.content);
  };

  return (
    <section className="chat-section" aria-label="Chat with summary bot">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 style={{ color: 'var(--primary-color)', textAlign: 'center' }}>Chat About Summary</h2>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="send-btn"
            onClick={handleSuggestedQuestionsButton}
            aria-label="Get suggested questions"
            style={{ backgroundColor: 'var(--btn-bg)', borderColor: 'var(--btn-bg)', color: 'white' }}
          >
            <FaLightbulb style={{ marginRight: 4 }} /> Suggest Questions
          </button>
          <button
            className="send-btn"
            style={{ backgroundColor: 'white', color: 'var(--primary-color)', border: '1px solid var(--primary-color)' }}
            onClick={clearChat}
            aria-label="Clear chat"
          >
            <FaTrash style={{ marginRight: 4 }} /> Clear
          </button>
        </div>
      </div>
      {/* Display the topic fetched from the API */}
      {paperTopicApi && (
        <div style={{ marginBottom: 16, color: 'var(--primary-color)', fontWeight: 600, fontSize: '1.1rem' }}>
          Paper Topic: {paperTopicApi}
        </div>
      )}
      {error && error.message && !error.isHandled && (
        <div style={{ padding: '12px 15px', marginBottom: 15, background: '#fff8f8', border: '1px solid #ffd0d0', borderRadius: 5, color: '#d8000c' }}>
          <strong>Error:</strong> {error.message}
          {error.suggestion && <div><strong>Suggestion:</strong> {error.suggestion}</div>}
        </div>
      )}
      <div className="chat-box" tabIndex={0}>
        {messages.length === 0 && (
          <div style={{ color: '#888', textAlign: 'center', padding: 20 }}>
            <p>No messages yet. Start the conversation by asking a question about the document.</p>
          </div>
        )}
        {messages.map((message, index) => {
          // Show bot typing effect (bouncing dots) for the last bot message if isLoading and it's the last message
          const isLastBotBubble =
            isLoading &&
            message.role === 'bot' &&
            index === messages.length - 1 &&
            (!message.content || message.content.trim() === '');
          return (
            <div key={index} className={`chat-bubble ${message.role}${message.isError ? ' error' : ''}`}>  
              <div className="icon">
                {message.role === 'user' ?
                  <FaUserCircle size={20} color="#3a0941" /> :
                  <FaRobot size={20} color="#d500f6" />
                }
              </div>
              <div className="message-content">
                {isLastBotBubble ? (
                  <div className="typing-indicator bouncing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                ) : message.isLoading ? null : (
                  <div className="message-text">
                    {(message.content || '').split('\n').map((line, i, lines) => (
                      <React.Fragment key={i}>
                        {line}
                        {message.showCursor && i === lines.length - 1 && (
                          <span className="typing-cursor">|</span>
                        )}
                        <br />
                      </React.Fragment>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
        {suggestedQuestions.length > 0 && (
          <div className="suggested-questions">
            <h4 className="suggested-questions-title">Suggested Questions:</h4>
            <div className="suggested-questions-list">
              {suggestedQuestions.map((question, index) => (
                <button
                  key={index}
                  className="suggested-question-btn"
                  onClick={() => handleQuestionClick(question)}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginTop: '1rem' }}>
        <textarea
          className="chat-input"
          rows={3}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type your question here..."
          aria-label="Chat input"
          disabled={isLoading || typingMessageIndex !== null}
          ref={inputRef}
        />
        <button
          className="send-btn"
          onClick={handleSendMessage}
          aria-label="Send message"
          disabled={!input.trim() || isLoading || typingMessageIndex !== null}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </section>
  );
};

export default Chat;
