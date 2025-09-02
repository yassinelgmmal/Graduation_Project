import React from 'react';
import { FaBook, FaRobot, FaComments, FaLightbulb, FaChartLine, FaShieldAlt } from 'react-icons/fa';
import './About.css';

const About = () => {
  return (
    <div className="about-container">
      <div className="about-header">
        <h1>About PolySumm</h1>
        <p className="about-subtitle">Your AI-Powered Research Assistant</p>
      </div>

      <div className="about-content">
        <section className="about-section">
          <div className="section-icon">
            <FaBook />
          </div>
          <div className="section-content">
            <h2>What is PolySumm?</h2>
            <p>
              PolySumm is an innovative AI-powered platform designed to help researchers, students, and academics 
              quickly understand and analyze research papers. Our advanced natural language processing technology 
              breaks down complex academic papers into clear, concise summaries while maintaining the essential 
              information and context.
            </p>
          </div>
        </section>

        <section className="about-section">
          <div className="section-icon">
            <FaLightbulb />
          </div>
          <div className="section-content">
            <h2>Our Mission</h2>
            <p>
              We aim to make academic research more accessible and efficient by providing intelligent tools that 
              help users quickly grasp the key concepts, methodologies, and findings of research papers. Our goal 
              is to save researchers valuable time while ensuring they don't miss critical information.
            </p>
          </div>
        </section>

        <section className="features-section">
          <h2>Key Features</h2>
          <div className="features-grid">
            <div className="feature-card">
              <FaRobot className="feature-icon" />
              <h3>AI-Powered Summarization</h3>
              <p>Advanced algorithms that understand and condense research papers while preserving key information</p>
            </div>
            <div className="feature-card">
              <FaComments className="feature-icon" />
              <h3>Interactive Chat</h3>
              <p>Engage in meaningful discussions about the paper's content with our intelligent chatbot</p>
            </div>
            <div className="feature-card">
              <FaChartLine className="feature-icon" />
              <h3>Topic Analysis</h3>
              <p>Get insights into the main topics and themes of the research paper</p>
            </div>
            <div className="feature-card">
              <FaShieldAlt className="feature-icon" />
              <h3>Secure & Private</h3>
              <p>Your research papers and data are handled with the utmost security and privacy</p>
            </div>
          </div>
        </section>

        <section className="about-section">
          <div className="section-icon">
            <FaComments />
          </div>
          <div className="section-content">
            <h2>How It Works</h2>
            <div className="steps-container">
              <div className="step">
                <span className="step-number">1</span>
                <h3>Upload Your Paper</h3>
                <p>Simply upload your research paper in PDF format</p>
              </div>
              <div className="step">
                <span className="step-number">2</span>
                <h3>AI Analysis</h3>
                <p>Our AI processes the paper and generates a comprehensive summary</p>
              </div>
              <div className="step">
                <span className="step-number">3</span>
                <h3>Interactive Discussion</h3>
                <p>Chat with our AI to explore the paper's content in detail</p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default About;
