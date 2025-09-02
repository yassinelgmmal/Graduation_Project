import React from 'react';
import './Hero.css';

const Hero = () => {
  return (
    <section className="hero-section">
      <div className="hero-content">
        <h1 className="hero-title">PolySumm: Smart Scientific Paper Summarization</h1>
        <p className="hero-subtitle">
          Upload your scientific papers and get precise, multi-method summaries in seconds.
        </p>
        <button className="hero-cta" onClick={() => window.location.href = '/upload'}>
          Get Started
        </button>
      </div>
    </section>
  );
};

export default Hero;
