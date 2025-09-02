import React from 'react';
import './About.css';

const About = () => {
  return (
    <section className="about-section">
      <h2 className="about-title">About PolySumm</h2>
      <p className="about-text">
        PolySumm is a cutting-edge scientific paper summarization platform designed to help researchers, students, and professionals extract key insights quickly and accurately. Using advanced AI techniques and multiple summarization methods, PolySumm delivers concise summaries, enabling efficient knowledge acquisition.
      </p>
      <p className="about-text">
        Whether you want to process papers using custom models, analyze specific paper components via APIs, or send full papers for end-to-end summarization, PolySumm supports multiple workflows tailored to your needs.
      </p>
      <p className="about-text">
        The platform is designed with a modern, sleek UI featuring light and dark themes, session persistence, and smooth transitions to ensure an intuitive user experience.
      </p>
    </section>
  );
};

export default About;
