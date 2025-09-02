import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaFileAlt, FaRobot, FaChartLine } from 'react-icons/fa';
import './Home.css';

const Home = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  const iconVariants = {
    hover: {
      scale: 1.2,
      rotate: 360,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
  <section className="home-section text-center">
      <motion.div 
        className="container py-5"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.h1 
          className="display-4 fw-bold mb-3" 
          style={{ color: 'var(--primary-color)' }}
          variants={itemVariants}
        >
          Welcome to PolySumm
        </motion.h1>
        
        <motion.p 
          className="lead mb-4"
          variants={itemVariants}
        >
        Scientific paper summarization made easy and efficient with AI-powered tools.
        </motion.p>

        <motion.div 
          className="features-container mb-5"
          variants={itemVariants}
        >
          <div className="features-row">
            <motion.div 
              className="feature-item"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <motion.div
                className="feature-icon"
                variants={iconVariants}
                whileHover="hover"
              >
                <FaFileAlt size={40} />
              </motion.div>
              <h3>Upload Papers</h3>
              <p>Upload your scientific papers in PDF format</p>
            </motion.div>

            <motion.div 
              className="feature-item"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <motion.div
                className="feature-icon"
                variants={iconVariants}
                whileHover="hover"
              >
                <FaRobot size={40} />
              </motion.div>
              <h3>AI Analysis</h3>
              <p>Get intelligent summaries using advanced AI</p>
            </motion.div>

            <motion.div 
              className="feature-item"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <motion.div
                className="feature-icon"
                variants={iconVariants}
                whileHover="hover"
              >
                <FaChartLine size={40} />
              </motion.div>
              <h3>Smart Insights</h3>
              <p>Extract key insights and main points</p>
            </motion.div>
          </div>
        </motion.div>

        <motion.div
          variants={itemVariants}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
      <Link 
        to="/upload" 
            className="btn btn-lg get-started-btn"
        style={{ 
          backgroundColor: 'var(--btn-bg)', 
          borderColor: 'var(--btn-bg)',
          color: 'white'
        }}
      >
        Get Started
      </Link>
        </motion.div>
      </motion.div>
  </section>
);
};

export default Home;

