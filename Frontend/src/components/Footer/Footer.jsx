import React from 'react';
import './Footer.css';

const Footer = () => (
  <footer className="footer text-center py-3" style={{ backgroundColor: 'var(--footer-bg)', color: 'var(--text-color)' }}>
    <div className="container">
      <small>
        © {new Date().getFullYear()} PolySumm. All rights reserved.
      </small>
    </div>
  </footer>
);

export default Footer;
