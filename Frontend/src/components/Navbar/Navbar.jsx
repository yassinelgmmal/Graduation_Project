import React, { useContext } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { SessionContext } from '../../context/SessionContext';
import ThemeSwitcher from '../ThemeSwitcher/ThemeSwitcher';
import './Navbar.css';

const Navbar = () => {
  const { resetSession } = useContext(SessionContext);
  const navigate = useNavigate();

  const handleUploadNewPaper = () => {
    // Remove documentId from both sessionStorage and localStorage
    sessionStorage.removeItem('documentId');
    localStorage.removeItem('documentId');
    resetSession();
    navigate('/upload');
  };

  return (
    <nav className="navbar navbar-expand-lg navbar-custom" style={{ backgroundColor: 'var(--nav-bg)' }}>
      <div className="container">
        <Link to="/" className="navbar-brand fw-bold fs-3" style={{ color: 'var(--primary-color)' }}>
          PolySumm
        </Link>
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarSupportedContent"
          aria-controls="navbarSupportedContent"
          aria-expanded="false"
          aria-label="Toggle navigation"
          style={{ borderColor: 'var(--primary-color)' }}
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarSupportedContent">
          <ul className="navbar-nav me-auto mb-2 mb-lg-0">
            <li className="nav-item mx-2">
              <Link to="/" className="nav-link" style={{ color: 'var(--text-color)' }}>
                Home
              </Link>
            </li>
            <li className="nav-item mx-2">
              <Link to="/results" className="nav-link" style={{ color: 'var(--text-color)' }}>
                Summary
              </Link>
            </li>
            <li className="nav-item mx-2">
              <Link to="/chat" className="nav-link" style={{ color: 'var(--text-color)' }}>
                Chat
              </Link>
            </li>
            <li className="nav-item mx-2">
              <Link to="/about" className="nav-link" style={{ color: 'var(--text-color)' }}>
                About
              </Link>
            </li>
          </ul>
          <div className="d-flex align-items-center gap-3">
            <button
              className="btn btn-primary"
              onClick={handleUploadNewPaper}
              style={{ backgroundColor: 'var(--btn-bg)', borderColor: 'var(--btn-bg)' }}
              onMouseOver={(e) => (e.currentTarget.style.backgroundColor = 'var(--btn-hover-bg)')}
              onMouseOut={(e) => (e.currentTarget.style.backgroundColor = 'var(--btn-bg)')}
            >
              Upload New Paper
            </button>
            <ThemeSwitcher />
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

