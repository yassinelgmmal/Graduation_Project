import React, { useContext } from 'react';
import { SessionContext } from '../../context/SessionContext';
import './ThemeSwitcher.css';

const ThemeSwitcher = () => {
  const { theme, setTheme } = useContext(SessionContext);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <button
      className="btn theme-switcher"
      onClick={toggleTheme}
      aria-label="Toggle dark/light theme"
      title="Toggle Dark/Light Theme"
      style={{ backgroundColor: 'var(--btn-bg)', color: 'var(--text-color)', borderColor: 'var(--btn-bg)' }}
      onMouseOver={(e) => (e.currentTarget.style.backgroundColor = 'var(--btn-hover-bg)')}
      onMouseOut={(e) => (e.currentTarget.style.backgroundColor = 'var(--btn-bg)')}
    >
      {theme === 'light' ? '🌙 Dark' : '☀️ Light'}
    </button>
  );
};

export default ThemeSwitcher;
