import React, { useContext } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar/Navbar';
import Footer from './components/Footer/Footer';
import Home from './pages/Home/Home';
import Upload from './pages/Upload/Upload';
import Processing from './pages/Processing/Processing';
import Results from './pages/Results/Results';
import Chat from './pages/Chat/Chat';
import About from './pages/About/About';
import { SessionContext } from './context/SessionContext';
import './App.css';

const App = () => {
  const { theme } = useContext(SessionContext);

  return (
    <div className={`app ${theme}`}>
      <Router>
        <Navbar />
        <main className="flex-grow-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/processing" element={<Processing />} />
            <Route path="/results" element={<Results />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        <Footer />
      </Router>
    </div>
  );
};

export default App;


