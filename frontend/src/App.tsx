import { useState } from 'react';
import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import DataUpload from './components/DataUpload';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'upload':
        return <DataUpload />;
      case 'settings':
        return (
          <div style={{ padding: '2rem' }}>
            <h1>Settings</h1>
            <p>Settings page coming soon...</p>
          </div>
        );
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="App">
      <Navigation
        currentPage={currentPage}
        onPageChange={setCurrentPage}
        userName="Book Latte"
      />
      
      <main className="main-content">
        {renderCurrentPage()}
      </main>
    </div>
  );
}

export default App;
