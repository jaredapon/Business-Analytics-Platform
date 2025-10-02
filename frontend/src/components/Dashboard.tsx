import React from 'react';
import styles from './Dashboard.module.css';

export const Dashboard: React.FC = () => {
  return (
    <div className={styles.dashboard}>
      <div className={styles.dashboardHeader}>
        <div>
          <h1>Welcome back, Book Latte!</h1>
          <p>You are now signed in to your analytics dashboard.</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;