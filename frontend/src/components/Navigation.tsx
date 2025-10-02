import React from 'react';
import { 
  MdDashboard, 
  MdUpload, 
  MdSettings, 
  MdLogout,
  MdPerson
} from 'react-icons/md';
import styles from './Navigation.module.css';

interface NavigationProps {
  currentPage: string;
  onPageChange: (page: string) => void;
  userName?: string;
}

export const Navigation: React.FC<NavigationProps> = ({ 
  currentPage, 
  onPageChange, 
  userName = "Book Latte"
}) => {
  const navigationItems = [
    { id: 'dashboard', label: 'Dashboard', icon: MdDashboard },
    { id: 'upload', label: 'Data Upload', icon: MdUpload },
    { id: 'settings', label: 'Settings', icon: MdSettings },
  ];

  return (
    <>
      {/* Top Navigation Bar */}
      <nav className={styles.topNav}>
        <div className={styles.topNavLeft}>
          <div className={styles.logo}>
            <h2>Book Latte Analytics</h2>
          </div>
        </div>
        
        <div className={styles.topNavRight}>
          <div className={styles.userProfile}>
            <MdPerson />
            <span>{userName}</span>
          </div>
          
          <button 
            className={styles.logoutButton}
            aria-label="Logout"
          >
            <MdLogout />
          </button>
        </div>
      </nav>

      {/* Sidebar Navigation */}
      <aside className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <h3>MENU</h3>
        </div>
        
        <nav className={styles.sidebarNav}>
          {navigationItems.map((item) => {
            const IconComponent = item.icon;
            return (
              <button
                key={item.id}
                className={`${styles.navItem} ${currentPage === item.id ? styles.navItemActive : ''}`}
                onClick={() => onPageChange(item.id)}
              >
                <IconComponent className={styles.navIcon} />
                <span className={styles.navLabel}>{item.label}</span>
              </button>
            );
          })}
        </nav>
        
        <div className={styles.sidebarFooter}>
          <div className={styles.sidebarSection}>
            <h4>OTHERS</h4>
            <button className={styles.navItem}>
              <MdLogout className={styles.navIcon} />
              <span className={styles.navLabel}>Log Out</span>
            </button>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Navigation;