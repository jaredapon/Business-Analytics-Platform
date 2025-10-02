import React from 'react';
import { MdClose } from 'react-icons/md';
import styles from './ConfirmationModal.module.css';

interface ConfirmationModalProps {
  isOpen: boolean;
  title: string;
  message: string | React.ReactNode;
  confirmText?: string;
  cancelText?: string;
  onConfirm: () => void;
  onCancel: () => void;
  isLoading?: boolean;
  children?: React.ReactNode;
  variant?: 'default' | 'warning' | 'danger' | 'success';
}

export const ConfirmationModal: React.FC<ConfirmationModalProps> = ({
  isOpen,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  onConfirm,
  onCancel,
  isLoading = false,
  children,
  variant = 'default'
}) => {
  if (!isOpen) return null;

  const getVariantClasses = () => {
    switch (variant) {
      case 'warning':
        return {
          modal: styles.warningModal,
          confirmButton: styles.warningConfirmButton
        };
      case 'danger':
        return {
          modal: styles.dangerModal,
          confirmButton: styles.dangerConfirmButton
        };
      case 'success':
        return {
          modal: styles.successModal,
          confirmButton: styles.successConfirmButton
        };
      default:
        return {
          modal: '',
          confirmButton: styles.defaultConfirmButton
        };
    }
  };

  const variantClasses = getVariantClasses();

  return (
    <div className={styles.modalOverlay}>
      <div className={`${styles.modal} ${variantClasses.modal}`}>
        <div className={styles.modalHeader}>
          <h3>{title}</h3>
          <button 
            className={styles.closeButton} 
            onClick={onCancel}
            disabled={isLoading}
            aria-label="Close modal"
          >
            <MdClose />
          </button>
        </div>
        
        <div className={styles.modalContent}>
          {typeof message === 'string' ? <p>{message}</p> : message}
          {children}
        </div>
        
        <div className={styles.modalActions}>
          <button 
            className={styles.cancelButton} 
            onClick={onCancel}
            disabled={isLoading}
          >
            {cancelText}
          </button>
          <button 
            className={`${styles.confirmButton} ${variantClasses.confirmButton}`}
            onClick={onConfirm}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className={styles.spinner}></span>
                Loading...
              </>
            ) : (
              confirmText
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmationModal;