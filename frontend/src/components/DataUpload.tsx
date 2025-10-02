import React, { useState, useCallback, useRef } from 'react';
import { 
  MdUploadFile, 
  MdClose, 
  MdDelete,
  MdDescription,
  MdTableChart,
  MdWarning
} from 'react-icons/md';
import ConfirmationModal from './ConfirmationModal';
import styles from './DataUpload.module.css';

interface FileWithPreview extends File {
  id: string;
  preview?: string;
}

const ACCEPTED_FILE_TYPES = {
  'text/csv': ['.csv'],
  'application/vnd.ms-excel': ['.xls'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
};

const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200MB
const MAX_FILES = 12;

export const DataUpload: React.FC = () => {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [errors, setErrors] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const generateFileId = () => `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const validateFile = (file: File): string | null => {
    if (file.size > MAX_FILE_SIZE) {
      return `File "${file.name}" is too large. Maximum size is ${MAX_FILE_SIZE / 1024 / 1024}MB.`;
    }

    const acceptedTypes = Object.keys(ACCEPTED_FILE_TYPES);
    const acceptedExtensions = Object.values(ACCEPTED_FILE_TYPES).flat();
    
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    const isValidType = acceptedTypes.includes(file.type) || acceptedExtensions.includes(fileExtension);
    
    if (!isValidType) {
      return `File "${file.name}" has an unsupported format. Accepted formats: CSV, Excel (.xls, .xlsx)`;
    }

    return null;
  };

  const processFiles = useCallback((fileList: FileList) => {
    const newErrors: string[] = [];
    const validFiles: FileWithPreview[] = [];

    if (files.length + fileList.length > MAX_FILES) {
      newErrors.push(`Cannot upload more than ${MAX_FILES} files at once.`);
      setErrors(newErrors);
      return;
    }

    Array.from(fileList).forEach((file) => {
      const error = validateFile(file);
      if (error) {
        newErrors.push(error);
      } else {
        const fileWithId = Object.assign(file, { id: generateFileId() }) as FileWithPreview;
        validFiles.push(fileWithId);
      }
    });

    if (newErrors.length > 0) {
      setErrors(newErrors);
    } else {
      setErrors([]);
    }

    setFiles(prev => [...prev, ...validFiles]);
  }, [files.length]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0) {
      processFiles(droppedFiles);
    }
  }, [processFiles]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      processFiles(selectedFiles);
    }
  }, [processFiles]);

  const removeFile = useCallback((fileId: string) => {
    setFiles(prev => prev.filter(file => file.id !== fileId));
    setErrors(prev => prev.filter(error => !error.includes(files.find(f => f.id === fileId)?.name || '')));
  }, [files]);

  const clearAllFiles = useCallback(() => {
    setFiles([]);
    setErrors([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (files.length === 0) return;
    
    setIsUploading(true);
    
    try {
      // Simulate upload progress for each file
      for (const file of files) {
        setUploadProgress(prev => ({ ...prev, [file.id]: 0 }));
        
        // Simulate upload progress
        for (let progress = 0; progress <= 100; progress += 10) {
          await new Promise(resolve => setTimeout(resolve, 100));
          setUploadProgress(prev => ({ ...prev, [file.id]: progress }));
        }
      }
      
      // Here you would make actual API calls to upload files
      console.log('Uploading files:', files);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Reset state after successful upload
      setFiles([]);
      setUploadProgress({});
      setShowConfirmation(false);
      
      // Show success message (you can implement a toast notification here)
      alert('Files uploaded successfully!');
      
    } catch (error) {
      console.error('Upload failed:', error);
      setErrors(['Upload failed. Please try again.']);
    } finally {
      setIsUploading(false);
    }
  }, [files]);

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    
    switch (extension) {
      case 'csv':
      case 'xlsx':
      case 'xls':
        return <MdTableChart size={24} />;
      default:
        return <MdDescription size={24} />;
    }
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className={styles.dataUploadContainer}>
      <div className={styles.dataUploadHeader}>
        <h1>Data Upload Center</h1>
        <p>Upload your data files for analysis. Supported formats: CSV, Excel (.xls, .xlsx)</p>
      </div>

      {/* Error Messages */}
      {errors.length > 0 && (
        <div className={styles.dataUploadErrorContainer}>
          {errors.map((error, index) => (
            <div key={index} className={styles.dataUploadErrorMessage}>
              {error}
            </div>
          ))}
        </div>
      )}

      {/* Drop Zone */}
      <div
        className={`${styles.dataUploadDropZone} ${isDragOver ? styles.dragOver : ''} ${files.length > 0 ? styles.hasFiles : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={openFileDialog}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={Object.values(ACCEPTED_FILE_TYPES).flat().join(',')}
          onChange={handleFileInput}
          className={styles.dataUploadHiddenInput}
        />
        
        <div className={styles.dataUploadDropZoneContent}>
          <div className={styles.dataUploadIcon}>
            <MdUploadFile size={48} />
          </div>
          <h3>Drop files here or click to browse</h3>
          <p>Support for CSV and Excel files</p>
          <p className={styles.dataUploadLimits}>
            Maximum {MAX_FILES} files • {MAX_FILE_SIZE / 1024 / 1024}MB per file
          </p>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className={styles.dataUploadFileList}>
          <div className={styles.dataUploadFileListHeader}>
            <h3>Selected Files ({files.length})</h3>
            <button 
              className={styles.dataUploadClearAllButton} 
              onClick={clearAllFiles}
              disabled={isUploading}
            >
              <MdDelete style={{ marginRight: '4px' }} />
              Clear All
            </button>
          </div>
          
          <div className={styles.dataUploadFileItems}>
            {files.map((file) => (
              <div key={file.id} className={styles.dataUploadFileItem}>
                <div className={styles.dataUploadFileIcon}>
                  {getFileIcon(file.name)}
                </div>
                
                <div className={styles.dataUploadFileDetails}>
                  <div className={styles.dataUploadFileName}>{file.name}</div>
                  <div className={styles.dataUploadFileMetadata}>
                    <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                    <span>•</span>
                    <span>{file.type || 'Unknown type'}</span>
                  </div>
                  
                  {uploadProgress[file.id] !== undefined && (
                    <div className={styles.dataUploadProgressBar}>
                      <div 
                        className={styles.dataUploadProgressFill}
                        style={{ width: `${uploadProgress[file.id]}%` }}
                      />
                      <span className={styles.dataUploadProgressText}>
                        {uploadProgress[file.id]}%
                      </span>
                    </div>
                  )}
                </div>
                
                <button
                  className={styles.dataUploadRemoveButton}
                  onClick={() => removeFile(file.id)}
                  disabled={isUploading}
                  aria-label={`Remove ${file.name}`}
                >
                  <MdClose />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload Actions */}
      {files.length > 0 && (
        <div className={styles.dataUploadActions}>
          <button
            className={styles.dataUploadButton}
            onClick={() => setShowConfirmation(true)}
            disabled={isUploading}
          >
            Upload {files.length} File{files.length > 1 ? 's' : ''}
          </button>
        </div>
      )}

      {/* Confirmation Modal */}
      <ConfirmationModal
        isOpen={showConfirmation}
        title="Confirm Upload"
        message={
          <div>
            <p>You are about to upload <strong>{files.length}</strong> file(s):</p>
            <div className={styles.dataUploadFileReviewList}>
              {files.map((file) => (
                <div key={file.id} className={styles.dataUploadFileReviewItem}>
                  <div className={styles.dataUploadFileInfo}>
                    <span className={styles.dataUploadFileName}>{file.name}</span>
                    <span className={styles.dataUploadFileSize}>
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                  <span className={styles.dataUploadFileType}>
                    {file.type || 'Unknown'}
                  </span>
                </div>
              ))}
            </div>
            <div className={styles.dataUploadWarning}>
              <p><MdWarning style={{ marginRight: '8px', verticalAlign: 'middle' }} />This action cannot be undone. Please review your files before proceeding.</p>
            </div>
          </div>
        }
        confirmText={isUploading ? 'Uploading...' : 'Confirm Upload'}
        onConfirm={handleUpload}
        onCancel={() => setShowConfirmation(false)}
        isLoading={isUploading}
        variant="warning"
      />
    </div>
  );
};

export default DataUpload;