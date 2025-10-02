import React, { useState, useCallback, useRef } from 'react';
import { 
  MdUploadFile, 
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
  const [selectedFileIds, setSelectedFileIds] = useState<Set<string>>(new Set());
  const [showRemoveSelectedConfirm, setShowRemoveSelectedConfirm] = useState(false);
  const [showClearAllConfirm, setShowClearAllConfirm] = useState(false);
  const [filePendingRemoval, setFilePendingRemoval] = useState<FileWithPreview | null>(null);

  const cleanFileName = useCallback((name: string) => {
    const dot = name.lastIndexOf('.');
    const base = dot > 0 ? name.slice(0, dot) : name;
    return base.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
  }, []);

  const getExtension = (name: string) => name.split('.').pop()?.toLowerCase() || 'unknown';

  const getDisplayName = useCallback((name: string) => {
    const base = cleanFileName(name);
    const ext = getExtension(name);
    return ext === 'unknown' ? base : `${base}.${ext}`;
  }, [cleanFileName]);

  const formatBytes = useCallback((bytes: number) => {
    if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${bytes} B`;
  }, []);

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

  // Internal single remove logic and also update selected set
  const removeFile = useCallback((fileId: string) => {
    setFiles(prev => prev.filter(file => file.id !== fileId));
    setSelectedFileIds(prev => {
      const next = new Set(prev);
      next.delete(fileId);
      return next;
    });
    setErrors(prev => prev.filter(error => !error.includes(files.find(f => f.id === fileId)?.name || '')));
  }, [files]);

  // Clear all files
  const clearAllFiles = useCallback(() => {
    setFiles([]);
    setErrors([]);
    setSelectedFileIds(new Set());
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  // Selection helpers
  const isSelected = useCallback((id: string) => selectedFileIds.has(id), [selectedFileIds]);
  const toggleSelect = useCallback((id: string) => {
    setSelectedFileIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);
  const selectAll = useCallback(() => {
    setSelectedFileIds(new Set(files.map(f => f.id)));
  }, [files]);
  const clearSelection = useCallback(() => setSelectedFileIds(new Set()), []);
  const allSelected = files.length > 0 && selectedFileIds.size === files.length;

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

  // Bulk remove selected confirm action
  const confirmRemoveSelected = () => {
    setFiles(prev => prev.filter(f => !selectedFileIds.has(f.id)));
    setSelectedFileIds(new Set());
    setShowRemoveSelectedConfirm(false);
  };

  // Single file remove confirm action
  const confirmRemoveSingle = () => {
    if (!filePendingRemoval) return;
    removeFile(filePendingRemoval.id);
    setFilePendingRemoval(null);
  };

  // Clear all confirm action
  const confirmClearAll = () => {
    clearAllFiles();
    setShowClearAllConfirm(false);
  };

  // Helpers
  const totalSizeFormatted = formatBytes(files.reduce((sum, f) => sum + f.size, 0));
  const typeCounts = files.reduce<Record<string, number>>((acc, f) => {
    const ext = getExtension(f.name);
    acc[ext] = (acc[ext] || 0) + 1;
    return acc;
  }, {});

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
            <MdUploadFile size={48} color="black"/>
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
            <div className={styles.dataUploadFileListHeaderLeft}>
              <h3>Selected Files ({files.length})</h3>
              <label className={styles.dataUploadSelectAll}>
                <input
                  type="checkbox"
                  className={styles.dataUploadFileCheckbox}
                  checked={allSelected}
                  onChange={() => (allSelected ? clearSelection() : selectAll())}
                />
                <span style={{ marginRight: '0.5rem' }}>
                  {allSelected ? 'Unselect all' : 'Select all'}
                </span>
              </label>
            </div>
            <div className={styles.dataUploadHeaderActions}>
              <button
                className={styles.dataUploadSecondaryButton}
                onClick={openFileDialog}
                disabled={isUploading}
              >
                Add Files
              </button>
              <button
                className={styles.dataUploadClearAllButton}
                onClick={() => setShowRemoveSelectedConfirm(true)}
                disabled={isUploading || selectedFileIds.size === 0}
                title={selectedFileIds.size === 0 ? 'No files selected' : `Remove ${selectedFileIds.size} selected`}
              >
                <MdDelete style={{ marginRight: '4px' }} />
                Remove Selected
              </button>
              <button
                className={styles.dataUploadClearAllButton}
                onClick={() => setShowClearAllConfirm(true)}
                disabled={isUploading}
              >
                <MdDelete style={{ marginRight: '4px' }} />
                Clear All
              </button>
            </div>
          </div>

          <div className={styles.dataUploadFileItems}>
            {files.map((file) => (
              <div
                key={file.id}
                className={`${styles.dataUploadFileItem} ${isSelected(file.id) ? styles.selected : ''}`}
              >
                <input
                  type="checkbox"
                  className={styles.dataUploadFileCheckbox}
                  checked={isSelected(file.id)}
                  onChange={() => toggleSelect(file.id)}
                  aria-label={`Select ${file.name}`}
                />
                <div className={styles.dataUploadFileIcon}>
                  {getFileIcon(file.name)}
                </div>

                <div className={styles.dataUploadFileDetails}>
                  <div
                    className={styles.dataUploadFileName}
                    title={file.name}
                  >
                    {getDisplayName(file.name)}
                  </div>
                  <div className={styles.dataUploadFileMetadata}>
                    <span>{formatBytes(file.size)}</span>
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
                  onClick={() => setFilePendingRemoval(file)}
                  disabled={isUploading}
                  aria-label={`Remove ${file.name}`}
                  title="Remove file"
                >
                  x
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

      {/* Confirmation Modal - Upload (redesigned content) */}
      <ConfirmationModal
        isOpen={showConfirmation}
        title="Review and Confirm Upload"
        message={
          <div>
            <div className={styles.dataUploadConfirmSummary}>
              <div className={styles.dataUploadStat}>
                <div className={styles.dataUploadStatLabel}>Total Files</div>
                <div className={styles.dataUploadStatValue}>{files.length}</div>
              </div>
              <div className={styles.dataUploadStat}>
                <div className={styles.dataUploadStatLabel}>Total Size</div>
                <div className={styles.dataUploadStatValue}>{totalSizeFormatted}</div>
              </div>
              <div className={styles.dataUploadStat}>
                <div className={styles.dataUploadStatLabel}>By Type</div>
                <div className={styles.dataUploadStatValue}>
                  {Object.entries(typeCounts).map(([ext, count], i) => (
                    <span key={ext}>
                      {ext.toUpperCase()}: {count}{i < Object.entries(typeCounts).length - 1 ? ' • ' : ''}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className={styles.dataUploadDivider} />

            <div className={styles.dataUploadFileReviewList}>
              {files.map((file) => (
                <div key={file.id} className={styles.dataUploadFileReviewItem}>
                  <div className={styles.dataUploadFileInfo}>
                    <span className={styles.dataUploadFileName} title={file.name}>
                      {getDisplayName(file.name)}
                    </span>
                    <span className={styles.dataUploadFileSize}>
                      {formatBytes(file.size)}
                    </span>
                  </div>
                  <span className={styles.dataUploadFileType}>
                    {getExtension(file.name).toUpperCase()
                    }
                  </span>
                </div>
              ))}
            </div>

            <div className={styles.dataUploadWarning}>
              <p>
                <MdWarning style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                Please verify file names, sizes, and types before uploading. This action cannot be undone.
              </p>
            </div>
          </div>
        }
        confirmText={isUploading ? 'Uploading...' : 'Confirm Upload'}
        onConfirm={handleUpload}
        onCancel={() => setShowConfirmation(false)}
        isLoading={isUploading}
        variant="warning"
      />

      {/* Confirmation Modal - Remove Selected */}
      <ConfirmationModal
        isOpen={showRemoveSelectedConfirm}
        title="Remove selected files?"
        message={
          <div>
            <p>You are about to remove {selectedFileIds.size} file(s):</p>
            <div className={styles.dataUploadFileReviewList}>
              {files.filter(f => selectedFileIds.has(f.id)).map(f => (
                <div key={f.id} className={styles.dataUploadFileReviewItem}>
                  <div className={styles.dataUploadFileInfo}>
                    <span className={styles.dataUploadFileName} title={f.name}>
                      {getDisplayName(f.name)}
                    </span>
                    <span className={styles.dataUploadFileSize}>
                      {formatBytes(f.size)}
                    </span>
                  </div>
                  <span className={styles.dataUploadFileType}>{getExtension(f.name).toUpperCase()}</span>
                </div>
              ))}
            </div>
          </div>
        }
        confirmText="Remove"
        onConfirm={confirmRemoveSelected}
        onCancel={() => setShowRemoveSelectedConfirm(false)}
        isLoading={false}
        variant="danger"
      />

      {/* Confirmation Modal - Clear All */}
      <ConfirmationModal
        isOpen={showClearAllConfirm}
        title="Clear all files?"
        message={<p>This will remove all selected files from the list. This action cannot be undone.</p>}
        confirmText="Clear All"
        onConfirm={confirmClearAll}
        onCancel={() => setShowClearAllConfirm(false)}
        isLoading={false}
        variant="danger"
      />

      {/* Confirmation Modal - Single Remove */}
      <ConfirmationModal
        isOpen={!!filePendingRemoval}
        title="Remove file?"
        message={
          filePendingRemoval ? (
            <div className={styles.dataUploadFileReviewList}>
              <div className={styles.dataUploadFileReviewItem}>
                <div className={styles.dataUploadFileInfo}>
                  <span className={styles.dataUploadFileName} title={filePendingRemoval.name}>
                    {getDisplayName(filePendingRemoval.name)}
                  </span>
                  <span className={styles.dataUploadFileSize}>
                    {formatBytes(filePendingRemoval.size)}
                  </span>
                </div>
                <span className={styles.dataUploadFileType}>
                  {getExtension(filePendingRemoval.name).toUpperCase()}
                </span>
              </div>
            </div>
          ) : null
        }
        confirmText="Remove"
        onConfirm={confirmRemoveSingle}
        onCancel={() => setFilePendingRemoval(null)}
        isLoading={false}
        variant="danger"
      />
    </div>
  );
};

export default DataUpload;