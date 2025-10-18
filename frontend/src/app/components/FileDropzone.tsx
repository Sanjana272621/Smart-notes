'use client';

import React, { useCallback, useRef, useState } from 'react';

type Props = {
  onFileSelect: (file: File) => void;
  isUploading?: boolean;
};

export default function FileDropzone({ onFileSelect, isUploading }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const dragCounter = useRef(0); // avoids flicker when dragging over children

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];

      const isPdf =
        file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');

      if (!isPdf) {
        alert('Only PDF files are accepted.');
        // reset so the same wrong file chosen again still triggers change
        if (inputRef.current) inputRef.current.value = '';
        return;
      }

      onFileSelect(file);

      // IMPORTANT: reset the input value so selecting the SAME file again fires onChange
      if (inputRef.current) inputRef.current.value = '';
    },
    [onFileSelect]
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => handleFiles(e.target.files),
    [handleFiles]
  );

  const onClick = useCallback(() => {
    if (!isUploading) inputRef.current?.click();
  }, [isUploading]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (!isUploading) inputRef.current?.click();
      }
    },
    [isUploading]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter.current = 0;
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const onDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) setIsDragging(false);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const onPaste = useCallback(
    (e: React.ClipboardEvent<HTMLDivElement>) => {
      // Some browsers support pasting files from clipboard
      const items = e.clipboardData?.files;
      if (items && items.length) handleFiles(items);
    },
    [handleFiles]
  );

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload PDF"
        onClick={onClick}
        onKeyDown={onKeyDown}
        onDrop={onDrop}
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onPaste={onPaste}
        className={[
          'flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 transition focus:outline-none',
          isDragging ? 'border-indigo-600 bg-indigo-50/60' : 'border-slate-300 hover:bg-slate-50',
          isUploading ? 'cursor-not-allowed opacity-70' : 'cursor-pointer',
        ].join(' ')}
      >
        <p className="text-base font-medium">
          {isUploading ? 'Uploading…' : 'Drag & drop your PDF here'}
        </p>
        <p className="mt-1 text-sm text-slate-500">or click to choose a file</p>

        {/* Hidden input that actually opens the OS file picker */}
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={onInputChange}
          disabled={isUploading}
        />
      </div>
    </div>
  );
}
