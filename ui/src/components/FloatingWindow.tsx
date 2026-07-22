import React, { useEffect, useRef, useState } from 'react';

interface FloatingWindowProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  showCloseButton?: boolean;
  width?: string;
  initialPosition?: { x: number; y: number };
}

export const FloatingWindow: React.FC<FloatingWindowProps> = ({
  isOpen,
  onClose,
  title,
  children,
  showCloseButton = true,
  width = 'max-w-md',
  initialPosition = { x: 20, y: 80 },
}) => {
  const [position, setPosition] = useState(initialPosition);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const windowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleEscKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscKey);
    }

    return () => {
      document.removeEventListener('keydown', handleEscKey);
    };
  }, [isOpen, onClose]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (windowRef.current) {
      setIsDragging(true);
      setDragOffset({
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      });
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging && windowRef.current) {
        const newX = e.clientX - dragOffset.x;
        const newY = e.clientY - dragOffset.y;

        // Get window dimensions
        const rect = windowRef.current.getBoundingClientRect();
        const windowWidth = rect.width;
        const windowHeight = rect.height;

        // Get viewport dimensions
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        // Calculate bounded position
        // Keep at least some part of the header visible (top >= 0)
        // and prevent the window from being moved completely out of view
        const boundedX = Math.max(-windowWidth + 50, Math.min(newX, viewportWidth - 50));
        const boundedY = Math.max(0, Math.min(newY, viewportHeight - 50));

        setPosition({
          x: boundedX,
          y: boundedY,
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  if (!isOpen) return null;

  return (
    <div
      ref={windowRef}
      className={`fixed z-[60] ${width} rounded-lg bg-gray-800 border border-gray-700 shadow-2xl transition-shadow ${
        isDragging ? 'shadow-blue-900/20' : ''
      }`}
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
      }}
    >
      {/* Window header */}
      {(title || showCloseButton) && (
        <div
          className="flex items-center justify-between rounded-t-lg border-b border-gray-700 bg-gray-850 px-4 py-2 cursor-move select-none"
          onMouseDown={handleMouseDown}
        >
          {title && <h3 className="text-sm font-semibold text-gray-100">{title}</h3>}

          {showCloseButton && (
            <button
              type="button"
              className="ml-auto inline-flex items-center justify-center rounded-md p-1 text-gray-400 hover:bg-gray-700 hover:text-gray-200 focus:outline-none"
              onClick={onClose}
              aria-label="Close window"
            >
              <svg
                className="h-4 w-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      )}

      {/* Window content */}
      <div className="px-4 py-3">{children}</div>
    </div>
  );
};
