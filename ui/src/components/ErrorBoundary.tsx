// ErrorBoundary.tsx
'use client';

import React, { ReactNode, ErrorInfo, Component } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(_: Error): ErrorBoundaryState {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // You can log the error to an error reporting service
    console.error("Error caught by ErrorBoundary:", error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="flex flex-col items-center justify-center h-full min-h-[200px] p-8 text-center">
            <div className="text-red-400 text-4xl mb-4">⚠</div>
            <h2 className="text-lg font-semibold text-gray-100 mb-2">Something went wrong</h2>
            <p className="text-gray-400 text-sm mb-6">An unexpected error occurred. Check the browser console for details.</p>
            <button
              onClick={() => this.setState({ hasError: false })}
              className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white text-sm rounded-md mr-3"
            >
              Try again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-gray-300 text-sm rounded-md"
            >
              Reload page
            </button>
          </div>
        )
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;