import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ToastProvider } from './context/ToastContext';
import { AppLayout } from './components/layout/AppLayout';
import { LandingPage } from './pages/LandingPage';
import { DashboardPage } from './pages/DashboardPage';
import { AnalysisUploadPage } from './pages/AnalysisUploadPage';
import { ResultsPage } from './pages/ResultsPage';
import { DatasetExplorerPage } from './pages/DatasetExplorerPage';
import { ModelTrainingPage } from './pages/ModelTrainingPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { DocumentationPage } from './pages/DocumentationPage';

export const App: React.FC = () => {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Landing Portal */}
          <Route path="/" element={<LandingPage />} />

          {/* GIS Workstation Dashboard App */}
          <Route element={<AppLayout />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/analysis/new" element={<AnalysisUploadPage />} />
            <Route path="/analysis/results/:id" element={<ResultsPage />} />
            <Route path="/analysis/results" element={<Navigate to="/analysis/results/job-vegas-2018-2024" replace />} />
            <Route path="/datasets" element={<DatasetExplorerPage />} />
            <Route path="/training" element={<ModelTrainingPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/docs" element={<DocumentationPage />} />
          </Route>

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  );
};

export default App;
