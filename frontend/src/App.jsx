import React from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import LandingPage from './pages/LandingPage';
import DashboardPage from './pages/DashboardPage';
import ConfigPage from './pages/ConfigPage';
import MonitorPage from './pages/MonitorPage';
import ExplorerPage from './pages/ExplorerPage';
import EvolutionPage from './pages/EvolutionPage';
import ComparisonPage from './pages/ComparisonPage';
import BestArchitecturePage from './pages/BestArchitecturePage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route element={<Layout />}>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/configure" element={<ConfigPage />} />
          <Route path="/monitor" element={<MonitorPage />} />
          <Route path="/monitor/:id" element={<MonitorPage />} />
          <Route path="/explorer" element={<ExplorerPage />} />
          <Route path="/evolution" element={<EvolutionPage />} />
          <Route path="/compare" element={<ComparisonPage />} />
          <Route path="/best" element={<BestArchitecturePage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
