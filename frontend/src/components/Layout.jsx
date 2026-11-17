import React, { useEffect, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  Activity,
  Award,
  Cpu,
  GitBranch,
  Home,
  Layers,
  LayoutDashboard,
  Menu,
  Scale,
  Sliders,
  Sparkles,
  X,
} from 'lucide-react';
import { api } from '../api/client';

const NAV_ITEMS = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/configure', label: 'New Experiment', icon: Sliders },
  { to: '/monitor', label: 'Live Monitor', icon: Activity },
  { to: '/explorer', label: 'Architecture Explorer', icon: Layers },
  { to: '/evolution', label: 'Evolution History', icon: GitBranch },
  { to: '/compare', label: 'Model Comparison', icon: Scale },
  { to: '/best', label: 'Best Architecture', icon: Award },
];

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [health, setHealth] = useState(null);
  const location = useLocation();

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const h = await api.getHealth();
        if (active) setHealth(h);
      } catch {
        if (active) setHealth(null);
      }
    };
    poll();
    const timer = setInterval(poll, 8000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-[#080B14] text-slate-100 flex">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:flex-col lg:w-64 border-r border-slate-800/80 bg-[#0B101E]/90 backdrop-blur-xl fixed inset-y-0 left-0 z-30">
        <div className="h-16 px-5 flex items-center justify-between border-b border-slate-800/80">
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-violet-600/30">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold tracking-wider text-sm text-white block">NEUROEVOLVE</span>
              <span className="text-[10px] font-mono text-cyan-400 tracking-widest block">NAS ENGINE v1.0</span>
            </div>
          </Link>
        </div>

        <nav className="flex-1 px-3 py-5 space-y-1 overflow-y-auto">
          <Link
            to="/"
            className="flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm text-slate-400 hover:text-white hover:bg-slate-800/60 transition-colors"
          >
            <Home className="w-4 h-4" />
            <span>Landing Home</span>
          </Link>
          <div className="my-3 border-t border-slate-800/70" />
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-violet-600/25 to-cyan-500/15 text-white border border-violet-500/40 shadow-violet-glow'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50 border border-transparent'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-800/80 space-y-2.5">
          <div className="p-3 rounded-xl bg-[#111827] border border-slate-800">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                PyTorch Engine
              </span>
              <span
                className={`inline-flex items-center gap-1.5 font-mono text-[11px] ${
                  health ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    health ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'
                  }`}
                />
                {health ? 'ONLINE' : 'OFFLINE'}
              </span>
            </div>
            {health && (
              <p className="mt-1.5 text-[11px] font-mono text-slate-500">
                Torch {health.pytorch_version} · {health.cpu_count} CPU Cores
              </p>
            )}
          </div>
          {health?.active_experiment_id && (
            <Link
              to={`/monitor/${health.active_experiment_id}`}
              className="flex items-center justify-between px-3 py-2 rounded-xl bg-violet-500/15 border border-violet-500/40 text-xs text-violet-200 hover:bg-violet-500/25 transition-colors"
            >
              <span className="flex items-center gap-2 font-medium">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                Search Active
              </span>
              <span className="font-mono text-[11px] text-cyan-300">View →</span>
            </Link>
          )}
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 inset-x-0 h-14 bg-[#0B101E]/95 backdrop-blur-md border-b border-slate-800 px-4 flex items-center justify-between z-40">
        <Link to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold tracking-wider text-sm">NEUROEVOLVE</span>
        </Link>
        <button
          onClick={() => setMobileOpen((o) => !o)}
          className="p-2 rounded-lg text-slate-300 hover:bg-slate-800"
          aria-label="Toggle Menu"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="lg:hidden fixed inset-0 top-14 bg-[#080B14]/95 backdrop-blur-xl z-30 p-4 space-y-1.5 overflow-y-auto">
          <Link
            to="/"
            className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm text-slate-300 hover:bg-slate-800/70"
          >
            <Home className="w-4 h-4" />
            Landing Home
          </Link>
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium ${
                  isActive
                    ? 'bg-violet-600/25 text-white border border-violet-500/40'
                    : 'text-slate-300 hover:bg-slate-800/70'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 lg:pl-64 pt-14 lg:pt-0 min-w-0">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
