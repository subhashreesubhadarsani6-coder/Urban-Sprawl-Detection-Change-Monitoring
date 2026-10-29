import React, { useState } from 'react';
import { 
  Bell, 
  Search, 
  Cpu, 
  ExternalLink, 
  CheckCircle2, 
  HelpCircle, 
  Activity,
  ChevronDown
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { ApiService } from '../../services/api';

export interface HeaderProps {
  onToggleSidebar?: () => void;
}

export const Header: React.FC<HeaderProps> = () => {
  const [showNotifications, setShowNotifications] = useState(false);

  const notifications = [
    { id: 1, title: 'Las Vegas Sprawl Audit Complete', time: '12m ago', unread: true },
    { id: 2, title: 'SpaceNet 7 Dataset Synchronized', time: '1h ago', unread: false },
    { id: 3, title: 'Training Epoch 28 Reached 86.8% mIoU', time: '2h ago', unread: false }
  ];

  return (
    <header className="h-14 border-b border-slate-800/80 bg-[#080c14]/90 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between sticky top-0 z-30">
      {/* Left section: Breadcrumb / Quick search */}
      <div className="flex items-center gap-4 flex-1 max-w-md">
        <div className="relative w-full">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search AOIs, coordinates, satellite scenes..."
            className="w-full bg-[#0d1322] border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-colors"
          />
        </div>
      </div>

      {/* Right section: System Status, Notifications, Profile */}
      <div className="flex items-center gap-3">
        {/* Backend & Engine Status Pill */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-[#0d1322] border border-slate-800 text-xs">
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
          </span>
          <span className="text-slate-400">Engine:</span>
          <span className="text-cyan-400 font-mono text-[11px] font-medium">
            {ApiService.isLiveBackend ? 'REST API (Live)' : 'Mock GIS / Demo'}
          </span>
        </div>

        {/* GPU node telemetry */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900/60 border border-slate-800/60 text-xs text-slate-400 font-mono">
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span>GPU-01</span>
          <span className="text-emerald-400">92% Idle</span>
        </div>

        {/* Notifications toggle */}
        <div className="relative">
          <button
            onClick={() => setShowNotifications(!showNotifications)}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 relative transition-colors"
            aria-label="Notifications"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-cyan-500"></span>
          </button>

          {showNotifications && (
            <div className="absolute right-0 mt-2 w-80 rounded-xl bg-[#0d1322] border border-slate-800 shadow-2xl py-2 z-50">
              <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800/80">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">Alerts & Telemetry</span>
                <span className="text-[10px] text-cyan-400 cursor-pointer hover:underline">Mark all read</span>
              </div>
              <div className="divide-y divide-slate-800/50 max-h-64 overflow-y-auto">
                {notifications.map(n => (
                  <div key={n.id} className="p-3 hover:bg-slate-850/40 transition-colors cursor-pointer flex gap-2.5">
                    <Activity className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-medium text-slate-200">{n.title}</p>
                      <span className="text-[10px] text-slate-500">{n.time}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Quick Docs Link */}
        <Link
          to="/docs"
          className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 transition-colors"
          title="GIS Formulas & API Reference"
        >
          <HelpCircle className="w-4 h-4" />
        </Link>

        {/* User avatar / profile button */}
        <div className="flex items-center gap-2 pl-2 border-l border-slate-800">
          <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center text-xs font-bold text-white shadow">
            US
          </div>
          <span className="hidden lg:inline text-xs font-medium text-slate-300">GIS Analyst</span>
        </div>
      </div>
    </header>
  );
};
