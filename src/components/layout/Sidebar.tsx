import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Globe2, 
  LayoutDashboard, 
  UploadCloud, 
  SplitSquareVertical, 
  Database, 
  BrainCircuit, 
  Award, 
  BookOpen, 
  ChevronLeft, 
  ChevronRight, 
  Satellite, 
  Layers,
  Menu,
  X
} from 'lucide-react';
import clsx from 'clsx';

interface SidebarProps {
  isMobileOpen?: boolean;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isMobileOpen = false,
  onCloseMobile
}) => {
  const [collapsed, setCollapsed] = useState(false);

  const navItems = [
    { to: '/dashboard', label: 'GIS Dashboard', icon: <LayoutDashboard className="w-4 h-4" /> },
    { to: '/analysis/new', label: 'Sprawl Analysis', icon: <UploadCloud className="w-4 h-4" />, badge: 'Upload' },
    { to: '/analysis/results/job-vegas-2018-2024', label: 'Results & Swipe', icon: <SplitSquareVertical className="w-4 h-4" /> },
    { to: '/datasets', label: 'Satellite Datasets', icon: <Database className="w-4 h-4" />, count: '4' },
    { to: '/training', label: 'Model Training', icon: <BrainCircuit className="w-4 h-4" />, pulse: true },
    { to: '/evaluation', label: 'Evaluation & Benchmarks', icon: <Award className="w-4 h-4" /> },
    { to: '/docs', label: 'API & GIS Docs', icon: <BookOpen className="w-4 h-4" /> },
  ];

  const sidebarContent = (
    <div className="flex flex-col h-full bg-[#080c14] border-r border-slate-800/80">
      {/* Brand header */}
      <div className="h-14 px-4 flex items-center justify-between border-b border-slate-800/80">
        <NavLink to="/" className="flex items-center gap-2.5 overflow-hidden group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-700 flex items-center justify-center text-white shadow-lg shadow-cyan-900/30 shrink-0 group-hover:scale-105 transition-transform">
            <Satellite className="w-4 h-4 text-cyan-100" />
          </div>
          {!collapsed && (
            <div className="flex flex-col">
              <span className="font-bold text-sm tracking-wider text-white flex items-center gap-1.5">
                URBANSCOPE <span className="text-[10px] px-1 py-0.2 rounded bg-cyan-950 text-cyan-400 border border-cyan-500/30">AI</span>
              </span>
              <span className="text-[10px] text-slate-400 tracking-tight">Spatio-Temporal GIS</span>
            </div>
          )}
        </NavLink>

        {/* Desktop collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="hidden lg:flex p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800/80 transition-colors"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>

        {/* Mobile close button */}
        {onCloseMobile && (
          <button
            onClick={onCloseMobile}
            className="lg:hidden p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation Links */}
      <div className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
        <div className="px-2 pb-2">
          {!collapsed && (
            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
              Workstation
            </span>
          )}
        </div>

        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            onClick={onCloseMobile}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all group relative',
                isActive
                  ? 'bg-cyan-950/40 text-cyan-300 border border-cyan-500/30 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-[#0d1322]'
              )
            }
            title={collapsed ? item.label : undefined}
          >
            <span className="shrink-0">{item.icon}</span>
            {!collapsed && (
              <span className="flex-1 truncate">{item.label}</span>
            )}
            {!collapsed && item.badge && (
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-cyan-900/60 text-cyan-300 border border-cyan-500/20">
                {item.badge}
              </span>
            )}
            {!collapsed && item.pulse && (
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            )}
            {!collapsed && item.count && (
              <span className="text-[10px] font-mono text-slate-500 bg-slate-850 px-1.5 py-0.5 rounded">
                {item.count}
              </span>
            )}
          </NavLink>
        ))}

        <div className="pt-4 px-2 pb-2">
          {!collapsed && (
            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
              Public Portal
            </span>
          )}
        </div>
        <NavLink
          to="/"
          onClick={onCloseMobile}
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-[#0d1322] transition-colors"
          title={collapsed ? 'Landing Page' : undefined}
        >
          <Globe2 className="w-4 h-4 shrink-0 text-slate-500 group-hover:text-cyan-400" />
          {!collapsed && <span>Platform Overview</span>}
        </NavLink>
      </div>

      {/* Footer telemetry card */}
      {!collapsed && (
        <div className="p-3 m-2 rounded-xl bg-[#0d1322] border border-slate-800 text-[11px] space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="flex items-center gap-1.5 font-medium">
              <Layers className="w-3.5 h-3.5 text-cyan-400" /> Active Tiles
            </span>
            <span className="font-mono text-slate-300">4 AOIs</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div className="bg-gradient-to-r from-cyan-500 to-emerald-400 h-full w-[68%]" />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 font-mono">
            <span>Buffer: 1.2 GB</span>
            <span>Sentinel-2 L2A</span>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <>
      {/* Desktop sidebar */}
      <aside
        className={clsx(
          'hidden lg:flex flex-col shrink-0 transition-all duration-300',
          collapsed ? 'w-16' : 'w-64'
        )}
      >
        {sidebarContent}
      </aside>

      {/* Mobile drawer overlay */}
      {isMobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-black/75 backdrop-blur-sm"
            onClick={onCloseMobile}
          />
          <div className="relative w-64 max-w-xs h-full z-10 shadow-2xl animate-slide-right">
            {sidebarContent}
          </div>
        </div>
      )}
    </>
  );
};
