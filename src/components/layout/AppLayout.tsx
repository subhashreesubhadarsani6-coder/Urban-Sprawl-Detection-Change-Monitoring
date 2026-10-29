import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { Menu } from 'lucide-react';

export const AppLayout: React.FC = () => {
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#080c14] text-slate-100">
      {/* Sidebar */}
      <Sidebar
        isMobileOpen={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
      />

      {/* Main content column */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Top Header */}
        <div className="flex items-center">
          {/* Mobile hamburger menu */}
          <button
            onClick={() => setMobileSidebarOpen(true)}
            className="lg:hidden p-3 text-slate-400 hover:text-white bg-[#080c14] border-b border-slate-800"
            aria-label="Open navigation menu"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex-1 min-w-0">
            <Header onToggleSidebar={() => setMobileSidebarOpen(!mobileSidebarOpen)} />
          </div>
        </div>

        {/* Scrollable Viewport */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
};
