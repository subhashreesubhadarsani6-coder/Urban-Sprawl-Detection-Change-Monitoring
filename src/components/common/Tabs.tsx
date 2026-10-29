import React from 'react';
import clsx from 'clsx';

export interface TabItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  badge?: string | number;
}

export interface TabsProps {
  tabs: TabItem[];
  activeTab: string;
  onChange: (id: string) => void;
  variant?: 'underline' | 'pills';
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({
  tabs,
  activeTab,
  onChange,
  variant = 'pills',
  className
}) => {
  if (variant === 'underline') {
    return (
      <div className={clsx('flex border-b border-slate-800 space-x-6', className)}>
        {tabs.map(tab => {
          const isActive = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              className={clsx(
                'flex items-center gap-2 pb-3 pt-1 text-sm font-medium transition-all relative',
                isActive
                  ? 'text-cyan-400 border-b-2 border-cyan-400'
                  : 'text-slate-400 hover:text-slate-200'
              )}
            >
              {tab.icon && <span>{tab.icon}</span>}
              <span>{tab.label}</span>
              {tab.badge !== undefined && (
                <span className={clsx(
                  'text-[10px] px-1.5 py-0.5 rounded-full font-mono',
                  isActive ? 'bg-cyan-950 text-cyan-300' : 'bg-slate-800 text-slate-400'
                )}>
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div className={clsx('inline-flex p-1 rounded-xl bg-[#090d18] border border-slate-800/80', className)}>
      {tabs.map(tab => {
        const isActive = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className={clsx(
              'flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150',
              isActive
                ? 'bg-[#18233c] text-cyan-300 shadow-md border border-cyan-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/40'
            )}
          >
            {tab.icon && <span>{tab.icon}</span>}
            <span>{tab.label}</span>
            {tab.badge !== undefined && (
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-800 text-slate-300 font-mono">
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
