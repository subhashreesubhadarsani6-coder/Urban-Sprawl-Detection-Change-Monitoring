import React from 'react';
import { Layers } from 'lucide-react';
import { Button } from './Button';

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  actionLabel,
  onAction,
  className
}) => {
  return (
    <div className={`flex flex-col items-center justify-center text-center p-8 sm:p-12 border border-dashed border-slate-800 rounded-2xl bg-[#090d18]/60 ${className || ''}`}>
      <div className="w-14 h-14 rounded-2xl bg-cyan-950/40 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-4 shadow-lg shadow-cyan-950/30">
        {icon || <Layers className="w-7 h-7" />}
      </div>
      <h3 className="text-lg font-semibold text-white tracking-wide mb-1.5">{title}</h3>
      <p className="text-sm text-slate-400 max-w-md mb-6 leading-relaxed">{description}</p>
      {actionLabel && onAction && (
        <Button onClick={onAction} variant="primary" size="md">
          {actionLabel}
        </Button>
      )}
    </div>
  );
};
