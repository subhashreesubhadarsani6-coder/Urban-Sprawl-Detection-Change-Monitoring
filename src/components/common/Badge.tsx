import React from 'react';
import clsx from 'clsx';
import { JobStatus } from '../../types';

export interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'completed' | 'processing' | 'queued' | 'failed' | 'warning' | 'emerald' | 'cyan' | 'purple';
  size?: 'sm' | 'md';
  dot?: boolean;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'default',
  size = 'md',
  dot = false,
  className
}) => {
  const variantStyles = {
    default: 'bg-slate-800/80 text-slate-300 border-slate-700/80',
    completed: 'bg-emerald-950/60 text-emerald-400 border-emerald-500/30',
    processing: 'bg-cyan-950/60 text-cyan-400 border-cyan-500/30 animate-pulse',
    queued: 'bg-amber-950/60 text-amber-300 border-amber-500/30',
    failed: 'bg-rose-950/60 text-rose-400 border-rose-500/30',
    warning: 'bg-orange-950/60 text-orange-400 border-orange-500/30',
    emerald: 'bg-emerald-950/60 text-emerald-400 border-emerald-500/30',
    cyan: 'bg-cyan-950/60 text-cyan-400 border-cyan-500/30',
    purple: 'bg-purple-950/60 text-purple-300 border-purple-500/30'
  };

  const dotColors = {
    default: 'bg-slate-400',
    completed: 'bg-emerald-400',
    processing: 'bg-cyan-400',
    queued: 'bg-amber-400',
    failed: 'bg-rose-400',
    warning: 'bg-orange-400',
    emerald: 'bg-emerald-400',
    cyan: 'bg-cyan-400',
    purple: 'bg-purple-400'
  };

  const sizeStyles = {
    sm: 'text-[11px] px-2 py-0.5 gap-1.5',
    md: 'text-xs px-2.5 py-1 gap-1.5'
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center font-medium rounded-full border tracking-wide uppercase',
        sizeStyles[size],
        variantStyles[variant],
        className
      )}
    >
      {dot && (
        <span className={clsx('w-1.5 h-1.5 rounded-full shrink-0', dotColors[variant])} />
      )}
      {children}
    </span>
  );
};

export const StatusBadge: React.FC<{ status: JobStatus; className?: string }> = ({ status, className }) => {
  const map: Record<JobStatus, { label: string; variant: 'completed' | 'processing' | 'queued' | 'failed' }> = {
    completed: { label: 'Completed', variant: 'completed' },
    processing: { label: 'In Progress', variant: 'processing' },
    queued: { label: 'Queued', variant: 'queued' },
    failed: { label: 'Failed', variant: 'failed' }
  };

  const config = map[status] || { label: status, variant: 'default' };

  return (
    <Badge variant={config.variant} dot className={className}>
      {config.label}
    </Badge>
  );
};
