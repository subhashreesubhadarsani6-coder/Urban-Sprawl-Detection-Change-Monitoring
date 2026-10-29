import React from 'react';
import clsx from 'clsx';

export const Skeleton: React.FC<{ className?: string }> = ({ className }) => {
  return (
    <div
      className={clsx(
        'animate-pulse bg-slate-800/60 rounded-md',
        className
      )}
    />
  );
};

export const CardSkeleton: React.FC = () => {
  return (
    <div className="p-5 rounded-xl border border-slate-800 bg-[#0d1322] space-y-3">
      <div className="flex items-center justify-between">
        <Skeleton className="h-4 w-1/3" />
        <Skeleton className="h-6 w-16 rounded-full" />
      </div>
      <Skeleton className="h-8 w-1/2" />
      <Skeleton className="h-4 w-full" />
      <div className="pt-2 flex gap-2">
        <Skeleton className="h-4 w-20" />
        <Skeleton className="h-4 w-24" />
      </div>
    </div>
  );
};

export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 4 }) => {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex gap-4 p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 items-center">
          <Skeleton className="h-4 w-1/4" />
          <Skeleton className="h-4 w-1/4" />
          <Skeleton className="h-4 w-1/6" />
          <Skeleton className="h-6 w-16 rounded-full" />
        </div>
      ))}
    </div>
  );
};
