import React, { HTMLAttributes } from 'react';
import clsx from 'clsx';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  hoverEffect?: boolean;
  glow?: 'cyan' | 'sprawl' | 'none';
  variant?: 'default' | 'elevated' | 'glass';
}

export const Card: React.FC<CardProps> = ({
  children,
  className,
  hoverEffect = false,
  glow = 'none',
  variant = 'default',
  ...props
}) => {
  const variantStyles = {
    default: 'bg-[#0d1322] border border-[#1e293f]',
    elevated: 'bg-[#121a2d] border border-slate-800 shadow-xl',
    glass: 'glass-panel'
  };

  const glowStyles = {
    none: '',
    cyan: 'shadow-gis-glow border-cyan-500/40',
    sprawl: 'shadow-sprawl-glow border-orange-500/40'
  };

  return (
    <div
      className={clsx(
        'rounded-xl p-5 transition-all duration-200',
        variantStyles[variant],
        hoverEffect && 'hover:border-cyan-500/40 hover:translate-y-[-2px] hover:shadow-lg',
        glowStyles[glow],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
