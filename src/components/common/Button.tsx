import React, { ButtonHTMLAttributes } from 'react';
import { Loader2 } from 'lucide-react';
import clsx from 'clsx';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'outline' | 'ghost' | 'emerald';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  className,
  disabled,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-medium transition-all duration-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed select-none active:scale-[0.98]';

  const sizeStyles = {
    sm: 'px-3 py-1.5 text-xs gap-1.5',
    md: 'px-4 py-2 text-sm gap-2',
    lg: 'px-5 py-2.5 text-base gap-2.5'
  };

  const variantStyles = {
    primary: 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-950/50 focus:ring-cyan-400 border border-cyan-500/30',
    secondary: 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700/80 focus:ring-slate-500',
    danger: 'bg-rose-600/90 hover:bg-rose-600 text-white shadow-lg shadow-rose-950/40 focus:ring-rose-500 border border-rose-500/30',
    outline: 'border border-cyan-500/40 text-cyan-400 hover:bg-cyan-950/30 hover:border-cyan-400 focus:ring-cyan-500',
    ghost: 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 focus:ring-slate-600',
    emerald: 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-950/50 focus:ring-emerald-400 border border-emerald-500/30'
  };

  return (
    <button
      className={clsx(baseStyles, sizeStyles[size], variantStyles[variant], className)}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <Loader2 className="w-4 h-4 animate-spin shrink-0" />
      ) : (
        leftIcon && <span className="shrink-0">{leftIcon}</span>
      )}
      <span>{children}</span>
      {!isLoading && rightIcon && <span className="shrink-0">{rightIcon}</span>}
    </button>
  );
};
