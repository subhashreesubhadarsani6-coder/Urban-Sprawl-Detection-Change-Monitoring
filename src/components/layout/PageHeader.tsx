import React from 'react';
import { ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';

export interface BreadcrumbItem {
  label: string;
  to?: string;
}

export interface PageHeaderProps {
  title: string;
  subtitle?: string;
  breadcrumbs?: BreadcrumbItem[];
  actions?: React.ReactNode;
  badge?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  subtitle,
  breadcrumbs,
  actions,
  badge
}) => {
  return (
    <div className="mb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-800/60 pb-5">
      <div>
        {/* Breadcrumbs */}
        {breadcrumbs && breadcrumbs.length > 0 && (
          <nav className="flex items-center gap-1.5 text-xs text-slate-500 mb-1.5">
            {breadcrumbs.map((crumb, idx) => (
              <React.Fragment key={idx}>
                {idx > 0 && <ChevronRight className="w-3.5 h-3.5 text-slate-600" />}
                {crumb.to ? (
                  <Link to={crumb.to} className="hover:text-cyan-400 transition-colors">
                    {crumb.label}
                  </Link>
                ) : (
                  <span className="text-slate-400 font-medium">{crumb.label}</span>
                )}
              </React.Fragment>
            ))}
          </nav>
        )}

        {/* Title & Badge */}
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white">{title}</h1>
          {badge}
        </div>

        {/* Subtitle */}
        {subtitle && (
          <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl leading-relaxed">{subtitle}</p>
        )}
      </div>

      {/* Actions */}
      {actions && (
        <div className="flex items-center gap-2.5 shrink-0 flex-wrap">
          {actions}
        </div>
      )}
    </div>
  );
};
