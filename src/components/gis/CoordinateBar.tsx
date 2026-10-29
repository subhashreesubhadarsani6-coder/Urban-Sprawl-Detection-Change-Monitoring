import React from 'react';
import { Compass, ZoomIn, Layers } from 'lucide-react';

export interface CoordinateBarProps {
  lat?: number;
  lng?: number;
  zoom?: number;
  projection?: string;
  resolutionMeters?: number;
}

export const CoordinateBar: React.FC<CoordinateBarProps> = ({
  lat = 36.084,
  lng = -115.25,
  zoom = 12,
  projection = 'WGS 84 (EPSG:4326)',
  resolutionMeters = 10
}) => {
  return (
    <div className="flex items-center justify-between flex-wrap gap-3 px-3 py-1.5 rounded-lg bg-[#080c14]/90 border border-slate-800/80 text-[11px] text-slate-400 font-mono backdrop-blur-md">
      <div className="flex items-center gap-4 flex-wrap">
        <span className="flex items-center gap-1.5 text-slate-300">
          <Compass className="w-3.5 h-3.5 text-cyan-400" />
          <span>LAT: {lat.toFixed(4)}°</span>
          <span>LNG: {lng.toFixed(4)}°</span>
        </span>
        <span className="text-slate-600">|</span>
        <span className="flex items-center gap-1 text-slate-400">
          <ZoomIn className="w-3 h-3 text-cyan-400" />
          <span>ZOOM: {zoom}x</span>
        </span>
        <span className="text-slate-600">|</span>
        <span>RES: {resolutionMeters}m / px</span>
      </div>

      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
        <span className="text-slate-300">{projection}</span>
      </div>
    </div>
  );
};
