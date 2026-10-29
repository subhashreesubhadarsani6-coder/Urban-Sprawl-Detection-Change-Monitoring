import React, { useState } from 'react';
import { Layers, Eye, EyeOff, Sliders, ChevronDown, ChevronUp } from 'lucide-react';
import clsx from 'clsx';

export type BaseMapLayer = 'dark' | 'satellite' | 'street';

export interface OverlayToggles {
  sprawlMask: boolean;
  vegetationNdvi: boolean;
  aoiBoundaries: boolean;
  heatMap: boolean;
}

export interface MapLayerControlsProps {
  activeBaseMap: BaseMapLayer;
  onBaseMapChange: (base: BaseMapLayer) => void;
  overlays: OverlayToggles;
  onOverlayToggle: (key: keyof OverlayToggles) => void;
  opacity: number;
  onOpacityChange: (opacity: number) => void;
}

export const MapLayerControls: React.FC<MapLayerControlsProps> = ({
  activeBaseMap,
  onBaseMapChange,
  overlays,
  onOverlayToggle,
  opacity,
  onOpacityChange
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="rounded-xl bg-[#0d1322]/95 border border-slate-700/80 shadow-2xl backdrop-blur-md overflow-hidden text-xs w-64">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3 py-2 flex items-center justify-between bg-[#0a0f1b] border-b border-slate-800 text-slate-200 font-medium"
      >
        <span className="flex items-center gap-2">
          <Layers className="w-3.5 h-3.5 text-cyan-400" />
          <span>GIS Layer Stack</span>
        </span>
        {isExpanded ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
      </button>

      {isExpanded && (
        <div className="p-3 space-y-3">
          {/* Base Map Selector */}
          <div>
            <label className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block mb-1.5">
              Base Tile Layer
            </label>
            <div className="grid grid-cols-3 gap-1">
              {(['dark', 'satellite', 'street'] as BaseMapLayer[]).map((base) => (
                <button
                  key={base}
                  onClick={() => onBaseMapChange(base)}
                  className={clsx(
                    'py-1 text-[11px] rounded capitalize font-medium transition-all',
                    activeBaseMap === base
                      ? 'bg-cyan-600 text-white font-semibold shadow'
                      : 'bg-slate-800/80 text-slate-400 hover:text-white'
                  )}
                >
                  {base}
                </button>
              ))}
            </div>
          </div>

          {/* Overlays list */}
          <div>
            <label className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block mb-1.5">
              Spectral & AI Overlays
            </label>
            <div className="space-y-1.5">
              <label className="flex items-center justify-between p-1.5 rounded hover:bg-slate-800/50 cursor-pointer">
                <span className="flex items-center gap-2 text-slate-300">
                  <span className="w-2.5 h-2.5 rounded-sm bg-orange-500 shrink-0"></span>
                  <span>AI Sprawl Mask</span>
                </span>
                <input
                  type="checkbox"
                  checked={overlays.sprawlMask}
                  onChange={() => onOverlayToggle('sprawlMask')}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 focus:ring-offset-0"
                />
              </label>

              <label className="flex items-center justify-between p-1.5 rounded hover:bg-slate-800/50 cursor-pointer">
                <span className="flex items-center gap-2 text-slate-300">
                  <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 shrink-0"></span>
                  <span>NDVI Vegetation Loss</span>
                </span>
                <input
                  type="checkbox"
                  checked={overlays.vegetationNdvi}
                  onChange={() => onOverlayToggle('vegetationNdvi')}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 focus:ring-offset-0"
                />
              </label>

              <label className="flex items-center justify-between p-1.5 rounded hover:bg-slate-800/50 cursor-pointer">
                <span className="flex items-center gap-2 text-slate-300">
                  <span className="w-2.5 h-2.5 rounded-sm bg-cyan-400 shrink-0"></span>
                  <span>AOI Boundaries</span>
                </span>
                <input
                  type="checkbox"
                  checked={overlays.aoiBoundaries}
                  onChange={() => onOverlayToggle('aoiBoundaries')}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 focus:ring-offset-0"
                />
              </label>

              <label className="flex items-center justify-between p-1.5 rounded hover:bg-slate-800/50 cursor-pointer">
                <span className="flex items-center gap-2 text-slate-300">
                  <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 shrink-0"></span>
                  <span>Sprawl Velocity Heatmap</span>
                </span>
                <input
                  type="checkbox"
                  checked={overlays.heatMap}
                  onChange={() => onOverlayToggle('heatMap')}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 focus:ring-offset-0"
                />
              </label>
            </div>
          </div>

          {/* Opacity slider */}
          <div className="pt-1 border-t border-slate-800/80">
            <div className="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Mask Opacity</span>
              <span className="font-mono text-cyan-400">{Math.round(opacity * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => onOpacityChange(parseFloat(e.target.value))}
              className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>
        </div>
      )}
    </div>
  );
};
