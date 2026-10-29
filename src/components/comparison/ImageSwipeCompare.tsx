import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Layers, Eye, Calendar, Sparkles, SlidersHorizontal, RefreshCw, Flame } from 'lucide-react';
import { ApiService } from '../../services/api';

export interface ImageSwipeCompareProps {
  preImageUrl: string;
  postImageUrl: string;
  changeMaskUrl?: string;
  probabilityMapUrl?: string;
  preDateLabel?: string;
  postDateLabel?: string;
  sensorLabel?: string;
  resolutionLabel?: string;
  expansionBadgeText?: string;
  className?: string;
}

export const ImageSwipeCompare: React.FC<ImageSwipeCompareProps> = ({
  preImageUrl,
  postImageUrl,
  changeMaskUrl,
  probabilityMapUrl,
  preDateLabel = 'T1: 2018-06-15',
  postDateLabel = 'T2: 2024-06-20',
  sensorLabel = 'Sentinel-2 MSI (10m)',
  resolutionLabel = '10m GSD',
  expansionBadgeText,
  className = ''
}) => {
  const [sliderPosition, setSliderPosition] = useState<number>(50); // percentage (0 - 100)
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [showMaskOverlay, setShowMaskOverlay] = useState<boolean>(true);
  const [overlayMode, setOverlayMode] = useState<'mask' | 'prob'>('mask');
  const [maskOpacity, setMaskOpacity] = useState<number>(0.75);
  const containerRef = useRef<HTMLDivElement>(null);

  const resolveUrl = (url?: string) => {
    if (!url) return '';
    if (url.startsWith('/api/')) {
      return `${ApiService.baseUrl}${url}`;
    }
    return url;
  };

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const clampedPercentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(clampedPercentage);
  }, []);

  const handleMouseDown = () => setIsDragging(true);

  useEffect(() => {
    const handleMouseUp = () => setIsDragging(false);
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        handleMove(e.clientX);
      }
    };
    const handleTouchMove = (e: TouchEvent) => {
      if (isDragging && e.touches[0]) {
        handleMove(e.touches[0].clientX);
      }
    };

    if (isDragging) {
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('touchend', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
    }

    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('touchend', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
    };
  }, [isDragging, handleMove]);

  const activeOverlayUrl = overlayMode === 'prob' && probabilityMapUrl 
    ? resolveUrl(probabilityMapUrl) 
    : resolveUrl(changeMaskUrl);

  return (
    <div className={`flex flex-col space-y-3 ${className}`}>
      {/* Top Controls Toolbar */}
      <div className="flex items-center justify-between flex-wrap gap-2 px-4 py-2.5 rounded-xl bg-[#0d1322] border border-slate-800 text-xs">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-cyan-400 font-semibold uppercase tracking-wider text-[11px]">
            <Sparkles className="w-3.5 h-3.5" /> Bi-Temporal Swipe Compare
          </span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400 font-mono text-[11px]">{sensorLabel}</span>
          <span className="text-slate-400 font-mono text-[11px] hidden sm:inline">[{resolutionLabel}]</span>
        </div>

        <div className="flex items-center gap-4 flex-wrap">
          {/* Overlay Mode Switcher */}
          {probabilityMapUrl && (
            <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-lg border border-slate-800">
              <button
                type="button"
                onClick={() => { setOverlayMode('mask'); setShowMaskOverlay(true); }}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                  showMaskOverlay && overlayMode === 'mask' 
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40' 
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Red Change Mask
              </button>
              <button
                type="button"
                onClick={() => { setOverlayMode('prob'); setShowMaskOverlay(true); }}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors flex items-center gap-1 ${
                  showMaskOverlay && overlayMode === 'prob' 
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' 
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Flame className="w-3 h-3 text-amber-400" />
                Probability Map
              </button>
            </div>
          )}

          {/* AI Mask Overlay Toggle */}
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showMaskOverlay}
              onChange={(e) => setShowMaskOverlay(e.target.checked)}
              className="rounded bg-slate-900 border-slate-700 text-rose-500 focus:ring-0 focus:ring-offset-0"
            />
            <span className="text-slate-300 font-medium">Overlay Change</span>
          </label>

          {showMaskOverlay && (
            <div className="flex items-center gap-2">
              <span className="text-slate-500 text-[11px]">Opacity:</span>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={maskOpacity}
                onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                className="w-20 accent-rose-500 h-1 bg-slate-800 rounded cursor-pointer"
                title={`Overlay opacity: ${Math.round(maskOpacity * 100)}%`}
              />
              <span className="text-slate-400 text-[10px] font-mono w-7">
                {Math.round(maskOpacity * 100)}%
              </span>
            </div>
          )}

          <button
            onClick={() => setSliderPosition(50)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            title="Reset slider to 50% center"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Split Image Container */}
      <div
        ref={containerRef}
        className="relative w-full h-[450px] sm:h-[550px] rounded-2xl overflow-hidden select-none border border-slate-800 shadow-2xl cursor-col-resize group bg-black"
        onMouseDown={handleMouseDown}
        onTouchStart={handleMouseDown}
      >
        {/* Post-Period Image (Right / Background Layer) */}
        <div className="absolute inset-0 w-full h-full">
          <img
            src={resolveUrl(postImageUrl)}
            alt="Post-expansion satellite view"
            className="w-full h-full object-cover pointer-events-none"
          />

          {/* Real AI Sprawl Mask Overlay on Post-period */}
          {showMaskOverlay && activeOverlayUrl && (
            <img
              src={activeOverlayUrl}
              alt="Deep Learning Change Detection Overlay"
              className={`absolute inset-0 w-full h-full object-cover pointer-events-none transition-opacity duration-150 ${
                overlayMode === 'prob' ? 'mix-blend-screen' : ''
              }`}
              style={{ opacity: maskOpacity }}
            />
          )}

          {/* Badge indicator */}
          {expansionBadgeText && (
            <div className="absolute top-4 right-4 z-10 bg-rose-600/90 text-white font-bold text-xs px-3 py-1.5 rounded-full shadow-lg border border-rose-400/50 backdrop-blur-sm flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
              {expansionBadgeText}
            </div>
          )}

          {/* Post Date Label */}
          <div className="absolute bottom-4 right-4 z-10 px-3 py-1.5 rounded-lg bg-black/80 border border-slate-700/80 text-white text-xs font-mono backdrop-blur-md flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-rose-400"></span>
            <span>{postDateLabel} (Post-Observation)</span>
          </div>
        </div>

        {/* Pre-Period Image (Left / Clipped Layer) */}
        <div
          className="absolute inset-0 h-full overflow-hidden"
          style={{ width: `${sliderPosition}%` }}
        >
          <img
            src={resolveUrl(preImageUrl)}
            alt="Pre-expansion satellite view"
            className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            style={{
              width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%',
              maxWidth: 'none'
            }}
          />

          {/* Pre Date Label */}
          <div className="absolute bottom-4 left-4 z-10 px-3 py-1.5 rounded-lg bg-black/80 border border-slate-700/80 text-white text-xs font-mono backdrop-blur-md flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span>{preDateLabel} (Pre-Observation)</span>
          </div>
        </div>

        {/* Draggable Divider Line & Handle */}
        <div
          className="absolute top-0 bottom-0 z-20 w-1 bg-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.8)] -translate-x-1/2 pointer-events-none"
          style={{ left: `${sliderPosition}%` }}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 rounded-full bg-[#0d1322] border-2 border-cyan-400 text-cyan-400 flex items-center justify-center shadow-2xl group-hover:scale-110 transition-transform">
            <SlidersHorizontal className="w-4 h-4 rotate-90" />
          </div>
        </div>
      </div>

      {/* Caption & helper info */}
      <div className="flex items-center justify-between text-[11px] text-slate-500 px-1 font-mono">
        <span>Slide horizontally to inspect pixel-aligned bi-temporal surface conversion.</span>
        <span>Split: {Math.round(sliderPosition)}% Earlier / {100 - Math.round(sliderPosition)}% Later</span>
      </div>
    </div>
  );
};
