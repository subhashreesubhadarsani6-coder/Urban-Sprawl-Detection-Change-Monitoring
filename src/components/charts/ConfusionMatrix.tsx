import React from 'react';
import { ConfusionMatrixClass } from '../../types';

export interface ConfusionMatrixProps {
  data: ConfusionMatrixClass[];
}

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ data }) => {
  const classes = ['Urban Built-Up', 'Barren Soil', 'Vegetation', 'Water'];

  // Lookup value helper
  const getValue = (actual: string, predicted: string) => {
    const item = data.find(d => d.actual === actual && d.predicted === predicted);
    return item ? item.value : 0;
  };

  const getCellColor = (actual: string, predicted: string, val: number) => {
    const isDiagonal = actual === predicted;
    if (isDiagonal) {
      // True positives (cyan gradient based on accuracy)
      if (val >= 95) return 'bg-cyan-600/80 text-white font-bold';
      if (val >= 90) return 'bg-cyan-700/70 text-cyan-100 font-bold';
      return 'bg-cyan-800/60 text-cyan-200';
    }
    // Misclassifications
    if (val > 4) return 'bg-rose-950/70 text-rose-300 font-medium';
    if (val > 1) return 'bg-amber-950/40 text-amber-300';
    return 'bg-slate-900/60 text-slate-500';
  };

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[500px]">
        {/* Top Header: Predicted Classes */}
        <div className="text-center font-semibold text-xs text-slate-400 uppercase tracking-wider mb-2">
          Predicted Class &rarr;
        </div>

        <div className="flex">
          {/* Left Y-axis label */}
          <div className="w-8 flex items-center justify-center">
            <span className="-rotate-90 text-xs font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap">
              &larr; Ground Truth
            </span>
          </div>

          {/* Matrix Grid */}
          <div className="flex-1">
            <div className="grid grid-cols-5 gap-1.5">
              {/* Top-left corner */}
              <div className="p-2 text-[10px] font-mono text-slate-500 flex items-center justify-center">
                Actual \ Pred
              </div>
              {classes.map(c => (
                <div key={c} className="p-2 text-center text-[11px] font-medium text-slate-300 truncate" title={c}>
                  {c.split(' ')[0]}
                </div>
              ))}

              {/* Rows */}
              {classes.map(actual => (
                <React.Fragment key={actual}>
                  {/* Row header */}
                  <div className="p-2 text-right text-[11px] font-medium text-slate-300 truncate pr-2 flex items-center justify-end" title={actual}>
                    {actual.split(' ')[0]}
                  </div>

                  {/* Row cells */}
                  {classes.map(predicted => {
                    const val = getValue(actual, predicted);
                    const cellClass = getCellColor(actual, predicted, val);
                    return (
                      <div
                        key={`${actual}-${predicted}`}
                        className={`p-3 rounded-lg flex flex-col items-center justify-center border border-slate-800/80 transition-all hover:scale-[1.03] ${cellClass}`}
                      >
                        <span className="text-sm font-mono">{val.toFixed(1)}%</span>
                        {actual === predicted && (
                          <span className="text-[9px] uppercase tracking-wider opacity-80">Recall</span>
                        )}
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-end gap-5 mt-4 text-[11px] text-slate-400 font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-cyan-700"></span>
            <span>Diagonal (Correct Detections)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-rose-950/70 border border-rose-800"></span>
            <span>Error Leakage (&gt;4%)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
