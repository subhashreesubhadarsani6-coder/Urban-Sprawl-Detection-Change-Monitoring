import React, { useEffect, useState } from 'react';
import { 
  Award, 
  BarChart3, 
  CheckCircle2, 
  AlertTriangle, 
  HelpCircle, 
  Sliders, 
  Layers, 
  ExternalLink,
  Zap,
  TrendingUp,
  Cpu
} from 'lucide-react';
import { PageHeader } from '../components/layout/PageHeader';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { ConfusionMatrix } from '../components/charts/ConfusionMatrix';
import { ApiService } from '../services/api';
import { ModelBenchmark, ConfusionMatrixClass } from '../types';

export const EvaluationPage: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<ModelBenchmark[]>([]);
  const [matrixData, setMatrixData] = useState<ConfusionMatrixClass[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const [bmList, matrix] = await Promise.all([
          ApiService.getBenchmarks(),
          ApiService.getConfusionMatrix()
        ]);
        setBenchmarks(bmList);
        setMatrixData(matrix);
      } catch (err) {
        console.error('Error fetching evaluation data:', err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Evaluation & Benchmark Leaderboard"
        subtitle="Cross-architectural comparison and pixel confusion analysis on standard remote sensing urban sprawl test sets."
        breadcrumbs={[
          { label: 'Workstation', to: '/dashboard' },
          { label: 'Evaluation' }
        ]}
      />

      {/* Top Benchmark Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-[11px] font-medium uppercase text-slate-400">Production Champion</div>
          <div className="text-xl font-bold font-mono text-cyan-400 mt-1">ChangeFormer</div>
          <div className="text-xs text-slate-400 mt-1">Hierarchical Transformer</div>
        </Card>

        <Card className="p-4">
          <div className="text-[11px] font-medium uppercase text-slate-400">Peak mIoU Accuracy</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">86.8%</div>
          <div className="text-xs text-emerald-400 mt-1">Test Split IoU</div>
        </Card>

        <Card className="p-4">
          <div className="text-[11px] font-medium uppercase text-slate-400">Lowest Latency</div>
          <div className="text-2xl font-bold font-mono text-white mt-1">14 ms</div>
          <div className="text-xs text-slate-400 mt-1">FC-Siam-diff (Baseline)</div>
        </Card>

        <Card className="p-4">
          <div className="text-[11px] font-medium uppercase text-slate-400">False Positive Rate</div>
          <div className="text-2xl font-bold font-mono text-indigo-400 mt-1">2.1%</div>
          <div className="text-xs text-slate-400 mt-1">Controlled with SWIR bands</div>
        </Card>
      </div>

      {/* Architecture Benchmark Leaderboard Table */}
      <Card className="p-0 overflow-hidden border-slate-800">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Multi-Temporal Change Detection Architectures</h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">Benchmark: SpaceNet 7 & S2-USG Test Split</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#0a0e1a] text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th className="px-5 py-3.5">Model Architecture</th>
                <th className="px-4 py-3.5">Backbone</th>
                <th className="px-4 py-3.5">Parameters</th>
                <th className="px-4 py-3.5">Inference (ms)</th>
                <th className="px-4 py-3.5">mIoU (%)</th>
                <th className="px-4 py-3.5">F1-Score (%)</th>
                <th className="px-4 py-3.5">Precision</th>
                <th className="px-4 py-3.5">Recall</th>
                <th className="px-4 py-3.5">VRAM</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300 font-mono text-[11px]">
              {benchmarks.map((bm, idx) => (
                <tr
                  key={bm.id}
                  className={`hover:bg-slate-850/40 transition-colors ${idx === 0 ? 'bg-cyan-950/20' : ''}`}
                >
                  <td className="px-5 py-3.5 font-sans font-semibold text-white flex items-center gap-2">
                    {idx === 0 && <span className="w-2 h-2 rounded-full bg-cyan-400"></span>}
                    <span>{bm.modelName}</span>
                    {idx === 0 && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-400 border border-cyan-500/30 font-mono uppercase">
                        Current SOTA
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3.5 text-slate-400 font-sans">{bm.backbone}</td>
                  <td className="px-4 py-3.5">{bm.parametersM}M</td>
                  <td className="px-4 py-3.5 text-emerald-400">{bm.inferenceSpeedMs} ms</td>
                  <td className="px-4 py-3.5 text-cyan-300 font-bold">{bm.mIoU}%</td>
                  <td className="px-4 py-3.5 text-white font-bold">{bm.f1Score}%</td>
                  <td className="px-4 py-3.5">{bm.precision}%</td>
                  <td className="px-4 py-3.5">{bm.recall}%</td>
                  <td className="px-4 py-3.5 text-slate-400">{bm.memoryFootprintMb} MB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Two-Column Section: Confusion Matrix + Error Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix Heatmap */}
        <Card className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-white">Pixel Confusion Matrix (ChangeFormer)</h3>
            </div>
            <span className="text-[11px] font-mono text-slate-500">4 Core Land Classes</span>
          </div>

          <ConfusionMatrix data={matrixData} />
        </Card>

        {/* Error Analysis & False Positive Inspection */}
        <Card className="space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <AlertTriangle className="w-4 h-4 text-orange-400" />
            <h3 className="text-sm font-semibold text-white">Remote Sensing Error Analysis & Mitigation</h3>
          </div>

          <div className="space-y-3 text-xs leading-relaxed">
            <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800/80 space-y-1">
              <div className="font-semibold text-orange-400 flex items-center justify-between">
                <span>1. Bare Desert Soil vs. Flat Concrete Roofs</span>
                <span className="text-[10px] font-mono text-slate-500">4.8% Confusion</span>
              </div>
              <p className="text-slate-400">
                In arid climates (e.g., Las Vegas, Dubai), sun-baked soil shares optical reflectance with unpainted industrial concrete roofs. Mitigated by integrating Sentinel-2 SWIR-1 (B11) and SWIR-2 (B12) bands to compute the NDBI index.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800/80 space-y-1">
              <div className="font-semibold text-amber-400 flex items-center justify-between">
                <span>2. Seasonal Cropland Fallow vs. Built-up</span>
                <span className="text-[10px] font-mono text-slate-500">3.2% Confusion</span>
              </div>
              <p className="text-slate-400">
                Plowed dry agricultural fields mimic ground clearing for subdivision construction. Handled by multi-date temporal stacking over a 12-month baseline rather than a single bi-temporal snapshot.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800/80 space-y-1">
              <div className="font-semibold text-cyan-400 flex items-center justify-between">
                <span>3. Water Reservoirs & Shadow Encroachment</span>
                <span className="text-[10px] font-mono text-slate-500">0.7% Confusion</span>
              </div>
              <p className="text-slate-400">
                Mountain cloud shadows occasionally mimic artificial lakes. Successfully filtered out with automated MNDWI thresholds (&gt; 0.3) combined with solar elevation vector normalization.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
