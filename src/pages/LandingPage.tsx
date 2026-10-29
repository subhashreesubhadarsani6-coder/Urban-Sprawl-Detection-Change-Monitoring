import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Satellite, 
  Layers, 
  Activity, 
  ArrowRight, 
  ShieldCheck, 
  Cpu, 
  Sparkles, 
  BarChart3, 
  SplitSquareVertical, 
  Globe2, 
  CheckCircle,
  Clock,
  Compass
} from 'lucide-react';
import { Button } from '../components/common/Button';

export const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 selection:bg-cyan-500/30">
      {/* Top Navigation */}
      <nav className="h-16 border-b border-slate-800/80 px-6 sm:px-12 flex items-center justify-between sticky top-0 z-40 bg-[#080c14]/90 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-700 flex items-center justify-center text-white shadow-lg shadow-cyan-900/30">
            <Satellite className="w-5 h-5 text-cyan-100" />
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-base tracking-wider text-white flex items-center gap-2">
              URBANSCOPE <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-500/30">AI</span>
            </span>
            <span className="text-[10px] text-slate-400 tracking-tight">Spatio-Temporal GIS Intelligence</span>
          </div>
        </div>

        <div className="hidden md:flex items-center gap-8 text-xs font-medium text-slate-300">
          <a href="#features" className="hover:text-cyan-400 transition-colors">Features</a>
          <a href="#pipeline" className="hover:text-cyan-400 transition-colors">Satellite Pipeline</a>
          <a href="#metrics" className="hover:text-cyan-400 transition-colors">Sprawl Indices</a>
          <Link to="/docs" className="hover:text-cyan-400 transition-colors">Documentation</Link>
        </div>

        <div className="flex items-center gap-3">
          <Link to="/dashboard">
            <Button variant="primary" size="sm" rightIcon={<ArrowRight className="w-4 h-4" />}>
              Launch GIS Workstation
            </Button>
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-20 pb-28 px-6 sm:px-12 lg:px-20 overflow-hidden">
        {/* Radar background decorative circle */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] rounded-full border border-cyan-500/10 pointer-events-none">
          <div className="w-full h-full rounded-full radar-sweep opacity-30"></div>
        </div>

        <div className="max-w-5xl mx-auto text-center relative z-10 space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-xs font-mono text-cyan-400">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Bi-Temporal Transformer & Siamese Change Detection</span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-tight">
            Next-Generation Satellite <br />
            <span className="bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 bg-clip-text text-transparent">
              Urban Sprawl & Land Change Intelligence
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-400 max-w-3xl mx-auto leading-relaxed">
            Automate peri-urban expansion monitoring, impervious surface tracking, and deforestation audits using deep learning models trained on Sentinel-2, Landsat-8, and commercial satellite imagery.
          </p>

          <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link to="/dashboard">
              <Button variant="primary" size="lg" className="w-full sm:w-auto text-sm px-6" rightIcon={<ArrowRight className="w-4 h-4" />}>
                Open Interactive Dashboard
              </Button>
            </Link>
            <Link to="/analysis/results/job-vegas-2018-2024">
              <Button variant="secondary" size="lg" className="w-full sm:w-auto text-sm px-6" leftIcon={<SplitSquareVertical className="w-4 h-4 text-cyan-400" />}>
                Explore Las Vegas Sprawl Audit
              </Button>
            </Link>
          </div>

          {/* Key capability summary stats */}
          <div className="pt-12 grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-4xl mx-auto">
            <div className="p-4 rounded-xl bg-[#0d1322]/80 border border-slate-800 text-left">
              <div className="text-2xl font-bold font-mono text-cyan-400">92.4%</div>
              <div className="text-xs text-slate-400 mt-0.5">ChangeFormer F1-Score</div>
            </div>
            <div className="p-4 rounded-xl bg-[#0d1322]/80 border border-slate-800 text-left">
              <div className="text-2xl font-bold font-mono text-emerald-400">10m - 0.5m</div>
              <div className="text-xs text-slate-400 mt-0.5">Multi-Sensor GSD</div>
            </div>
            <div className="p-4 rounded-xl bg-[#0d1322]/80 border border-slate-800 text-left">
              <div className="text-2xl font-bold font-mono text-orange-400">4 Classes</div>
              <div className="text-xs text-slate-400 mt-0.5">LULC Land Partition</div>
            </div>
            <div className="p-4 rounded-xl bg-[#0d1322]/80 border border-slate-800 text-left">
              <div className="text-2xl font-bold font-mono text-indigo-400">&lt; 45 ms</div>
              <div className="text-xs text-slate-400 mt-0.5">Tile Inference Latency</div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Showcase Grid */}
      <section id="features" className="py-20 px-6 sm:px-12 lg:px-20 bg-[#060910] border-t border-slate-800/80">
        <div className="max-w-6xl mx-auto space-y-12">
          <div className="text-center space-y-2">
            <span className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-semibold">
              Comprehensive Capabilities
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-white">
              End-to-End Remote Sensing Sprawl Analytics
            </h2>
            <p className="text-sm text-slate-400 max-w-2xl mx-auto">
              From raw multi-spectral satellite ingestion to sub-pixel change detection and automated compliance reporting.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-2xl bg-[#0d1322] border border-slate-800 hover:border-cyan-500/40 transition-all space-y-4">
              <div className="w-12 h-12 rounded-xl bg-cyan-950/60 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                <SplitSquareVertical className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold text-white">Bi-Temporal Swipe Comparison</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Interactively drag and swipe between historical satellite observations and contemporary passes to visually inspect built-up encroachment.
              </p>
              <div className="pt-2">
                <Link to="/analysis/results/job-vegas-2018-2024" className="text-xs text-cyan-400 hover:underline flex items-center gap-1 font-medium">
                  Try Swipe Tool &rarr;
                </Link>
              </div>
            </div>

            <div className="p-6 rounded-2xl bg-[#0d1322] border border-slate-800 hover:border-orange-500/40 transition-all space-y-4">
              <div className="w-12 h-12 rounded-xl bg-orange-950/60 border border-orange-500/30 flex items-center justify-center text-orange-400">
                <Layers className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold text-white">Multi-Spectral Indexing</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Compute normalized difference vegetation (NDVI), built-up (NDBI), and water (MNDWI) indices to isolate urban structures from bare soil.
              </p>
              <div className="pt-2">
                <Link to="/docs" className="text-xs text-orange-400 hover:underline flex items-center gap-1 font-medium">
                  Review Spectral Formulas &rarr;
                </Link>
              </div>
            </div>

            <div className="p-6 rounded-2xl bg-[#0d1322] border border-slate-800 hover:border-emerald-500/40 transition-all space-y-4">
              <div className="w-12 h-12 rounded-xl bg-emerald-950/60 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold text-white">Neural Network Training</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Fine-tune state-of-the-art architectures (ChangeFormer, Siam-UNet, BIT-CD) with custom hyperparameters, learning rates, and loss functions.
              </p>
              <div className="pt-2">
                <Link to="/training" className="text-xs text-emerald-400 hover:underline flex items-center gap-1 font-medium">
                  Launch Training Console &rarr;
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Satellite Ingestion Architecture */}
      <section id="pipeline" className="py-20 px-6 sm:px-12 lg:px-20 border-t border-slate-800/80">
        <div className="max-w-5xl mx-auto space-y-12">
          <div className="text-center space-y-2">
            <span className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-semibold">
              Pipeline Architecture
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-white">
              How UrbanScope AI Detects Urban Expansion
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-5 rounded-xl bg-[#0d1322] border border-slate-800 relative space-y-3">
              <span className="font-mono text-cyan-400 text-xs font-bold">01 / INGESTION</span>
              <h4 className="text-sm font-semibold text-white">Satellite Pair Pairing</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Retrieve ortho-rectified surface reflectance tiles (T1 and T2) from Copernicus or USGS archives.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-[#0d1322] border border-slate-800 relative space-y-3">
              <span className="font-mono text-cyan-400 text-xs font-bold">02 / CORRECTION</span>
              <h4 className="text-sm font-semibold text-white">Atmospheric Calibration</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Cloud shadow masking, seasonal solar angle normalization, and band registration alignment.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-[#0d1322] border border-slate-800 relative space-y-3">
              <span className="font-mono text-cyan-400 text-xs font-bold">03 / INFERENCE</span>
              <h4 className="text-sm font-semibold text-white">Transformer Attention</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Bi-temporal feature difference modeling with hierarchical self-attention to generate pixel change masks.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-[#0d1322] border border-slate-800 relative space-y-3">
              <span className="font-mono text-cyan-400 text-xs font-bold">04 / METRICS</span>
              <h4 className="text-sm font-semibold text-white">LULC Sprawl Vectorization</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Convert raster masks into GeoJSON boundary polygons, net expansion km², and vegetation depletion indices.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 px-6 sm:px-12 border-t border-slate-800 text-xs text-slate-500 bg-[#05080f]">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Satellite className="w-4 h-4 text-cyan-400" />
            <span className="font-semibold text-slate-400">UrbanScope AI Spatio-Temporal Platform</span>
          </div>
          <div>
            Built with React, TypeScript, Tailwind CSS, Leaflet & Recharts.
          </div>
          <div className="flex gap-4">
            <Link to="/dashboard" className="hover:text-slate-300">Dashboard</Link>
            <Link to="/analysis/new" className="hover:text-slate-300">New Analysis</Link>
            <Link to="/docs" className="hover:text-slate-300">API Documentation</Link>
          </div>
        </div>
      </footer>
    </div>
  );
};
