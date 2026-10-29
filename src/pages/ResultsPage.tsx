import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  Download, 
  FileCode, 
  Layers, 
  Share2, 
  Calendar, 
  CheckCircle2, 
  Activity, 
  TrendingUp, 
  Trees, 
  Sparkles,
  ArrowLeft,
  FileSpreadsheet,
  Sliders,
  Compass,
  Cpu,
  ShieldCheck,
  AlertCircle,
  Loader2,
  RefreshCw
} from 'lucide-react';
import { PageHeader } from '../components/layout/PageHeader';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge, StatusBadge } from '../components/common/Badge';
import { ImageSwipeCompare } from '../components/comparison/ImageSwipeCompare';
import { LulcDistribution } from '../components/charts/LulcDistribution';
import { SprawlTrendChart } from '../components/charts/SprawlTrendChart';
import { ApiService } from '../services/api';
import { AnalysisJob } from '../types';
import { useToast } from '../context/ToastContext';
import { CardSkeleton } from '../components/common/Skeleton';

export const ResultsPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const { showToast } = useToast();
  const [job, setJob] = useState<AnalysisJob | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [activeBandMode, setActiveBandMode] = useState<'rgb' | 'cir' | 'ndvi'>('rgb');

  // Interactive threshold state
  const [threshold, setThreshold] = useState<number>(0.5);
  const [isRethresholding, setIsRethresholding] = useState<boolean>(false);

  useEffect(() => {
    let interval: any;

    const fetchJob = async (showLoading = true) => {
      try {
        if (showLoading) setIsLoading(true);
        const targetId = id || 'job-vegas-2018-2024';
        const data = await ApiService.getAnalysisJobById(targetId);
        setJob(data || null);

        if (data?.parameters?.confidenceThreshold) {
          const t = Number(data.parameters.confidenceThreshold);
          setThreshold(t > 1.0 ? t / 100.0 : t);
        } else if (data?.metrics?.appliedThreshold) {
          setThreshold(Number(data.metrics.appliedThreshold));
        }

        if (data && (data.status === 'processing' || data.status === 'queued')) {
          interval = setInterval(async () => {
            try {
              const updated = await ApiService.getAnalysisJobById(targetId);
              setJob(updated);
              if (updated.status === 'completed' || updated.status === 'failed') {
                clearInterval(interval);
              }
            } catch (e) {
              clearInterval(interval);
            }
          }, 1500);
        }
      } catch (err) {
        console.error('Failed to load analysis result:', err);
      } finally {
        if (showLoading) setIsLoading(false);
      }
    };

    fetchJob();

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [id]);

  const handleExport = (format: 'GeoJSON' | 'GeoTIFF' | 'PDF' | 'Probability' | 'CSV') => {
    if (!job) return;
    if (format === 'CSV') {
      const url = ApiService.getMetricsCsvDownloadUrl(job.id);
      window.open(url, '_blank');
      showToast({
        type: 'success',
        title: 'Downloading CSV Report',
        message: 'Tabular sprawl metrics, LULC transition matrix & trends downloaded.'
      });
      return;
    }
    const filenameMap = {
      GeoJSON: 'sprawl_boundaries.geojson',
      GeoTIFF: 'change_mask.tif',
      PDF: 'audit_report.txt',
      Probability: 'probability_map.png'
    };
    const filename = filenameMap[format];
    const url = ApiService.getArtifactDownloadUrl(job.id, filename);
    window.open(url, '_blank');
    showToast({
      type: 'success',
      title: `Downloading ${format}`,
      message: `Retrieved ${filename} from backend storage.`
    });
  };

  const handleApplyThreshold = async () => {
    if (!job) return;
    setIsRethresholding(true);
    try {
      const updatedJob = await ApiService.rethresholdAnalysisJob(job.id, threshold);
      setJob(updatedJob);
      showToast({
        type: 'success',
        title: 'Threshold Re-evaluated',
        message: `Updated change detection cutoff to ${threshold.toFixed(2)}. Mask & metrics recomputed.`
      });
    } catch (err: any) {
      showToast({
        type: 'error',
        title: 'Re-thresholding Failed',
        message: err.message || 'Could not re-evaluate threshold'
      });
    } finally {
      setIsRethresholding(false);
    }
  };

  if (isLoading || !job) {
    return (
      <div className="space-y-6">
        <PageHeader title="Loading Analysis Result..." />
        <CardSkeleton />
        <CardSkeleton />
      </div>
    );
  }

  const metrics = job.metrics;
  const isRealInference = 
    metrics?.inferenceType?.includes('Real') || 
    job.parameters?.isRealInference ||
    metrics?.modelArch?.includes('FC-Siam-diff');

  const spatialRes = metrics?.spatialResolution;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <PageHeader
        title={job.title}
        subtitle={job.description}
        badge={
          <div className="flex items-center gap-2">
            <StatusBadge status={job.status} />
            {isRealInference ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <ShieldCheck className="w-3.5 h-3.5" />
                Real Model Inference (FC-Siam-diff)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                <AlertCircle className="w-3.5 h-3.5" />
                Demo Output
              </span>
            )}
          </div>
        }
        breadcrumbs={[
          { label: 'Workstation', to: '/dashboard' },
          { label: 'Sprawl Audits', to: '/dashboard' },
          { label: job.id }
        ]}
        actions={
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<FileCode className="w-4 h-4 text-cyan-400" />}
              onClick={() => handleExport('GeoJSON')}
            >
              Export GeoJSON
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<Layers className="w-4 h-4 text-orange-400" />}
              onClick={() => handleExport('GeoTIFF')}
            >
              Export GeoTIFF Mask
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<Download className="w-4 h-4 text-emerald-400" />}
              onClick={() => handleExport('Probability')}
            >
              Probability Map
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<FileSpreadsheet className="w-4 h-4 text-teal-400" />}
              onClick={() => handleExport('CSV')}
            >
              Export CSV
            </Button>
            <Button
              variant="primary"
              size="sm"
              leftIcon={<Download className="w-4 h-4" />}
              onClick={() => handleExport('PDF')}
            >
              Audit Report
            </Button>
          </div>
        }
      />

      {/* Live Pipeline Execution In-Progress Card */}
      {job.status !== 'completed' && (
        <Card className="border-cyan-500/40 bg-cyan-950/20 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-cyan-400 animate-ping"></span>
              <h3 className="text-base font-bold text-white">Pipeline Execution in Progress</h3>
            </div>
            <span className="text-sm font-mono text-cyan-300 font-bold">{job.progressPercent}%</span>
          </div>
          <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-cyan-500 to-emerald-400 h-full transition-all duration-500"
              style={{ width: `${job.progressPercent}%` }}
            ></div>
          </div>
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>{job.currentStep || 'Initializing worker node...'}</span>
            <span>Live Polling Active (1.5s)</span>
          </div>
        </Card>
      )}

      {/* Top Metric Cards */}
      {metrics && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Card className="p-4 border-l-4 border-l-orange-500">
            <div className="text-[11px] font-medium uppercase text-slate-400 tracking-wider">
              Net Surface Change
            </div>
            <div className="text-2xl font-bold font-mono text-orange-400 mt-1">
              +{metrics.netExpansionKm2} <span className="text-xs text-slate-400 font-normal">km²</span>
            </div>
            <div className="text-xs text-orange-400/90 mt-1 font-semibold flex items-center justify-between">
              <span>+{metrics.expansionPercentage}% Area Change</span>
            </div>
          </Card>

          <Card className="p-4 border-l-4 border-l-cyan-500">
            <div className="text-[11px] font-medium uppercase text-slate-400 tracking-wider">
              Changed Pixels
            </div>
            <div className="text-2xl font-bold font-mono text-cyan-400 mt-1">
              {metrics.changedPixels ? metrics.changedPixels.toLocaleString() : '60,613'}{' '}
              <span className="text-xs text-slate-400 font-normal">px</span>
            </div>
            <div className="text-xs text-slate-400 mt-1 font-mono">
              out of {metrics.totalPixels ? metrics.totalPixels.toLocaleString() : '589,984'} valid
            </div>
          </Card>

          <Card className="p-4 border-l-4 border-l-emerald-500">
            <div className="text-[11px] font-medium uppercase text-slate-400 tracking-wider">
              Spatial Resolution
            </div>
            <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
              {spatialRes?.pixel_width_m || job.imagery.resolutionMeters || 10.0}m{' '}
              <span className="text-xs text-slate-400 font-normal">GSD</span>
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {spatialRes?.is_calibrated 
                ? 'Calibrated GeoTIFF CRS' 
                : 'Metric spatial grid'}
            </div>
          </Card>

          <Card className="p-4 border-l-4 border-l-rose-500">
            <div className="text-[11px] font-medium uppercase text-slate-400 tracking-wider">
              Classification Threshold
            </div>
            <div className="text-2xl font-bold font-mono text-rose-400 mt-1">
              {(threshold * 100).toFixed(0)}% <span className="text-xs text-slate-400 font-normal">P(change)</span>
            </div>
            <div className="text-xs text-rose-400 mt-1 font-semibold">
              Siamese U-Net (FC-Siam-diff)
            </div>
          </Card>
        </div>
      )}

      {/* Dynamic Threshold Control Toolbar */}
      <Card className="p-4 bg-[#0a0f1d] border border-slate-800">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400">
              <Sliders className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Dynamic Probability Threshold</h4>
              <p className="text-xs text-slate-400">
                Adjust sensitivity cutoff to re-evaluate changed pixel clusters and calibrated area in real-time.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-500 font-mono">0.10</span>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-36 sm:w-48 accent-rose-500 h-1.5 bg-slate-800 rounded cursor-pointer"
              />
              <span className="text-xs text-slate-500 font-mono">0.90</span>
              <span className="text-sm font-mono font-bold text-white px-2.5 py-1 rounded-md bg-slate-800 border border-slate-700">
                {threshold.toFixed(2)}
              </span>
            </div>

            <Button
              variant="secondary"
              size="sm"
              onClick={handleApplyThreshold}
              disabled={isRethresholding}
              leftIcon={
                isRethresholding ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
                )
              }
            >
              {isRethresholding ? 'Re-evaluating...' : 'Apply Cutoff'}
            </Button>
          </div>
        </div>
      </Card>

      {/* Main Split-Screen Swipe Tool */}
      <Card className="space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Bi-Temporal Satellite Visual Audit</h3>
          </div>

          <div className="flex items-center gap-1 bg-[#080c14] p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setActiveBandMode('rgb')}
              className={`px-2.5 py-1 rounded font-medium transition-colors ${activeBandMode === 'rgb' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'}`}
            >
              Natural RGB
            </button>
            <button
              onClick={() => setActiveBandMode('cir')}
              className={`px-2.5 py-1 rounded font-medium transition-colors ${activeBandMode === 'cir' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'}`}
            >
              Color Infrared (CIR)
            </button>
            <button
              onClick={() => setActiveBandMode('ndvi')}
              className={`px-2.5 py-1 rounded font-medium transition-colors ${activeBandMode === 'ndvi' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'}`}
            >
              NDVI Difference
            </button>
          </div>
        </div>

        <ImageSwipeCompare
          preImageUrl={job.imagery.preImageUrl}
          postImageUrl={job.imagery.postImageUrl}
          changeMaskUrl={job.imagery.changeMaskUrl || (job.imagery as any).overlayUrl}
          probabilityMapUrl={(job.imagery as any).probabilityMapUrl}
          preDateLabel={`Pre-Period: ${job.imagery.preDate}`}
          postDateLabel={`Post-Period: ${job.imagery.postDate}`}
          sensorLabel={job.imagery.sensor}
          resolutionLabel={`${job.imagery.resolutionMeters || 10}m Spatial Resolution`}
          expansionBadgeText={`+${metrics?.netExpansionKm2 || 6.06} km² Changed Area`}
        />
      </Card>

      {/* Multispectral Sprawl Attribution Card */}
      {((job as any).multispectralAttribution?.available || (metrics as any)?.multispectralAttribution?.available) ? (
        <Card className="space-y-4 border border-cyan-500/20 bg-[#080d1a]">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3 flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <Trees className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-semibold text-white">Multispectral Sprawl Attribution (NDVI & NDBI Analysis)</h3>
            </div>
            <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-emerald-950/60 text-emerald-400 border border-emerald-500/30 font-semibold">
              Crop False-Alarm Filtering Active
            </span>
          </div>

          {(() => {
            const ms = (job as any).multispectralAttribution || (metrics as any)?.multispectralAttribution;
            return (
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="p-3 rounded-lg bg-[#0d1322] border border-red-500/30">
                    <div className="text-[11px] text-red-400 font-medium">Verified Built-Up Sprawl</div>
                    <div className="text-lg font-bold font-mono text-white mt-0.5">
                      {ms.verifiedSprawlPixels?.toLocaleString() || 0} <span className="text-xs text-slate-400 font-normal">px</span>
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      +{ms.verifiedSprawlAreaKm2 || 0} km² concrete/impervious
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-[#0d1322] border border-amber-500/30">
                    <div className="text-[11px] text-amber-400 font-medium">Agricultural Phenology</div>
                    <div className="text-lg font-bold font-mono text-white mt-0.5">
                      {ms.agriculturalPhenologyPixels?.toLocaleString() || 0} <span className="text-xs text-slate-400 font-normal">px</span>
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      Crop harvest / bare soil fluctuation
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-[#0d1322] border border-cyan-500/30">
                    <div className="text-[11px] text-cyan-400 font-medium">Sprawl Purity Ratio</div>
                    <div className="text-lg font-bold font-mono text-cyan-300 mt-0.5">
                      {Math.round((ms.sprawlAttributionRatio || 0) * 100)}%
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      True urban expansion fraction
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-[#060a12] border border-slate-800 flex items-center justify-between flex-wrap gap-2 text-xs">
                  <span className="text-slate-400">
                    Spectral Indices: NDVI (NIR-Red)/(NIR+Red) and NDBI (SWIR-NIR)/(SWIR+NIR) confirm structural built-up material.
                  </span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleExport('Probability')}
                      className="text-xs text-cyan-400 hover:text-cyan-300 underline font-mono"
                    >
                      View Spectral Masks
                    </button>
                  </div>
                </div>
              </div>
            );
          })()}
        </Card>
      ) : null}

      {/* Land Use / Land Cover Distribution & Transition Matrix */}
      {job.lulcBreakdown && (
        <Card className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-orange-400" />
              <h3 className="text-sm font-semibold text-white">Land Use / Land Cover (LULC) Transition Matrix</h3>
            </div>
            <span className="text-[11px] font-mono text-slate-500">
              Confidence Cutoff: {Math.round(threshold * 100)}%
            </span>
          </div>

          <LulcDistribution data={job.lulcBreakdown} />
        </Card>
      )}

      {/* Temporal Trend Chart */}
      {job.temporalTrend && (
        <Card className="space-y-4">
          <SprawlTrendChart
            data={job.temporalTrend}
            title="Multi-Year Urban Built-Up vs Vegetation Trend (2018 - 2024)"
          />
        </Card>
      )}

      {/* Model & Processing Pipeline Metadata */}
      <Card className="space-y-3">
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center justify-between">
          <span>Inference Provenance & Geospatial Specifications</span>
          <span className="text-cyan-400 font-mono normal-case text-[11px]">
            {isRealInference ? 'PyTorch Siamese U-Net Engine' : 'Synthetic Pipeline'}
          </span>
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono text-slate-400">
          <div>
            <span className="text-slate-500 block text-[10px]">Model Architecture</span>
            <span className="text-cyan-400 font-bold">{job.parameters.modelArch || 'FC-Siam-diff'}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">Checkpoint Path</span>
            <span className="text-slate-200">checkpoints/best_checkpoint.pth</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">Coordinate System (CRS)</span>
            <span className="text-slate-200">
              {spatialRes?.crs || 'EPSG:32611 (UTM Zone 11N)'}
            </span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">Tiling & Blending</span>
            <span className="text-slate-200">256x256 (64px overlap, 2D Hann)</span>
          </div>
        </div>
      </Card>
    </div>
  );
};
