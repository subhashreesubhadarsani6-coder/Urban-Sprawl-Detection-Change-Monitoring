import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { 
  Activity, 
  TrendingUp, 
  Trees, 
  Building2, 
  AlertTriangle, 
  ArrowUpRight, 
  UploadCloud, 
  Filter, 
  MapPin, 
  ChevronRight,
  ExternalLink,
  Layers
} from 'lucide-react';
import { PageHeader } from '../components/layout/PageHeader';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { StatusBadge, Badge } from '../components/common/Badge';
import { GisMap } from '../components/gis/GisMap';
import { SprawlTrendChart } from '../components/charts/SprawlTrendChart';
import { ApiService } from '../services/api';
import { AreaOfInterest, AnalysisJob } from '../types';
import { CardSkeleton, TableSkeleton } from '../components/common/Skeleton';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const [aois, setAois] = useState<AreaOfInterest[]>([]);
  const [jobs, setJobs] = useState<AnalysisJob[]>([]);
  const [selectedAoi, setSelectedAoi] = useState<AreaOfInterest | undefined>(undefined);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const [aoiList, jobList] = await Promise.all([
          ApiService.getAois(),
          ApiService.getAnalysisJobs()
        ]);
        setAois(aoiList);
        setJobs(jobList);
        if (aoiList.length > 0) {
          setSelectedAoi(aoiList[0]);
        }
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  const totalMonitoredArea = aois.reduce((acc, a) => acc + (a.totalAreaKm2 || 0), 0);
  const avgSprawlRate = aois.length > 0 ? (aois.reduce((acc, a) => acc + (a.urbanGrowthRateAnnual || 0), 0) / aois.length) : 0;
  const totalVegetationLoss = jobs.reduce((acc, j) => acc + (j.metrics?.vegetationLossKm2 || 0), 0);
  const totalNetExpansion = jobs.reduce((acc, j) => acc + (j.metrics?.netExpansionKm2 || 0), 0);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <PageHeader
        title="GIS Sprawl Monitoring Dashboard"
        subtitle="Global real-time satellite observation across monitored metropolitan corridors."
        breadcrumbs={[
          { label: 'Workstation', to: '/dashboard' },
          { label: 'Dashboard' }
        ]}
        actions={
          <div className="flex items-center gap-2.5">
            <Link to="/analysis/new">
              <Button variant="primary" size="sm" leftIcon={<UploadCloud className="w-4 h-4" />}>
                New Sprawl Audit
              </Button>
            </Link>
          </div>
        }
      />

      {/* Top Aggregate KPI Cards */}
      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <CardSkeleton />
          <CardSkeleton />
          <CardSkeleton />
          <CardSkeleton />
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs font-medium uppercase tracking-wider">Monitored Footprint</span>
              <span className="p-2 rounded-lg bg-cyan-950/50 text-cyan-400">
                <Building2 className="w-4 h-4" />
              </span>
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-white">
                {totalMonitoredArea.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} <span className="text-sm font-normal text-slate-400">km²</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-slate-400 mt-1">
                <span className="text-cyan-400 font-mono font-medium">{aois.length} Active AOIs</span>
                <span>in SQLite database</span>
              </div>
            </div>
          </Card>

          <Card className="flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs font-medium uppercase tracking-wider">Sprawl Velocity</span>
              <span className="p-2 rounded-lg bg-orange-950/50 text-orange-400">
                <TrendingUp className="w-4 h-4" />
              </span>
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-orange-400">
                +{avgSprawlRate.toFixed(1)}% <span className="text-sm font-normal text-slate-400">/ yr</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-slate-400 mt-1">
                <Badge variant={avgSprawlRate > 5 ? "warning" : "emerald"} size="sm">
                  {avgSprawlRate > 5 ? "High Risk" : "Moderate"}
                </Badge>
                <span>Average growth rate</span>
              </div>
            </div>
          </Card>

          <Card className="flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs font-medium uppercase tracking-wider">Vegetation Canopy Loss</span>
              <span className="p-2 rounded-lg bg-emerald-950/50 text-emerald-400">
                <Trees className="w-4 h-4" />
              </span>
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-rose-400">
                -{totalVegetationLoss.toFixed(1)} <span className="text-sm font-normal text-slate-400">km²</span>
              </div>
              <div className="flex items-center gap-1 text-xs text-rose-400 mt-1">
                <ArrowUpRight className="w-3.5 h-3.5 rotate-45" />
                <span>Audited Canopy Depletion</span>
              </div>
            </div>
          </Card>

          <Card className="flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs font-medium uppercase tracking-wider">Impervious Surface</span>
              <span className="p-2 rounded-lg bg-indigo-950/50 text-indigo-400">
                <Activity className="w-4 h-4" />
              </span>
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-white">
                +{totalNetExpansion.toFixed(1)} <span className="text-sm font-normal text-slate-400">km²</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-slate-400 mt-1">
                <span className="text-indigo-400 font-mono font-medium">Net Built Expansion</span>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Main Interactive Map Section */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <h3 className="text-base font-bold text-white tracking-wide">Interactive Geospatial Canvas</h3>
            <span className="text-xs text-slate-400 font-mono hidden sm:inline">
              (Click any AOI bounding box or marker to inspect)
            </span>
          </div>

          {selectedAoi && (
            <div className="text-xs text-cyan-400 font-medium flex items-center gap-1 bg-cyan-950/40 px-2.5 py-1 rounded-lg border border-cyan-500/20">
              <MapPin className="w-3.5 h-3.5" />
              <span>Selected: {selectedAoi.name}</span>
            </div>
          )}
        </div>

        <GisMap
          aois={aois}
          selectedAoi={selectedAoi}
          onSelectAoi={(aoi) => setSelectedAoi(aoi)}
          height="520px"
        />
      </div>

      {/* Multi-column layout: Temporal Trends + Monitored AOI quick cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trend Chart (2 columns) */}
        <div className="lg:col-span-2">
          <Card className="h-full flex flex-col justify-between">
            <SprawlTrendChart
              data={
                jobs[0]?.temporalTrend || [
                  { year: 2018, urbanAreaKm2: 184.2, vegetationAreaKm2: 42.6 },
                  { year: 2020, urbanAreaKm2: 204.8, vegetationAreaKm2: 37.9 },
                  { year: 2022, urbanAreaKm2: 228.0, vegetationAreaKm2: 32.8 },
                  { year: 2024, urbanAreaKm2: 247.6, vegetationAreaKm2: 29.8 }
                ]
              }
              title={`Spatio-Temporal Trends: ${selectedAoi ? selectedAoi.city : 'Las Vegas Valley'}`}
            />
          </Card>
        </div>

        {/* Monitored Regions Quick List (1 column) */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-semibold uppercase tracking-wider text-slate-300">Monitored Regions</span>
            <span>{aois.length} AOIs active</span>
          </div>

          <div className="space-y-2.5">
            {aois.map(aoi => {
              const isSelected = selectedAoi?.id === aoi.id;
              return (
                <div
                  key={aoi.id}
                  onClick={() => setSelectedAoi(aoi)}
                  className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500/50 shadow-md'
                      : 'bg-[#0d1322] border-slate-800/80 hover:border-slate-700 hover:bg-[#111827]'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <img
                      src={aoi.thumbnailUrl}
                      alt={aoi.name}
                      className="w-10 h-10 rounded-lg object-cover border border-slate-700 shrink-0"
                    />
                    <div className="min-w-0">
                      <h4 className="text-xs font-semibold text-white truncate">{aoi.name}</h4>
                      <p className="text-[11px] text-slate-400">{aoi.city}, {aoi.country}</p>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-xs font-mono font-bold text-orange-400">+{aoi.urbanGrowthRateAnnual}%/yr</div>
                    <Badge
                      variant={aoi.riskLevel === 'Severe' ? 'failed' : aoi.riskLevel === 'High' ? 'warning' : 'emerald'}
                      size="sm"
                    >
                      {aoi.riskLevel}
                    </Badge>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Recent Analysis Pipeline Runs Table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <h3 className="text-base font-bold text-white tracking-wide">Recent Change Detection Runs</h3>
          </div>
          <Link to="/analysis/new" className="text-xs text-cyan-400 hover:underline">
            Launch New Job &rarr;
          </Link>
        </div>

        <Card className="p-0 overflow-hidden border-slate-800">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#0a0e1a] text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="px-5 py-3.5">Analysis Job</th>
                  <th className="px-4 py-3.5">Model Architecture</th>
                  <th className="px-4 py-3.5">Sensor / GSD</th>
                  <th className="px-4 py-3.5">Date Range</th>
                  <th className="px-4 py-3.5">Status</th>
                  <th className="px-5 py-3.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {jobs.map(job => (
                  <tr key={job.id} className="hover:bg-slate-850/30 transition-colors">
                    <td className="px-5 py-3.5 font-medium text-white max-w-xs truncate">
                      <div>{job.title}</div>
                      <div className="text-[10px] text-slate-500 font-mono mt-0.5">{job.id}</div>
                    </td>
                    <td className="px-4 py-3.5 font-mono text-[11px] text-cyan-300">
                      {job.parameters.modelArch.split(' ')[0]}
                    </td>
                    <td className="px-4 py-3.5 text-slate-400">
                      {job.imagery.sensor} ({job.imagery.resolutionMeters}m)
                    </td>
                    <td className="px-4 py-3.5 font-mono text-[11px] text-slate-400">
                      {job.imagery.preDate.slice(0, 7)} &rarr; {job.imagery.postDate.slice(0, 7)}
                    </td>
                    <td className="px-4 py-3.5">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="px-5 py-3.5 text-right">
                      {job.status === 'completed' ? (
                        <Link to={`/analysis/results/${job.id}`}>
                          <Button variant="outline" size="sm" rightIcon={<ChevronRight className="w-3.5 h-3.5" />}>
                            View Audit
                          </Button>
                        </Link>
                      ) : job.status === 'processing' ? (
                        <div className="flex items-center justify-end gap-2 text-cyan-400 font-mono text-[11px]">
                          <div className="w-16 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                            <div className="bg-cyan-500 h-full" style={{ width: `${job.progressPercent}%` }}></div>
                          </div>
                          <span>{job.progressPercent}%</span>
                        </div>
                      ) : (
                        <span className="text-slate-500 text-[11px]">In Queue</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
};
