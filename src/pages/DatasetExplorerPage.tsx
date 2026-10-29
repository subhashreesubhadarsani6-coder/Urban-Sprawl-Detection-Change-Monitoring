import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Database, 
  Search, 
  Filter, 
  ExternalLink, 
  HardDrive, 
  Layers, 
  Globe2, 
  Calendar, 
  Tag, 
  ArrowRight,
  Download
} from 'lucide-react';
import { PageHeader } from '../components/layout/PageHeader';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import { Modal } from '../components/common/Modal';
import { ApiService } from '../services/api';
import { DatasetItem } from '../types';
import { useToast } from '../context/ToastContext';

export const DatasetExplorerPage: React.FC = () => {
  const navigate = useNavigate();
  const { showToast } = useToast();

  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [search, setSearch] = useState('');
  const [selectedSensor, setSelectedSensor] = useState('all');
  const [activeModalDataset, setActiveModalDataset] = useState<DatasetItem | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchDatasets = async () => {
      setIsLoading(true);
      try {
        const data = await ApiService.getDatasets(selectedSensor, search);
        setDatasets(data);
      } catch (err) {
        console.error('Error loading datasets:', err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchDatasets();
  }, [selectedSensor, search]);

  const handleUseInAnalysis = (ds: DatasetItem) => {
    showToast({
      type: 'info',
      title: 'Dataset Loaded',
      message: `Configured ${ds.title} imagery presets for sprawl detection.`
    });
    navigate('/analysis/new');
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Satellite Datasets & Sprawl Benchmarks"
        subtitle="Explore curated multi-spectral satellite imagery archives, temporal urban challenges, and training ground truth."
        breadcrumbs={[
          { label: 'Workstation', to: '/dashboard' },
          { label: 'Datasets' }
        ]}
      />

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 bg-[#0d1322] p-3 rounded-xl border border-slate-800">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search datasets, sensors, cities..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full bg-[#080c14] border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <span className="text-xs text-slate-400 font-medium whitespace-nowrap">Filter Sensor:</span>
          <select
            value={selectedSensor}
            onChange={e => setSelectedSensor(e.target.value)}
            className="bg-[#080c14] border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All Satellite Sensors</option>
            <option value="Sentinel-2">Copernicus Sentinel-2 (10m)</option>
            <option value="PlanetScope">PlanetScope (3.0m)</option>
            <option value="WorldView">WorldView-3 (0.5m)</option>
            <option value="Landsat">Landsat-8 (30m)</option>
          </select>
        </div>
      </div>

      {/* Dataset Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {datasets.map(ds => (
          <Card
            key={ds.id}
            hoverEffect
            className="flex flex-col justify-between space-y-4"
          >
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-base font-bold text-white tracking-wide hover:text-cyan-400 transition-colors">
                    {ds.title}
                  </h3>
                  <div className="text-xs text-slate-400 mt-0.5">Provided by {ds.provider}</div>
                </div>
                <Badge variant="cyan" size="sm">{ds.resolution}</Badge>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed">
                {ds.description}
              </p>

              {/* Metadata specs */}
              <div className="grid grid-cols-2 gap-2 p-2.5 rounded-lg bg-[#080c14] border border-slate-800/80 text-[11px] font-mono text-slate-400">
                <div className="flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  <span>{ds.tileCount.toLocaleString()} Image Tiles</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <HardDrive className="w-3.5 h-3.5 text-emerald-400" />
                  <span>{ds.totalSizeGb} GB Archive</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Globe2 className="w-3.5 h-3.5 text-orange-400" />
                  <span className="truncate">{ds.region}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Calendar className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Updated {ds.lastUpdated}</span>
                </div>
              </div>

              {/* Tags */}
              <div className="flex items-center gap-1.5 flex-wrap">
                {ds.tags.map(t => (
                  <span key={t} className="text-[10px] px-2 py-0.5 rounded-md bg-slate-800/80 text-slate-400 font-mono">
                    #{t}
                  </span>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between pt-2 border-t border-slate-800">
              <span className="text-[11px] text-slate-500 font-mono truncate max-w-[180px]">
                {ds.license}
              </span>

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setActiveModalDataset(ds)}
                >
                  View Specs
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  rightIcon={<ArrowRight className="w-3.5 h-3.5" />}
                  onClick={() => handleUseInAnalysis(ds)}
                >
                  Load in Pipeline
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Dataset Details Modal */}
      {activeModalDataset && (
        <Modal
          isOpen={Boolean(activeModalDataset)}
          onClose={() => setActiveModalDataset(null)}
          title={activeModalDataset.title}
          subtitle={`Provider: ${activeModalDataset.provider}`}
          maxWidth="2xl"
          footer={
            <div className="flex items-center justify-between w-full">
              <span className="text-xs text-slate-500 font-mono">{activeModalDataset.license}</span>
              <div className="flex gap-2">
                <Button variant="secondary" size="sm" onClick={() => setActiveModalDataset(null)}>
                  Close
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => {
                    handleUseInAnalysis(activeModalDataset);
                    setActiveModalDataset(null);
                  }}
                >
                  Load in Sprawl Analysis
                </Button>
              </div>
            </div>
          }
        >
          <div className="space-y-4 text-xs">
            <p className="text-slate-300 leading-relaxed text-sm">
              {activeModalDataset.description}
            </p>

            <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800 space-y-2">
              <h4 className="font-semibold text-white uppercase text-[11px] tracking-wider">Spatial & Sensor Specifications</h4>
              <div className="grid grid-cols-2 gap-2 text-slate-400 font-mono">
                <div>Sensor: <span className="text-cyan-400">{activeModalDataset.sensor}</span></div>
                <div>Ground Sample Distance: <span className="text-white">{activeModalDataset.resolution}</span></div>
                <div>Total Volume: <span className="text-white">{activeModalDataset.totalSizeGb} GB</span></div>
                <div>Tile Count: <span className="text-white">{activeModalDataset.tileCount} scenes</span></div>
                <div className="col-span-2">Spatial Bounds: <span className="text-slate-300">{activeModalDataset.spatialExtent}</span></div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold text-white uppercase text-[11px] tracking-wider mb-1.5">Spectral Band Allocations</h4>
              <div className="flex gap-1.5 flex-wrap">
                {activeModalDataset.spectralBands.map(b => (
                  <span key={b} className="px-2 py-1 rounded bg-slate-800 text-cyan-300 font-mono text-[11px]">
                    {b}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};
