import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  UploadCloud, 
  MapPin, 
  Calendar, 
  Sliders, 
  CheckCircle2, 
  FileText, 
  Info, 
  Sparkles,
  Layers,
  ArrowRight,
  Loader2,
  Globe,
  Search,
  Satellite
} from 'lucide-react';
import { PageHeader } from '../components/layout/PageHeader';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { useToast } from '../context/ToastContext';
import { ApiService } from '../services/api';
import { SensorType, ModelArchitecture } from '../types';

export const AnalysisUploadPage: React.FC = () => {
  const navigate = useNavigate();
  const { showToast } = useToast();

  const [title, setTitle] = useState('Southwest Metro Urban Sprawl Bi-Temporal Audit');
  const [description, setDescription] = useState('Change detection audit tracking residential expansion and agricultural land conversion.');
  const [selectedAoi, setSelectedAoi] = useState('aoi-las-vegas');
  const [sensor, setSensor] = useState<SensorType>('Sentinel-2 MSI');
  const [modelArch, setModelArch] = useState<ModelArchitecture>('FC-Siam-diff');
  
  const [preDate, setPreDate] = useState('2018-06-15');
  const [postDate, setPostDate] = useState('2024-06-20');

  const [calcNdvi, setCalcNdvi] = useState(true);
  const [calcNdbi, setCalcNdbi] = useState(true);
  const [calcMndwi, setCalcMndwi] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.85);
  const [cloudTolerance, setCloudTolerance] = useState(10);

  // File states
  const [preFile, setPreFile] = useState<File | null>(null);
  const [postFile, setPostFile] = useState<File | null>(null);
  const [preFileName, setPreFileName] = useState<string>('S2A_MSIL2A_20180615_T11SPA.tif (GeoTIFF)');
  const [postFileName, setPostFileName] = useState<string>('S2B_MSIL2A_20240620_T11SPA.tif (GeoTIFF)');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [pipelineStep, setPipelineStep] = useState<string>('');

  // Satellite Ingestion States
  const [ingestionMode, setIngestionMode] = useState<'upload' | 'copernicus'>('upload');
  const [minLon, setMinLon] = useState<number>(-115.35);
  const [minLat, setMinLat] = useState<number>(36.05);
  const [maxLon, setMaxLon] = useState<number>(-115.10);
  const [maxLat, setMaxLat] = useState<number>(36.25);
  const [isSearchingScenes, setIsSearchingScenes] = useState<boolean>(false);
  const [discoveredScenes, setDiscoveredScenes] = useState<any[]>([]);
  const [providerHealth, setProviderHealth] = useState<any>(null);

  useEffect(() => {
    ApiService.getSatelliteProviderHealth()
      .then(setProviderHealth)
      .catch(() => setProviderHealth({ connected: false, provider: 'Satellite Provider', authenticated: false, message: 'Check credentials.' }));
  }, []);

  const handleSearchScenes = async () => {
    setIsSearchingScenes(true);
    try {
      const res = await ApiService.searchSatelliteScenes({
        min_lon: minLon,
        min_lat: minLat,
        max_lon: maxLon,
        max_lat: maxLat,
        date_from: preDate,
        date_to: postDate,
        max_cloud_cover: cloudTolerance
      });
      setDiscoveredScenes(res.scenes);
      showToast({
        type: 'success',
        title: 'Satellite Imagery Catalog Found',
        message: `Located ${res.count} cloud-filtered scenes from ${providerHealth?.provider || 'Satellite Provider'}.`
      });
    } catch (err: any) {
      showToast({
        type: 'error',
        title: 'Catalog Query Failed',
        message: err.message || 'Could not query satellite provider'
      });
    } finally {
      setIsSearchingScenes(false);
    }
  };

  const handleSelectScenePair = (t1: any, t2: any) => {
    setPreFileName(`${t1.scene_id}.tif (Copernicus L2A)`);
    setPostFileName(`${t2.scene_id}.tif (Copernicus L2A)`);
    setPreDate(t1.acquisition_date);
    setPostDate(t2.acquisition_date);
    showToast({
      type: 'info',
      title: 'Satellite Pair Staged',
      message: `Selected ${t1.platform} (${t1.acquisition_date}) and ${t2.platform} (${t2.acquisition_date}).`
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      let preUrl = 'https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=1200&q=80';
      let postUrl = 'https://images.unsplash.com/photo-1477959858617-67f30bc75b82?auto=format&fit=crop&w=1200&q=80';

      if (preFile) {
        setPipelineStep(`Uploading ${preFile.name} to secure backend storage...`);
        const preResult = await ApiService.uploadFile(preFile);
        preUrl = preResult.url;
      }

      if (postFile) {
        setPipelineStep(`Uploading ${postFile.name} to secure backend storage...`);
        const postResult = await ApiService.uploadFile(postFile);
        postUrl = postResult.url;
      }

      setPipelineStep('Initializing analysis job in SQLite database...');
      const createdJob = await ApiService.createAnalysisJob({
        title,
        description,
        aoiId: selectedAoi,
        parameters: {
          modelArch,
          confidenceThreshold,
          calculateNdvi: calcNdvi,
          calculateNdbi: calcNdbi,
          calculateMndwi: calcMndwi,
          cloudMaskTolerance: cloudTolerance,
          spatialFilteringRadius: 3
        },
        imagery: {
          preDate,
          postDate,
          sensor,
          resolutionMeters: sensor === 'PlanetScope SuperDove' ? 3 : 10,
          cloudCoveragePre: 1.2,
          cloudCoveragePost: 0.9,
          bandsAvailable: ['B02', 'B03', 'B04', 'B08', 'B11'],
          preImageUrl: preUrl,
          postImageUrl: postUrl
        }
      });

      showToast({
        type: 'success',
        title: 'Analysis Job Initialized',
        message: `Task ${createdJob.id} queued successfully for remote sensing inference.`
      });

      // Redirect directly to the live created job!
      navigate(`/analysis/results/${createdJob.id}`);
    } catch (err: any) {
      showToast({
        type: 'error',
        title: 'Submission Failed',
        message: err.message || 'Could not queue analysis job'
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <PageHeader
        title="Launch Sprawl Analysis Pipeline"
        subtitle="Ingest multi-temporal satellite imagery pairs (T1 vs T2) to detect urban sprawl and compute land-cover conversion."
        breadcrumbs={[
          { label: 'Workstation', to: '/dashboard' },
          { label: 'Sprawl Analysis', to: '/analysis/new' },
          { label: 'New Upload' }
        ]}
      />

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Step 1: Project Metadata & Region */}
        <Card className="space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <span className="w-6 h-6 rounded-md bg-cyan-950 text-cyan-400 font-mono text-xs flex items-center justify-center font-bold">1</span>
            <h3 className="text-sm font-semibold text-white">Analysis Project & Target AOI</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium text-slate-300 block mb-1">Audit Title</label>
              <input
                type="text"
                value={title}
                onChange={e => setTitle(e.target.value)}
                required
                className="w-full bg-[#080c14] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-slate-300 block mb-1">Target Area of Interest (AOI)</label>
              <select
                value={selectedAoi}
                onChange={e => setSelectedAoi(e.target.value)}
                className="w-full bg-[#080c14] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
              >
                <option value="aoi-las-vegas">Las Vegas Valley Southwest Expansion (USA)</option>
                <option value="aoi-austin">Austin East Tech Corridor & Suburban Fringe (USA)</option>
                <option value="aoi-bengaluru">Bengaluru Southern Peripheral Corridor (India)</option>
                <option value="aoi-dubai">Dubai South & Al Maktoum Logistics Zone (UAE)</option>
                <option value="custom">Custom GeoJSON Bounding Box (Lat/Lng input)</option>
              </select>
            </div>
          </div>
        </Card>

        {/* Step 2: Multi-Temporal Imagery Uploaders / Ingestion */}
        <Card className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3 flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-md bg-cyan-950 text-cyan-400 font-mono text-xs flex items-center justify-center font-bold">2</span>
              <h3 className="text-sm font-semibold text-white">Satellite Imagery Ingestion</h3>
            </div>
            
            <div className="flex items-center gap-2">
              <div className="flex bg-[#080c14] p-1 rounded-lg border border-slate-800 text-xs">
                <button
                  type="button"
                  onClick={() => setIngestionMode('upload')}
                  className={`px-3 py-1 rounded font-medium transition-colors ${ingestionMode === 'upload' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'}`}
                >
                  Local Raster Upload
                </button>
                <button
                  type="button"
                  onClick={() => setIngestionMode('copernicus')}
                  className={`px-3 py-1 rounded font-medium transition-colors flex items-center gap-1.5 ${ingestionMode === 'copernicus' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'}`}
                >
                  <Satellite className="w-3.5 h-3.5" />
                  Copernicus Data Space
                </button>
              </div>
            </div>
          </div>

          {ingestionMode === 'copernicus' ? (
            <div className="p-4 rounded-xl border border-cyan-500/20 bg-cyan-950/10 space-y-4">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <Globe className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs font-semibold text-white">Copernicus / Sentinel Hub Satellite Search</span>
                </div>
                {providerHealth && (
                  <span className={`text-[11px] px-2.5 py-0.5 rounded-full border ${
                    providerHealth.authenticated 
                      ? 'bg-emerald-950/60 text-emerald-400 border-emerald-500/30' 
                      : 'bg-amber-950/60 text-amber-400 border-amber-500/30'
                  }`}>
                    {providerHealth.provider} ({providerHealth.authenticated ? 'Connected' : 'Offline Sandbox'})
                  </span>
                )}
              </div>

              {/* Bounding Box Inputs */}
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-2">AOI Bounding Box (WGS84 Coordinates)</label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <div>
                    <span className="text-[10px] text-slate-400 block mb-0.5">Min Longitude</span>
                    <input
                      type="number"
                      step="0.01"
                      value={minLon}
                      onChange={e => setMinLon(parseFloat(e.target.value))}
                      className="w-full bg-[#080c14] border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                    />
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block mb-0.5">Min Latitude</span>
                    <input
                      type="number"
                      step="0.01"
                      value={minLat}
                      onChange={e => setMinLat(parseFloat(e.target.value))}
                      className="w-full bg-[#080c14] border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                    />
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block mb-0.5">Max Longitude</span>
                    <input
                      type="number"
                      step="0.01"
                      value={maxLon}
                      onChange={e => setMaxLon(parseFloat(e.target.value))}
                      className="w-full bg-[#080c14] border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                    />
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block mb-0.5">Max Latitude</span>
                    <input
                      type="number"
                      step="0.01"
                      value={maxLat}
                      onChange={e => setMaxLat(parseFloat(e.target.value))}
                      className="w-full bg-[#080c14] border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                    />
                  </div>
                </div>
              </div>

              {/* Dates & Cloud Slider */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <span className="text-[11px] text-slate-400 block mb-1">T1 Start Date</span>
                  <input
                    type="date"
                    value={preDate}
                    onChange={e => setPreDate(e.target.value)}
                    className="w-full bg-[#080c14] border border-slate-700 rounded px-2.5 py-1.5 text-xs text-white"
                  />
                </div>
                <div>
                  <span className="text-[11px] text-slate-400 block mb-1">T2 End Date</span>
                  <input
                    type="date"
                    value={postDate}
                    onChange={e => setPostDate(e.target.value)}
                    className="w-full bg-[#080c14] border border-slate-700 rounded px-2.5 py-1.5 text-xs text-white"
                  />
                </div>
                <div>
                  <span className="text-[11px] text-slate-400 block mb-1">Max Cloud Tolerance: {cloudTolerance}%</span>
                  <input
                    type="range"
                    min="0"
                    max="30"
                    value={cloudTolerance}
                    onChange={e => setCloudTolerance(parseInt(e.target.value))}
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between pt-2">
                <Button
                  type="button"
                  variant="primary"
                  size="sm"
                  onClick={handleSearchScenes}
                  disabled={isSearchingScenes}
                  leftIcon={isSearchingScenes ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
                >
                  {isSearchingScenes ? 'Searching Catalog...' : 'Query Satellite Catalog'}
                </Button>
                <span className="text-[11px] text-slate-400 font-mono">
                  {discoveredScenes.length > 0 ? `${discoveredScenes.length} candidate scenes available` : 'Ready to search'}
                </span>
              </div>

              {/* Discovered Scenes List */}
              {discoveredScenes.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                  {discoveredScenes.map((scene, idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-[#080c14] border border-slate-800 space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-cyan-400 font-mono">{scene.platform}</span>
                        <span className="text-[10px] text-slate-400">{scene.cloud_coverage_percent}% clouds</span>
                      </div>
                      <div className="text-xs text-white font-mono truncate">{scene.scene_id}</div>
                      <div className="text-[10px] text-slate-400">Date: {scene.acquisition_date} &bull; {scene.resolution_meters}m GSD</div>
                      <div className="text-[10px] text-slate-500 truncate">Bands: {scene.bands?.join(', ')}</div>
                    </div>
                  ))}
                  <div className="col-span-full">
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      onClick={() => handleSelectScenePair(discoveredScenes[0], discoveredScenes[1] || discoveredScenes[0])}
                      className="w-full"
                    >
                      Stage This Satellite Pair for Deep-Learning Audit
                    </Button>
                  </div>
                </div>
              )}
            </div>
          ) : null}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* T1 Pre-Period Dropzone */}
            <div className="p-4 rounded-xl border-2 border-dashed border-slate-700/80 bg-[#080c14]/60 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                  <Calendar className="w-3.5 h-3.5" /> Historical Observation (T1)
                </span>
                <span className="text-[10px] bg-emerald-950/60 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20">Pre-Period</span>
              </div>

              <div>
                <label className="text-[11px] text-slate-400 block mb-1">Acquisition Date</label>
                <input
                  type="date"
                  value={preDate}
                  onChange={e => setPreDate(e.target.value)}
                  className="w-full bg-[#0d1322] border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
                />
              </div>

              <div className="p-3 rounded-lg bg-[#0d1322] border border-slate-800 flex items-center gap-3">
                <FileText className="w-6 h-6 text-emerald-400 shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className="text-xs text-slate-200 font-mono truncate">{preFileName}</div>
                  <div className="text-[10px] text-slate-500">4 Bands (B02, B03, B04, B08) &bull; 84.2 MB</div>
                </div>
              </div>

              <label className="block text-center py-2 px-3 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-xs text-slate-300 cursor-pointer transition-colors">
                <span>Select Replacement Satellite Image / GeoTIFF</span>
                <input
                  type="file"
                  className="hidden"
                  accept=".tif,.tiff,.png,.jpg,.jpeg,.jp2"
                  onChange={e => {
                    if (e.target.files?.[0]) {
                      setPreFile(e.target.files[0]);
                      setPreFileName(e.target.files[0].name);
                    }
                  }}
                />
              </label>
            </div>

            {/* T2 Post-Period Dropzone */}
            <div className="p-4 rounded-xl border-2 border-dashed border-slate-700/80 bg-[#080c14]/60 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-orange-400 flex items-center gap-1.5">
                  <Calendar className="w-3.5 h-3.5" /> Contemporary Observation (T2)
                </span>
                <span className="text-[10px] bg-orange-950/60 text-orange-400 px-2 py-0.5 rounded border border-orange-500/20">Post-Period</span>
              </div>

              <div>
                <label className="text-[11px] text-slate-400 block mb-1">Acquisition Date</label>
                <input
                  type="date"
                  value={postDate}
                  onChange={e => setPostDate(e.target.value)}
                  className="w-full bg-[#0d1322] border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
                />
              </div>

              <div className="p-3 rounded-lg bg-[#0d1322] border border-slate-800 flex items-center gap-3">
                <FileText className="w-6 h-6 text-orange-400 shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className="text-xs text-slate-200 font-mono truncate">{postFileName}</div>
                  <div className="text-[10px] text-slate-500">4 Bands (B02, B03, B04, B08) &bull; 86.8 MB</div>
                </div>
              </div>

              <label className="block text-center py-2 px-3 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-xs text-slate-300 cursor-pointer transition-colors">
                <span>Select Replacement Satellite Image / GeoTIFF</span>
                <input
                  type="file"
                  className="hidden"
                  accept=".tif,.tiff,.png,.jpg,.jpeg,.jp2"
                  onChange={e => {
                    if (e.target.files?.[0]) {
                      setPostFile(e.target.files[0]);
                      setPostFileName(e.target.files[0].name);
                    }
                  }}
                />
              </label>
            </div>
          </div>
        </Card>

        {/* Step 3: Model & Spectral Processing Parameters */}
        <Card className="space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <span className="w-6 h-6 rounded-md bg-cyan-950 text-cyan-400 font-mono text-xs flex items-center justify-center font-bold">3</span>
            <h3 className="text-sm font-semibold text-white">Model Selection & Spectral Indices</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium text-slate-300 block mb-1">Satellite Constellation</label>
              <select
                value={sensor}
                onChange={e => setSensor(e.target.value as SensorType)}
                className="w-full bg-[#080c14] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white"
              >
                <option value="Sentinel-2 MSI">Copernicus Sentinel-2 MSI (10m Resolution)</option>
                <option value="Landsat-8 OLI">USGS Landsat-8 / 9 OLI (30m Resolution)</option>
                <option value="PlanetScope SuperDove">PlanetScope SuperDove (3.0m High-Cadence)</option>
                <option value="WorldView-3">Maxar WorldView-3 (0.5m Very-High-Resolution)</option>
              </select>
            </div>

            <div>
              <label className="text-xs font-medium text-slate-300 block mb-1">Deep Learning Architecture</label>
              <select
                value={modelArch}
                onChange={e => setModelArch(e.target.value as ModelArchitecture)}
                className="w-full bg-[#080c14] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white"
              >
                <option value="ChangeFormer (Transformer)">ChangeFormer (Siamese Hierarchical Transformer - 92.4% F1)</option>
                <option value="BIT-CD (Bitemporal Trans)">BIT-CD (Bitemporal Image Transformer - Fast 28ms)</option>
                <option value="Siam-UNet (Siamese CNN)">Siam-UNet (Dual-branch Siamese ConvNet)</option>
                <option value="FC-Siam-diff">FC-Siam-diff (Baseline Difference Net)</option>
              </select>
            </div>
          </div>

          {/* Spectral Indices Checkboxes */}
          <div className="pt-2">
            <label className="text-xs font-medium text-slate-300 block mb-2">Automated Remote Sensing Indices</label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <label className="p-3 rounded-lg bg-[#080c14] border border-slate-800 flex items-center justify-between cursor-pointer hover:border-cyan-500/40">
                <div>
                  <div className="text-xs font-semibold text-emerald-400">NDVI</div>
                  <div className="text-[10px] text-slate-400">Normalized Veg Index</div>
                </div>
                <input
                  type="checkbox"
                  checked={calcNdvi}
                  onChange={e => setCalcNdvi(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500"
                />
              </label>

              <label className="p-3 rounded-lg bg-[#080c14] border border-slate-800 flex items-center justify-between cursor-pointer hover:border-cyan-500/40">
                <div>
                  <div className="text-xs font-semibold text-orange-400">NDBI</div>
                  <div className="text-[10px] text-slate-400">Built-Up Index</div>
                </div>
                <input
                  type="checkbox"
                  checked={calcNdbi}
                  onChange={e => setCalcNdbi(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500"
                />
              </label>

              <label className="p-3 rounded-lg bg-[#080c14] border border-slate-800 flex items-center justify-between cursor-pointer hover:border-cyan-500/40">
                <div>
                  <div className="text-xs font-semibold text-cyan-400">MNDWI</div>
                  <div className="text-[10px] text-slate-400">Water Mask Index</div>
                </div>
                <input
                  type="checkbox"
                  checked={calcMndwi}
                  onChange={e => setCalcMndwi(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500"
                />
              </label>
            </div>
          </div>

          {/* Threshold Sliders */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Sprawl Confidence Threshold</span>
                <span className="font-mono text-cyan-400 font-bold">{Math.round(confidenceThreshold * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.99"
                step="0.01"
                value={confidenceThreshold}
                onChange={e => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Cloud Mask Tolerance</span>
                <span className="font-mono text-cyan-400 font-bold">&le; {cloudTolerance}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="30"
                step="1"
                value={cloudTolerance}
                onChange={e => setCloudTolerance(parseInt(e.target.value))}
                className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>
          </div>
        </Card>

        {/* Submit action */}
        <div className="flex items-center justify-between p-4 rounded-xl bg-[#0d1322] border border-slate-800">
          <div className="text-xs text-slate-400 font-mono">
            {isSubmitting ? pipelineStep : 'Ready to submit to PyTorch inference worker'}
          </div>

          <Button
            type="submit"
            variant="primary"
            size="md"
            isLoading={isSubmitting}
            rightIcon={<ArrowRight className="w-4 h-4" />}
          >
            {isSubmitting ? 'Processing Pipeline...' : 'Run Sprawl Change Detection'}
          </Button>
        </div>
      </form>
    </div>
  );
};
