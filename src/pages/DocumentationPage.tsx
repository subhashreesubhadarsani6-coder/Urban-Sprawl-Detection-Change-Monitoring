import React, { useState } from 'react';
import { 
  BookOpen, 
  Code2, 
  FileText, 
  Layers, 
  Terminal, 
  Copy, 
  Check, 
  ExternalLink,
  Cpu,
  Database
} from 'lucide-react';
import { PageHeader } from '../components/layout/PageHeader';
import { Card } from '../components/common/Card';
import { Tabs } from '../components/common/Tabs';
import { Button } from '../components/common/Button';

export const DocumentationPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('spectral');
  const [copiedEndpoint, setCopiedEndpoint] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedEndpoint(id);
    setTimeout(() => setCopiedEndpoint(null), 2000);
  };

  const tabs = [
    { id: 'spectral', label: 'Spectral Indices Guide', icon: <Layers className="w-3.5 h-3.5" /> },
    { id: 'formats', label: 'Satellite Data Formats', icon: <FileText className="w-3.5 h-3.5" /> },
    { id: 'api', label: 'REST API Reference', icon: <Code2 className="w-3.5 h-3.5" /> },
    { id: 'backend', label: 'Python Backend Setup', icon: <Terminal className="w-3.5 h-3.5" /> }
  ];

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <PageHeader
        title="Technical Reference & Documentation"
        subtitle="Mathematical formulation of remote sensing indices, multi-spectral band architectures, and typed REST API contracts."
        breadcrumbs={[
          { label: 'Workstation', to: '/dashboard' },
          { label: 'Documentation' }
        ]}
      />

      <Tabs
        tabs={tabs}
        activeTab={activeTab}
        onChange={setActiveTab}
        variant="pills"
      />

      {/* Tab 1: Spectral Indices Guide */}
      {activeTab === 'spectral' && (
        <div className="space-y-5">
          <Card className="space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
              Normalized Difference Vegetation Index (NDVI)
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Measures healthy green vegetation density by comparing absorption in the red chlorophyll band against high reflectance in the near-infrared (NIR) cellular structure band.
            </p>
            <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800 font-mono text-xs text-emerald-400">
              NDVI = (NIR - Red) / (NIR + Red)
              <br />
              <span className="text-[11px] text-slate-500">Sentinel-2: (B08 - B04) / (B08 + B04) &bull; Landsat-8: (B5 - B4) / (B5 + B4)</span>
            </div>
            <div className="text-xs text-slate-400 space-y-1">
              <div>&bull; <strong>NDVI &gt; 0.5:</strong> Dense tree canopy, intact agricultural fields.</div>
              <div>&bull; <strong>NDVI &lt; 0.15:</strong> Built-up urban zones, barren soil, impervious concrete.</div>
            </div>
          </Card>

          <Card className="space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-orange-400"></span>
              Normalized Difference Built-up Index (NDBI)
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Differentiates human-made impervious surfaces (asphalt, concrete, roofing) from natural background landscapes using short-wave infrared (SWIR) reflectance.
            </p>
            <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800 font-mono text-xs text-orange-400">
              NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
              <br />
              <span className="text-[11px] text-slate-500">Sentinel-2: (B11 - B08) / (B11 + B08) &bull; Landsat-8: (B6 - B5) / (B6 + B5)</span>
            </div>
            <div className="text-xs text-slate-400 space-y-1">
              <div>&bull; <strong>NDBI &gt; 0.0:</strong> Active built-up residential subdivisions, commercial warehouses.</div>
              <div>&bull; <strong>NDBI &lt; -0.1:</strong> Forest, natural scrubland, open water bodies.</div>
            </div>
          </Card>

          <Card className="space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
              Modified Normalized Difference Water Index (MNDWI)
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Identifies natural rivers, retention ponds, and coastal boundaries while suppressing false alarms caused by urban shadow artifacts.
            </p>
            <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800 font-mono text-xs text-cyan-400">
              MNDWI = (Green - SWIR1) / (Green + SWIR1)
              <br />
              <span className="text-[11px] text-slate-500">Sentinel-2: (B03 - B11) / (B03 + B11)</span>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 2: Satellite Data Formats */}
      {activeTab === 'formats' && (
        <div className="space-y-4">
          <Card className="space-y-3">
            <h3 className="text-sm font-semibold text-white">Supported Raster & Vector Formats</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800 space-y-1.5">
                <div className="font-semibold text-cyan-400 font-mono">GeoTIFF (.tif, .tiff)</div>
                <p className="text-slate-400">
                  Standard georeferenced raster file storing spatial tie points, coordinate reference systems (CRS), and 16-bit multi-spectral reflectance numbers.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800 space-y-1.5">
                <div className="font-semibold text-emerald-400 font-mono">Cloud-Optimized GeoTIFF (COG)</div>
                <p className="text-slate-400">
                  Internally tiled raster with downscaled overview pyramids. Enables HTTP range requests so only the visible map viewport is downloaded.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800 space-y-1.5">
                <div className="font-semibold text-orange-400 font-mono">GeoJSON (.geojson)</div>
                <p className="text-slate-400">
                  WGS 84 polygon vector definitions of urban expansion boundaries, building clusters, and parcel conversion areas.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800 space-y-1.5">
                <div className="font-semibold text-purple-400 font-mono">ESRI Shapefile (.shp / .zip)</div>
                <p className="text-slate-400">
                  Industry standard GIS vector format supported for export into ArcGIS Pro and QGIS workstations.
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: REST API Reference */}
      {activeTab === 'api' && (
        <div className="space-y-4">
          <Card className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-mono uppercase text-cyan-400 font-bold">POST</span>
                <span className="text-sm font-mono text-white ml-2">/api/v1/analysis</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard('POST /api/v1/analysis', 'post-analysis')}
              >
                {copiedEndpoint === 'post-analysis' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              </Button>
            </div>
            <p className="text-xs text-slate-300">
              Submit a bi-temporal satellite pair for urban sprawl change detection inference.
            </p>
            <div className="p-3 rounded-lg bg-[#080c14] border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto">
              <pre>{`{
  "title": "Las Vegas Valley 2018-2024 Audit",
  "aoi_id": "aoi-las-vegas",
  "model_architecture": "ChangeFormer",
  "confidence_threshold": 0.85,
  "indices": {
    "calculate_ndvi": true,
    "calculate_ndbi": true,
    "calculate_mndwi": true
  },
  "t1_image_url": "s3://urbanscope-raw/20180615.tif",
  "t2_image_url": "s3://urbanscope-raw/20240620.tif"
}`}</pre>
            </div>
          </Card>

          <Card className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-mono uppercase text-emerald-400 font-bold">GET</span>
                <span className="text-sm font-mono text-white ml-2">/api/v1/analysis/{'{id}'}</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard('GET /api/v1/analysis/{id}', 'get-analysis')}
              >
                {copiedEndpoint === 'get-analysis' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              </Button>
            </div>
            <p className="text-xs text-slate-300">
              Retrieve pipeline status, change polygon vector coordinates, and LULC conversion matrices.
            </p>
          </Card>

          <Card className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-mono uppercase text-cyan-400 font-bold">POST</span>
                <span className="text-sm font-mono text-white ml-2">/api/v1/analysis/{'{id}'}/rethreshold</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard('POST /api/v1/analysis/{id}/rethreshold?threshold=0.6', 'rethresh')}
              >
                {copiedEndpoint === 'rethresh' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              </Button>
            </div>
            <p className="text-xs text-slate-300">
              Dynamically re-evaluate change mask, changed pixels, and area from saved probability map in under 150ms.
            </p>
          </Card>

          <Card className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-mono uppercase text-emerald-400 font-bold">GET</span>
                <span className="text-sm font-mono text-white ml-2">/api/v1/analysis/{'{id}'}/export/csv</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard('GET /api/v1/analysis/{id}/export/csv', 'csv-export')}
              >
                {copiedEndpoint === 'csv-export' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              </Button>
            </div>
            <p className="text-xs text-slate-300">
              Export tabular change metrics, transition matrices, and temporal trend tables in standard CSV format.
            </p>
          </Card>
        </div>
      )}

      {/* Tab 4: Backend Setup Guide */}
      {activeTab === 'backend' && (
        <Card className="space-y-4">
          <h3 className="text-base font-bold text-white">Connecting the Python / PyTorch Backend</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            The frontend is decoupled and configured with a typed API service layer (<code className="text-cyan-400">src/services/api.ts</code>). To connect your real Python inference backend:
          </p>
          <div className="p-4 rounded-lg bg-[#080c14] border border-slate-800 space-y-3 text-xs font-mono">
            <div className="text-slate-400"># 1. Create a .env file in the frontend root</div>
            <div className="text-cyan-300">VITE_API_BASE_URL=http://localhost:8000</div>
            <div className="text-slate-400 pt-2"># 2. Run your FastAPI / PyTorch inference server</div>
            <div className="text-emerald-400">uvicorn main:app --reload --port 8000</div>
          </div>
          <p className="text-xs text-slate-400">
            When <code className="text-cyan-400">VITE_API_BASE_URL</code> is present, the frontend automatically routes all requests to your live server. When omitted, it runs seamlessly in interactive demo mode using realistic geospatial fixtures.
          </p>
        </Card>
      )}
    </div>
  );
};
