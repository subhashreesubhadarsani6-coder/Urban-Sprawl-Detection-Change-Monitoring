import { 
  AreaOfInterest, 
  AnalysisJob, 
  DatasetItem, 
  ModelBenchmark, 
  TrainingSession, 
  TrainingHyperparameters,
  JobStatus
} from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface UploadResult {
  status: string;
  filename: string;
  original_name: string;
  size_bytes: number;
  extension: string;
  url: string;
}

export interface SystemStatus {
  engine_status: string;
  active_gpu: string;
  gpu_memory_used_mb: number;
  gpu_memory_total_mb: number;
  active_jobs_count: number;
  sentinel_api_connected: boolean;
  storage_used_mb: number;
}

export interface JobProgress {
  id: string;
  status: string;
  progressPercent: number;
  currentStep: string;
}

export const ApiService = {
  isLiveBackend: true,
  baseUrl: BASE_URL,

  // Area of Interest (AOI) Endpoints
  async getAois(): Promise<AreaOfInterest[]> {
    const res = await fetch(`${BASE_URL}/api/v1/aoi`);
    if (!res.ok) throw new Error(`Failed to fetch AOIs: ${res.statusText}`);
    return res.json();
  },

  async getAoiById(id: string): Promise<AreaOfInterest> {
    const res = await fetch(`${BASE_URL}/api/v1/aoi/${id}`);
    if (!res.ok) throw new Error(`Failed to fetch AOI '${id}': ${res.statusText}`);
    return res.json();
  },

  // Analysis & Change Detection Jobs
  async getAnalysisJobs(statusFilter?: JobStatus): Promise<AnalysisJob[]> {
    const query = statusFilter ? `?status=${statusFilter}` : '';
    const res = await fetch(`${BASE_URL}/api/v1/analysis${query}`);
    if (!res.ok) throw new Error(`Failed to fetch analysis jobs: ${res.statusText}`);
    return res.json();
  },

  async getAnalysisJobById(id: string): Promise<AnalysisJob> {
    const res = await fetch(`${BASE_URL}/api/v1/analysis/${id}`);
    if (!res.ok) throw new Error(`Failed to fetch analysis job '${id}': ${res.statusText}`);
    return res.json();
  },

  async getJobProgress(id: string): Promise<JobProgress> {
    const res = await fetch(`${BASE_URL}/api/v1/analysis/${id}/progress`);
    if (!res.ok) throw new Error(`Failed to fetch progress for job '${id}'`);
    return res.json();
  },

  async rethresholdAnalysisJob(id: string, threshold: number): Promise<AnalysisJob> {
    const res = await fetch(`${BASE_URL}/api/v1/analysis/${id}/rethreshold?threshold=${threshold}`, {
      method: 'POST'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to rethreshold analysis job');
    }
    return res.json();
  },

  async createAnalysisJob(data: {
    title: string;
    description?: string;
    aoiId: string;
    parameters?: any;
    imagery?: any;
  }): Promise<AnalysisJob> {
    const res = await fetch(`${BASE_URL}/api/v1/analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to create analysis job');
    }
    return res.json();
  },

  // File Upload Endpoint
  async uploadFile(file: File): Promise<UploadResult> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${BASE_URL}/api/v1/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'File upload failed');
    }
    return res.json();
  },

  getArtifactDownloadUrl(jobId: string, filename: string): string {
    return `${BASE_URL}/api/v1/artifacts/${jobId}/${filename}`;
  },

  getMetricsCsvDownloadUrl(jobId: string): string {
    return `${BASE_URL}/api/v1/analysis/${jobId}/export/csv`;
  },

  // Datasets Endpoints
  async getDatasets(sensorFilter?: string, search?: string): Promise<DatasetItem[]> {
    const params = new URLSearchParams();
    if (sensorFilter && sensorFilter !== 'all') params.append('sensor', sensorFilter);
    if (search) params.append('q', search);

    const res = await fetch(`${BASE_URL}/api/v1/datasets?${params.toString()}`);
    if (!res.ok) throw new Error(`Failed to fetch datasets: ${res.statusText}`);
    return res.json();
  },

  // Model Training Endpoints
  async getTrainingSession(): Promise<TrainingSession> {
    const res = await fetch(`${BASE_URL}/api/v1/training/active`);
    if (!res.ok) throw new Error(`Failed to fetch training status: ${res.statusText}`);
    return res.json();
  },

  async startTrainingRun(config: TrainingHyperparameters): Promise<TrainingSession> {
    const res = await fetch(`${BASE_URL}/api/v1/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to start training session');
    }
    return res.json();
  },

  async togglePauseTraining(): Promise<TrainingSession> {
    const res = await fetch(`${BASE_URL}/api/v1/training/pause`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error('Failed to toggle pause on training');
    return res.json();
  },

  // Model Evaluation & Benchmarks
  async getBenchmarks(): Promise<ModelBenchmark[]> {
    const res = await fetch(`${BASE_URL}/api/v1/evaluation/benchmarks`);
    if (!res.ok) throw new Error(`Failed to fetch benchmarks: ${res.statusText}`);
    return res.json();
  },

  async getConfusionMatrix(): Promise<{ actual: string; predicted: string; value: number }[]> {
    const res = await fetch(`${BASE_URL}/api/v1/evaluation/confusion-matrix`);
    if (!res.ok) throw new Error(`Failed to fetch confusion matrix: ${res.statusText}`);
    return res.json();
  },

  // System status
  async getSystemStatus(): Promise<SystemStatus> {
    const res = await fetch(`${BASE_URL}/api/v1/status`);
    if (!res.ok) throw new Error('Failed to fetch system status');
    return res.json();
  },

  // Satellite Provider Ingestion API
  async getSatelliteProviderHealth(): Promise<{
    connected: boolean;
    provider: string;
    authenticated: boolean;
    message: string;
    available_collections: string[];
  }> {
    const res = await fetch(`${BASE_URL}/api/v1/satellite/provider/health`);
    if (!res.ok) throw new Error('Failed to fetch satellite provider status');
    return res.json();
  },

  async searchSatelliteScenes(params: {
    min_lon: number;
    min_lat: number;
    max_lon: number;
    max_lat: number;
    date_from: string;
    date_to: string;
    max_cloud_cover?: number;
  }): Promise<{ status: string; count: number; scenes: any[] }> {
    const res = await fetch(`${BASE_URL}/api/v1/satellite/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to search satellite scenes');
    }
    return res.json();
  },

  async acquireSatellitePair(params: {
    t1_scene_id: string;
    t2_scene_id: string;
    min_lon: number;
    min_lat: number;
    max_lon: number;
    max_lat: number;
  }): Promise<any> {
    const res = await fetch(`${BASE_URL}/api/v1/satellite/acquire`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to acquire satellite scenes');
    }
    return res.json();
  }
};
