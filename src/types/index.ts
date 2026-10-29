export type SensorType = 'Sentinel-2 MSI' | 'Landsat-8 OLI' | 'PlanetScope SuperDove' | 'WorldView-3';

export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export type ModelArchitecture = 'ChangeFormer (Transformer)' | 'Siam-UNet (Siamese CNN)' | 'BIT-CD (Bitemporal Trans)' | 'FC-Siam-diff';

export interface BoundingBox {
  minLat: number;
  minLng: number;
  maxLat: number;
  maxLng: number;
}

export interface AreaOfInterest {
  id: string;
  name: string;
  city: string;
  country: string;
  center: [number, number]; // [lat, lng]
  zoom: number;
  bbox: BoundingBox;
  monitoredSince: string;
  totalAreaKm2: number;
  urbanGrowthRateAnnual: number;
  riskLevel: 'Low' | 'Moderate' | 'High' | 'Severe';
  thumbnailUrl: string;
}

export interface MultiTemporalImageryPair {
  preDate: string;
  postDate: string;
  sensor: SensorType;
  resolutionMeters: number;
  cloudCoveragePre: number;
  cloudCoveragePost: number;
  bandsAvailable: string[];
  preImageUrl: string;
  postImageUrl: string;
  changeMaskUrl?: string;
  overlayUrl?: string;
  probabilityMapUrl?: string;
  geoTiffUrl?: string;
  geojsonUrl?: string;
  auditReportUrl?: string;
  ndviDiffUrl?: string;
}

export interface LulcClassBreakdown {
  className: 'Urban Built-Up' | 'Vegetation & Tree Canopy' | 'Agricultural Land' | 'Water Bodies' | 'Barren & Bare Soil';
  color: string;
  preAreaKm2: number;
  postAreaKm2: number;
  netChangeKm2: number;
  percentageChange: number;
}

export interface SpatialResolutionInfo {
  pixel_width_m: number;
  pixel_height_m: number;
  pixel_area_m2: number;
  is_calibrated: boolean;
  crs?: string | null;
  resolution_source?: string;
}

export interface SprawlMetrics {
  totalUrbanPreKm2: number;
  totalUrbanPostKm2: number;
  netExpansionKm2: number;
  expansionPercentage: number;
  annualSprawlRatePercent: number;
  vegetationLossKm2: number;
  waterEncroachmentKm2: number;
  imperviousSurfaceIncreasePercent: number;
  sprawlDispersionIndex: number; // Leapfrog / sprawling index
  compactnessRatio: number;
  changedPixels?: number;
  validPixels?: number;
  totalPixels?: number;
  appliedThreshold?: number;
  inferenceType?: string;
  modelArch?: string;
  checkpointUsed?: string;
  spatialResolution?: SpatialResolutionInfo;
}

export interface AnalysisJob {
  id: string;
  aoiId: string;
  title: string;
  description: string;
  status: JobStatus;
  progressPercent: number;
  currentStep?: string;
  createdAt: string;
  completedAt?: string;
  parameters: {
    modelArch: ModelArchitecture;
    confidenceThreshold: number;
    calculateNdvi: boolean;
    calculateNdbi: boolean;
    calculateMndwi: boolean;
    cloudMaskTolerance: number;
    spatialFilteringRadius: number;
    isRealInference?: boolean;
  };
  imagery: MultiTemporalImageryPair;
  metrics?: SprawlMetrics;
  lulcBreakdown?: LulcClassBreakdown[];
  temporalTrend?: { year: number; urbanAreaKm2: number; vegetationAreaKm2: number }[];
}

export interface DatasetItem {
  id: string;
  title: string;
  provider: string;
  sensor: SensorType;
  resolution: string;
  tileCount: number;
  totalSizeGb: number;
  license: string;
  region: string;
  spatialExtent: string;
  spectralBands: string[];
  description: string;
  downloadUrl: string;
  lastUpdated: string;
  tags: string[];
}

export interface TrainingHyperparameters {
  modelArchitecture: ModelArchitecture;
  backbone: 'ResNet-50' | 'Swin-Transformer-Base' | 'EfficientNet-B4' | 'MiT-B2';
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: 'AdamW' | 'Adam' | 'SGD + Momentum';
  lossFunction: 'Combined (Focal + Dice)' | 'BCE + Dice Loss' | 'Contrastive Loss';
  weightDecay: number;
  validationSplit: number;
  useDataAugmentation: boolean;
  cropSize: number;
}

export interface TrainingTelemetryPoint {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  mIoU: number;
  f1Score: number;
  learningRate: number;
  timeSec: number;
}

export interface TrainingSession {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  currentEpoch: number;
  totalEpochs: number;
  config: TrainingHyperparameters;
  history: TrainingTelemetryPoint[];
  bestMIoU: number;
  bestF1Score: number;
  estimatedTimeRemainingSec?: number;
}

export interface ModelBenchmark {
  id: string;
  modelName: string;
  architecture: string;
  backbone: string;
  parametersM: number;
  inferenceSpeedMs: number;
  f1Score: number;
  mIoU: number;
  precision: number;
  recall: number;
  falsePositiveRate: number;
  memoryFootprintMb: number;
}

export interface ConfusionMatrixClass {
  actual: string;
  predicted: string;
  value: number; // percentage or count
}
