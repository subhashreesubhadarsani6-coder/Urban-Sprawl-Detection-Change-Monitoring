import { 
  AreaOfInterest, 
  AnalysisJob, 
  DatasetItem, 
  ModelBenchmark, 
  TrainingSession,
  ConfusionMatrixClass
} from '../types';

export const MOCK_AOIS: AreaOfInterest[] = [
  {
    id: 'aoi-las-vegas',
    name: 'Las Vegas Valley Southwest Expansion',
    city: 'Las Vegas, NV',
    country: 'United States',
    center: [36.084, -115.25],
    zoom: 12,
    bbox: {
      minLat: 35.98,
      minLng: -115.35,
      maxLat: 36.18,
      maxLng: -115.15
    },
    monitoredSince: '2018-05-10',
    totalAreaKm2: 485.2,
    urbanGrowthRateAnnual: 4.8,
    riskLevel: 'Severe',
    thumbnailUrl: 'https://images.unsplash.com/photo-1581351123004-757df051db8e?auto=format&fit=crop&w=600&q=80'
  },
  {
    id: 'aoi-austin',
    name: 'Austin East Tech Corridor & Suburban Fringe',
    city: 'Austin, TX',
    country: 'United States',
    center: [30.267, -97.743],
    zoom: 12,
    bbox: {
      minLat: 30.15,
      minLng: -97.85,
      maxLat: 30.40,
      maxLng: -97.60
    },
    monitoredSince: '2019-03-15',
    totalAreaKm2: 362.4,
    urbanGrowthRateAnnual: 5.6,
    riskLevel: 'High',
    thumbnailUrl: 'https://images.unsplash.com/photo-1531218150217-54595bc2b934?auto=format&fit=crop&w=600&q=80'
  },
  {
    id: 'aoi-bengaluru',
    name: 'Bengaluru Southern Peripheral Corridor',
    city: 'Bengaluru',
    country: 'India',
    center: [12.925, 77.625],
    zoom: 12,
    bbox: {
      minLat: 12.82,
      minLng: 77.52,
      maxLat: 13.02,
      maxLng: 77.75
    },
    monitoredSince: '2017-01-20',
    totalAreaKm2: 520.8,
    urbanGrowthRateAnnual: 6.2,
    riskLevel: 'Severe',
    thumbnailUrl: 'https://images.unsplash.com/photo-1596176530529-78163a4f7af2?auto=format&fit=crop&w=600&q=80'
  },
  {
    id: 'aoi-dubai',
    name: 'Dubai South & Al Maktoum Logistics Zone',
    city: 'Dubai',
    country: 'United Arab Emirates',
    center: [24.896, 55.161],
    zoom: 11,
    bbox: {
      minLat: 24.78,
      minLng: 55.05,
      maxLat: 25.02,
      maxLng: 55.28
    },
    monitoredSince: '2020-09-01',
    totalAreaKm2: 410.5,
    urbanGrowthRateAnnual: 3.9,
    riskLevel: 'Moderate',
    thumbnailUrl: 'https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&w=600&q=80'
  }
];

export const MOCK_ANALYSIS_JOBS: AnalysisJob[] = [
  {
    id: 'job-vegas-2018-2024',
    aoiId: 'aoi-las-vegas',
    title: 'Las Vegas Valley Bi-temporal Sprawl Audit (2018 - 2024)',
    description: 'Bitemporal change detection measuring desert conversion to residential subdivisions and commercial logistics hubs.',
    status: 'completed',
    progressPercent: 100,
    createdAt: '2024-10-12T14:20:00Z',
    completedAt: '2024-10-12T14:26:42Z',
    parameters: {
      modelArch: 'ChangeFormer (Transformer)',
      confidenceThreshold: 0.85,
      calculateNdvi: true,
      calculateNdbi: true,
      calculateMndwi: true,
      cloudMaskTolerance: 10,
      spatialFilteringRadius: 3
    },
    imagery: {
      preDate: '2018-06-15',
      postDate: '2024-06-20',
      sensor: 'Sentinel-2 MSI',
      resolutionMeters: 10,
      cloudCoveragePre: 1.2,
      cloudCoveragePost: 0.8,
      bandsAvailable: ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B08 (NIR)', 'B11 (SWIR-1)', 'B12 (SWIR-2)'],
      // High quality satellite imagery representative of pre and post urbanization
      preImageUrl: 'https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=1200&q=80',
      postImageUrl: 'https://images.unsplash.com/photo-1477959858617-67f30bc75b82?auto=format&fit=crop&w=1200&q=80',
      changeMaskUrl: 'https://images.unsplash.com/photo-1524661135-423995f22d0b?auto=format&fit=crop&w=1200&q=80'
    },
    metrics: {
      totalUrbanPreKm2: 184.2,
      totalUrbanPostKm2: 247.6,
      netExpansionKm2: 63.4,
      expansionPercentage: 34.42,
      annualSprawlRatePercent: 5.73,
      vegetationLossKm2: 12.8,
      waterEncroachmentKm2: 0.4,
      imperviousSurfaceIncreasePercent: 28.9,
      sprawlDispersionIndex: 0.74,
      compactnessRatio: 0.41
    },
    lulcBreakdown: [
      { className: 'Urban Built-Up', color: '#f97316', preAreaKm2: 184.2, postAreaKm2: 247.6, netChangeKm2: 63.4, percentageChange: 34.4 },
      { className: 'Barren & Bare Soil', color: '#eab308', preAreaKm2: 245.0, postAreaKm2: 196.8, netChangeKm2: -48.2, percentageChange: -19.7 },
      { className: 'Vegetation & Tree Canopy', color: '#10b981', preAreaKm2: 42.6, postAreaKm2: 29.8, netChangeKm2: -12.8, percentageChange: -30.0 },
      { className: 'Water Bodies', color: '#06b6d4', preAreaKm2: 8.4, postAreaKm2: 8.0, netChangeKm2: -0.4, percentageChange: -4.8 },
      { className: 'Agricultural Land', color: '#84cc16', preAreaKm2: 5.0, postAreaKm2: 3.0, netChangeKm2: -2.0, percentageChange: -40.0 }
    ],
    temporalTrend: [
      { year: 2018, urbanAreaKm2: 184.2, vegetationAreaKm2: 42.6 },
      { year: 2019, urbanAreaKm2: 193.1, vegetationAreaKm2: 40.2 },
      { year: 2020, urbanAreaKm2: 204.8, vegetationAreaKm2: 37.9 },
      { year: 2021, urbanAreaKm2: 216.5, vegetationAreaKm2: 35.1 },
      { year: 2022, urbanAreaKm2: 228.0, vegetationAreaKm2: 32.8 },
      { year: 2023, urbanAreaKm2: 238.4, vegetationAreaKm2: 31.0 },
      { year: 2024, urbanAreaKm2: 247.6, vegetationAreaKm2: 29.8 }
    ]
  },
  {
    id: 'job-austin-2019-2024',
    aoiId: 'aoi-austin',
    title: 'Austin East Suburbs & Giga-Corridor Analysis',
    description: 'Farmland and scrubland conversion around Travis County ring road.',
    status: 'completed',
    progressPercent: 100,
    createdAt: '2024-10-18T09:12:00Z',
    completedAt: '2024-10-18T09:18:15Z',
    parameters: {
      modelArch: 'BIT-CD (Bitemporal Trans)',
      confidenceThreshold: 0.80,
      calculateNdvi: true,
      calculateNdbi: true,
      calculateMndwi: false,
      cloudMaskTolerance: 15,
      spatialFilteringRadius: 3
    },
    imagery: {
      preDate: '2019-04-10',
      postDate: '2024-04-15',
      sensor: 'PlanetScope SuperDove',
      resolutionMeters: 3,
      cloudCoveragePre: 2.1,
      cloudCoveragePost: 1.4,
      bandsAvailable: ['Red', 'Green', 'Blue', 'NIR'],
      preImageUrl: 'https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80',
      postImageUrl: 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=1200&q=80',
      changeMaskUrl: 'https://images.unsplash.com/photo-1508739773434-c26b3d09e071?auto=format&fit=crop&w=1200&q=80'
    },
    metrics: {
      totalUrbanPreKm2: 120.4,
      totalUrbanPostKm2: 168.9,
      netExpansionKm2: 48.5,
      expansionPercentage: 40.28,
      annualSprawlRatePercent: 8.05,
      vegetationLossKm2: 32.1,
      waterEncroachmentKm2: 1.2,
      imperviousSurfaceIncreasePercent: 35.6,
      sprawlDispersionIndex: 0.81,
      compactnessRatio: 0.38
    },
    lulcBreakdown: [
      { className: 'Urban Built-Up', color: '#f97316', preAreaKm2: 120.4, postAreaKm2: 168.9, netChangeKm2: 48.5, percentageChange: 40.3 },
      { className: 'Agricultural Land', color: '#84cc16', preAreaKm2: 140.2, postAreaKm2: 101.4, netChangeKm2: -38.8, percentageChange: -27.7 },
      { className: 'Vegetation & Tree Canopy', color: '#10b981', preAreaKm2: 82.5, postAreaKm2: 73.8, netChangeKm2: -8.7, percentageChange: -10.5 },
      { className: 'Water Bodies', color: '#06b6d4', preAreaKm2: 12.0, postAreaKm2: 11.2, netChangeKm2: -0.8, percentageChange: -6.7 },
      { className: 'Barren & Bare Soil', color: '#eab308', preAreaKm2: 7.3, postAreaKm2: 7.1, netChangeKm2: -0.2, percentageChange: -2.7 }
    ],
    temporalTrend: [
      { year: 2019, urbanAreaKm2: 120.4, vegetationAreaKm2: 82.5 },
      { year: 2020, urbanAreaKm2: 128.9, vegetationAreaKm2: 80.2 },
      { year: 2021, urbanAreaKm2: 140.1, vegetationAreaKm2: 78.4 },
      { year: 2022, urbanAreaKm2: 151.7, vegetationAreaKm2: 76.1 },
      { year: 2023, urbanAreaKm2: 160.2, vegetationAreaKm2: 74.9 },
      { year: 2024, urbanAreaKm2: 168.9, vegetationAreaKm2: 73.8 }
    ]
  },
  {
    id: 'job-bengaluru-in-progress',
    aoiId: 'aoi-bengaluru',
    title: 'Bengaluru Southern Peripheral Sprawl Evaluation',
    description: 'Multi-spectral ChangeFormer inference running across 520 km² tile stack.',
    status: 'processing',
    progressPercent: 74,
    currentStep: 'Calculating NDBI spectral difference and morphological closing filters...',
    createdAt: '2024-10-21T18:30:00Z',
    parameters: {
      modelArch: 'ChangeFormer (Transformer)',
      confidenceThreshold: 0.88,
      calculateNdvi: true,
      calculateNdbi: true,
      calculateMndwi: true,
      cloudMaskTolerance: 12,
      spatialFilteringRadius: 4
    },
    imagery: {
      preDate: '2017-02-14',
      postDate: '2023-03-20',
      sensor: 'Sentinel-2 MSI',
      resolutionMeters: 10,
      cloudCoveragePre: 0.9,
      cloudCoveragePost: 1.1,
      bandsAvailable: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
      preImageUrl: 'https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=1200&q=80',
      postImageUrl: 'https://images.unsplash.com/photo-1477959858617-67f30bc75b82?auto=format&fit=crop&w=1200&q=80'
    }
  },
  {
    id: 'job-dubai-queued',
    aoiId: 'aoi-dubai',
    title: 'Dubai South Desert Logistics Hub Growth Rate',
    description: 'Scheduled for nighttime batch processing pipeline on GPU cluster node-04.',
    status: 'queued',
    progressPercent: 0,
    currentStep: 'Waiting in cluster queue (position 2)...',
    createdAt: '2024-10-21T19:10:00Z',
    parameters: {
      modelArch: 'Siam-UNet (Siamese CNN)',
      confidenceThreshold: 0.85,
      calculateNdvi: false,
      calculateNdbi: true,
      calculateMndwi: false,
      cloudMaskTolerance: 5,
      spatialFilteringRadius: 3
    },
    imagery: {
      preDate: '2020-09-01',
      postDate: '2024-09-15',
      sensor: 'Landsat-8 OLI',
      resolutionMeters: 30,
      cloudCoveragePre: 0.2,
      cloudCoveragePost: 0.4,
      bandsAvailable: ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
      preImageUrl: 'https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=1200&q=80',
      postImageUrl: 'https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&w=1200&q=80'
    }
  }
];

export const MOCK_DATASETS: DatasetItem[] = [
  {
    id: 'ds-spacenet-7',
    title: 'SpaceNet 7: Multi-Temporal Urban Development Challenge',
    provider: 'Maxar Technologies & Radiant Earth',
    sensor: 'WorldView-3',
    resolution: '0.5m GSD',
    tileCount: 4280,
    totalSizeGb: 142.5,
    license: 'CC-BY-SA-4.0 (Open Benchmark)',
    region: 'Global (24 Distinct Fast-Growing Cities)',
    spatialExtent: 'Lat [-34.6 to 45.2], Lng [-122.4 to 121.5]',
    spectralBands: ['Red', 'Green', 'Blue', 'Near-Infrared (NIR)'],
    description: 'Gold-standard benchmark tracks monthly high-resolution satellite imagery over 24 months, recording building footprint polygon appearances.',
    downloadUrl: 'https://spacenet.ai/sn7-challenge/',
    lastUpdated: '2024-08-14',
    tags: ['Building Footprints', 'Monthly Time Series', 'Sub-meter', 'Benchmark']
  },
  {
    id: 'ds-dynamicearthnet',
    title: 'DynamicEarthNet: Daily PlanetScope Bi-temporal Change',
    provider: 'European Space Agency (ESA) & Planet Labs',
    sensor: 'PlanetScope SuperDove',
    resolution: '3.0m GSD',
    tileCount: 15400,
    totalSizeGb: 88.0,
    license: 'ESA Scientific Open License',
    region: 'Global 75 Metropolitan AOIs',
    spatialExtent: 'Lat [-40 to 60], Lng [-130 to 140]',
    spectralBands: ['Coastal Blue', 'Blue', 'Green I', 'Green II', 'Red', 'Red Edge', 'NIR'],
    description: 'High cadence multi-spectral observations with pixel-level ground truth annotations for 7 land-use classes.',
    downloadUrl: 'https://mediatum.ub.tum.de/1615019',
    lastUpdated: '2024-05-18',
    tags: ['Daily Cadence', 'Multi-class LULC', 'Commercial Satellite', 'Change Masks']
  },
  {
    id: 'ds-sentinel2-sprawl',
    title: 'Sentinel-2 Urban Sprawl Global Benchmark (S2-USG)',
    provider: 'Copernicus Open Access Hub / UrbanScope Lab',
    sensor: 'Sentinel-2 MSI',
    resolution: '10m - 20m Multi-spectral',
    tileCount: 28400,
    totalSizeGb: 310.2,
    license: 'Open Access Creative Commons',
    region: 'North America, South Asia, Latin America, Europe',
    spatialExtent: 'Global Continental Coverage',
    spectralBands: ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B08 (NIR)', 'B11 (SWIR-1)', 'B12 (SWIR-2)'],
    description: 'Carefully curated bi-temporal pairs spanning 2017 to 2024 with verified urban boundary polygon vectors and NDVI/NDBI index masks.',
    downloadUrl: 'https://copernicus.eu/sentinel-2',
    lastUpdated: '2024-09-30',
    tags: ['Free & Open', 'Cloud-Optimized GeoTIFF', 'SWIR Bands', 'Vegetation & Sprawl']
  },
  {
    id: 'ds-urban3d',
    title: 'Urban3D Challenge Satellite Segmentation Corpus',
    provider: 'US Geospatial Research / IEEE GRSS',
    sensor: 'WorldView-3',
    resolution: '0.3m Ultra-HD',
    tileCount: 1250,
    totalSizeGb: 65.4,
    license: 'Research & Non-Commercial',
    region: 'Jacksonville, Tampa, Omaha, Atlanta',
    spatialExtent: 'Continental US Metros',
    spectralBands: ['RGB', 'Panchromatic', 'DSM (Digital Surface Model)'],
    description: 'Stereo multi-angle imagery paired with LiDAR-derived digital surface models for urban height profile and sprawl boundary detection.',
    downloadUrl: 'https://ieee-dataport.org/urban3d',
    lastUpdated: '2024-02-10',
    tags: ['3D Elevation', 'Stereo Pairs', 'Sub-meter', 'Structural Height']
  }
];

export const MOCK_BENCHMARKS: ModelBenchmark[] = [
  {
    id: 'bm-changeformer',
    modelName: 'ChangeFormer',
    architecture: 'Transformer with Siamese Hierarchical Encoders',
    backbone: 'MiT-B2 Transformer',
    parametersM: 41.2,
    inferenceSpeedMs: 42,
    f1Score: 92.4,
    mIoU: 86.8,
    precision: 93.1,
    recall: 91.7,
    falsePositiveRate: 2.1,
    memoryFootprintMb: 1450
  },
  {
    id: 'bm-bitcd',
    modelName: 'BIT-CD',
    architecture: 'Bitemporal Image Transformer',
    backbone: 'ResNet-50',
    parametersM: 11.5,
    inferenceSpeedMs: 28,
    f1Score: 90.8,
    mIoU: 84.1,
    precision: 91.5,
    recall: 90.2,
    falsePositiveRate: 2.8,
    memoryFootprintMb: 820
  },
  {
    id: 'bm-siamunet',
    modelName: 'Siam-UNet (Concatenation)',
    architecture: 'Dual-branch Siamese Convolutional UNet',
    backbone: 'VGG-16 / ResNet-34',
    parametersM: 26.8,
    inferenceSpeedMs: 34,
    f1Score: 87.2,
    mIoU: 79.5,
    precision: 88.0,
    recall: 86.4,
    falsePositiveRate: 4.6,
    memoryFootprintMb: 1100
  },
  {
    id: 'bm-fcsiamdiff',
    modelName: 'FC-Siam-diff',
    architecture: 'Fully Convolutional Siamese Difference',
    backbone: 'Standard ConvNet',
    parametersM: 1.35,
    inferenceSpeedMs: 14,
    f1Score: 82.0,
    mIoU: 72.4,
    precision: 83.4,
    recall: 80.6,
    falsePositiveRate: 6.9,
    memoryFootprintMb: 350
  }
];

export const MOCK_CONFUSION_MATRIX: ConfusionMatrixClass[] = [
  { actual: 'Urban Built-Up', predicted: 'Urban Built-Up', value: 92.4 },
  { actual: 'Urban Built-Up', predicted: 'Barren Soil', value: 4.8 },
  { actual: 'Urban Built-Up', predicted: 'Vegetation', value: 2.1 },
  { actual: 'Urban Built-Up', predicted: 'Water', value: 0.7 },
  
  { actual: 'Barren Soil', predicted: 'Urban Built-Up', value: 5.2 },
  { actual: 'Barren Soil', predicted: 'Barren Soil', value: 89.1 },
  { actual: 'Barren Soil', predicted: 'Vegetation', value: 4.9 },
  { actual: 'Barren Soil', predicted: 'Water', value: 0.8 },
  
  { actual: 'Vegetation', predicted: 'Urban Built-Up', value: 1.8 },
  { actual: 'Vegetation', predicted: 'Barren Soil', value: 3.2 },
  { actual: 'Vegetation', predicted: 'Vegetation', value: 94.2 },
  { actual: 'Vegetation', predicted: 'Water', value: 0.8 },
  
  { actual: 'Water', predicted: 'Urban Built-Up', value: 0.4 },
  { actual: 'Water', predicted: 'Barren Soil', value: 0.6 },
  { actual: 'Water', predicted: 'Vegetation', value: 1.2 },
  { actual: 'Water', predicted: 'Water', value: 97.8 }
];

export const MOCK_TRAINING_SESSION: TrainingSession = {
  id: 'train-run-408',
  name: 'ChangeFormer-MiT-B2-S2-UrbanSprawl-v2',
  status: 'running',
  currentEpoch: 28,
  totalEpochs: 50,
  bestMIoU: 86.8,
  bestF1Score: 92.4,
  estimatedTimeRemainingSec: 1320,
  config: {
    modelArchitecture: 'ChangeFormer (Transformer)',
    backbone: 'MiT-B2',
    epochs: 50,
    batchSize: 16,
    learningRate: 0.0001,
    optimizer: 'AdamW',
    lossFunction: 'Combined (Focal + Dice)',
    weightDecay: 0.01,
    validationSplit: 0.2,
    useDataAugmentation: true,
    cropSize: 512
  },
  history: [
    { epoch: 1, trainLoss: 0.684, valLoss: 0.641, mIoU: 52.4, f1Score: 61.2, learningRate: 0.0001, timeSec: 58 },
    { epoch: 4, trainLoss: 0.512, valLoss: 0.490, mIoU: 64.1, f1Score: 71.8, learningRate: 0.0001, timeSec: 57 },
    { epoch: 8, trainLoss: 0.410, valLoss: 0.395, mIoU: 72.8, f1Score: 79.5, learningRate: 0.0001, timeSec: 58 },
    { epoch: 12, trainLoss: 0.325, valLoss: 0.312, mIoU: 78.4, f1Score: 84.6, learningRate: 0.0001, timeSec: 56 },
    { epoch: 16, trainLoss: 0.264, valLoss: 0.258, mIoU: 81.9, f1Score: 87.8, learningRate: 0.00008, timeSec: 57 },
    { epoch: 20, trainLoss: 0.221, valLoss: 0.219, mIoU: 83.7, f1Score: 89.9, learningRate: 0.00006, timeSec: 59 },
    { epoch: 24, trainLoss: 0.188, valLoss: 0.191, mIoU: 85.2, f1Score: 91.2, learningRate: 0.00004, timeSec: 58 },
    { epoch: 28, trainLoss: 0.165, valLoss: 0.174, mIoU: 86.8, f1Score: 92.4, learningRate: 0.00003, timeSec: 57 }
  ]
};
