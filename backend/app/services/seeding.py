import logging
from ..database import SessionLocal, Base, engine
from ..models import (
    AreaOfInterest,
    AnalysisJob,
    DatasetItem,
    TrainingSession,
    ModelBenchmark,
    ConfusionMatrixEntry
)

logger = logging.getLogger("seeding")

def seed_database():
    """Initializes tables and seeds default records if database is empty."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Ensure default baseline artifacts exist on disk
        from .storage import generate_job_artifacts
        generate_job_artifacts("job-vegas-2018-2024", "Las Vegas Valley", {"netExpansionKm2": 63.4, "annualSprawlRatePercent": 5.12, "vegetationLossKm2": 12.8})
        generate_job_artifacts("job-austin-2019-2024", "Austin Metropolitan Area", {"netExpansionKm2": 48.5, "annualSprawlRatePercent": 8.05, "vegetationLossKm2": 32.1})

        # Check if already seeded
        if db.query(AreaOfInterest).first():
            logger.info("Database already initialized and seeded.")
            return

        logger.info("Seeding initial database records...")

        # 1. Areas of Interest
        aois = [
            AreaOfInterest(
                id="aoi-las-vegas",
                name="Las Vegas Valley Southwest Expansion",
                city="Las Vegas, NV",
                country="United States",
                center_lat=36.084,
                center_lng=-115.25,
                zoom=12,
                bbox={"minLat": 35.98, "minLng": -115.35, "maxLat": 36.18, "maxLng": -115.15},
                monitored_since="2018-05-10",
                total_area_km2=485.2,
                urban_growth_rate_annual=4.8,
                risk_level="Severe",
                thumbnail_url="https://images.unsplash.com/photo-1581351123004-757df051db8e?auto=format&fit=crop&w=600&q=80"
            ),
            AreaOfInterest(
                id="aoi-austin",
                name="Austin East Tech Corridor & Suburban Fringe",
                city="Austin, TX",
                country="United States",
                center_lat=30.267,
                center_lng=-97.743,
                zoom=12,
                bbox={"minLat": 30.15, "minLng": -97.85, "maxLat": 30.40, "maxLng": -97.60},
                monitored_since="2019-03-15",
                total_area_km2=362.4,
                urban_growth_rate_annual=5.6,
                risk_level="High",
                thumbnail_url="https://images.unsplash.com/photo-1531218150217-54595bc2b934?auto=format&fit=crop&w=600&q=80"
            ),
            AreaOfInterest(
                id="aoi-bengaluru",
                name="Bengaluru Southern Peripheral Corridor",
                city="Bengaluru",
                country="India",
                center_lat=12.925,
                center_lng=77.625,
                zoom=12,
                bbox={"minLat": 12.82, "minLng": 77.52, "maxLat": 13.02, "maxLng": 77.75},
                monitored_since="2017-01-20",
                total_area_km2=520.8,
                urban_growth_rate_annual=6.2,
                risk_level="Severe",
                thumbnail_url="https://images.unsplash.com/photo-1596176530529-78163a4f7af2?auto=format&fit=crop&w=600&q=80"
            ),
            AreaOfInterest(
                id="aoi-dubai",
                name="Dubai South & Al Maktoum Logistics Zone",
                city="Dubai",
                country="United Arab Emirates",
                center_lat=24.896,
                center_lng=55.161,
                zoom=11,
                bbox={"minLat": 24.78, "minLng": 55.05, "maxLat": 25.02, "maxLng": 55.28},
                monitored_since="2020-09-01",
                total_area_km2=410.5,
                urban_growth_rate_annual=3.9,
                risk_level="Moderate",
                thumbnail_url="https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&w=600&q=80"
            )
        ]
        db.add_all(aois)

        # 2. Initial Analysis Jobs
        jobs = [
            AnalysisJob(
                id="job-vegas-2018-2024",
                aoi_id="aoi-las-vegas",
                title="Las Vegas Valley Bi-temporal Sprawl Audit (2018 - 2024)",
                description="Bitemporal change detection measuring desert conversion to residential subdivisions and commercial logistics hubs.",
                status="completed",
                progress_percent=100,
                current_step="Analysis complete. Sprawl polygons and LULC matrix generated.",
                created_at="2024-10-12T14:20:00Z",
                completed_at="2024-10-12T14:26:42Z",
                parameters={
                    "modelArch": "Siamese U-Net (FC-Siam-diff)",
                    "confidenceThreshold": 0.50,
                    "calculateNdvi": True,
                    "calculateNdbi": True,
                    "calculateMndwi": True,
                    "cloudMaskTolerance": 10,
                    "spatialFilteringRadius": 3,
                    "isRealInference": True
                },
                imagery={
                    "preDate": "2018-06-15",
                    "postDate": "2024-06-20",
                    "sensor": "Sentinel-2 MSI",
                    "resolutionMeters": 10,
                    "cloudCoveragePre": 1.2,
                    "cloudCoveragePost": 0.8,
                    "bandsAvailable": ["B02", "B03", "B04", "B08", "B11", "B12"],
                    "preImageUrl": "/api/v1/storage/uploads/sample_t1_lasvegas.png",
                    "postImageUrl": "/api/v1/storage/uploads/sample_t2_lasvegas.png",
                    "changeMaskUrl": "/api/v1/artifacts/job-vegas-2018-2024/change_overlay.png",
                    "overlayUrl": "/api/v1/artifacts/job-vegas-2018-2024/change_overlay.png",
                    "probabilityMapUrl": "/api/v1/artifacts/job-vegas-2018-2024/probability_map.png",
                    "geoTiffUrl": "/api/v1/artifacts/job-vegas-2018-2024/change_mask.tif",
                    "geojsonUrl": "/api/v1/artifacts/job-vegas-2018-2024/sprawl_boundaries.geojson",
                    "auditReportUrl": "/api/v1/artifacts/job-vegas-2018-2024/audit_report.txt"
                },
                metrics={
                    "totalUrbanPreKm2": 52.93,
                    "totalUrbanPostKm2": 58.99,
                    "netExpansionKm2": 6.06,
                    "expansionPercentage": 10.27,
                    "annualSprawlRatePercent": 2.05,
                    "vegetationLossKm2": 3.94,
                    "waterEncroachmentKm2": 0.0,
                    "imperviousSurfaceIncreasePercent": 15.4,
                    "sprawlDispersionIndex": 0.72,
                    "compactnessRatio": 0.43,
                    "changedPixels": 60613,
                    "validPixels": 589984,
                    "totalPixels": 589984,
                    "spatialResolution": {
                        "pixel_width_m": 10.0,
                        "pixel_height_m": 10.0,
                        "pixel_area_m2": 100.0,
                        "is_calibrated": False,
                        "resolution_source": "Sentinel-2 MSI (Estimated 10m GSD)"
                    },
                    "modelArch": "Siamese U-Net (FC-Siam-diff)",
                    "inferenceType": "Real Model Inference (FC-Siam-diff)",
                    "checkpointUsed": "checkpoints/best_checkpoint.pth",
                    "appliedThreshold": 0.50
                },
                lulc_breakdown=[
                    {"className": "Urban Built-Up", "color": "#f97316", "preAreaKm2": 184.2, "postAreaKm2": 247.6, "netChangeKm2": 63.4, "percentageChange": 34.4},
                    {"className": "Barren & Bare Soil", "color": "#eab308", "preAreaKm2": 245.0, "postAreaKm2": 196.8, "netChangeKm2": -48.2, "percentageChange": -19.7},
                    {"className": "Vegetation & Tree Canopy", "color": "#10b981", "preAreaKm2": 42.6, "postAreaKm2": 29.8, "netChangeKm2": -12.8, "percentageChange": -30.0},
                    {"className": "Water Bodies", "color": "#06b6d4", "preAreaKm2": 8.4, "postAreaKm2": 8.0, "netChangeKm2": -0.4, "percentageChange": -4.8},
                    {"className": "Agricultural Land", "color": "#84cc16", "preAreaKm2": 5.0, "postAreaKm2": 3.0, "netChangeKm2": -2.0, "percentageChange": -40.0}
                ],
                temporal_trend=[
                    {"year": 2018, "urbanAreaKm2": 184.2, "vegetationAreaKm2": 42.6},
                    {"year": 2019, "urbanAreaKm2": 193.1, "vegetationAreaKm2": 40.2},
                    {"year": 2020, "urbanAreaKm2": 204.8, "vegetationAreaKm2": 37.9},
                    {"year": 2021, "urbanAreaKm2": 216.5, "vegetationAreaKm2": 35.1},
                    {"year": 2022, "urbanAreaKm2": 228.0, "vegetationAreaKm2": 32.8},
                    {"year": 2023, "urbanAreaKm2": 238.4, "vegetationAreaKm2": 31.0},
                    {"year": 2024, "urbanAreaKm2": 247.6, "vegetationAreaKm2": 29.8}
                ]
            ),
            AnalysisJob(
                id="job-austin-2019-2024",
                aoi_id="aoi-austin",
                title="Austin East Suburbs & Giga-Corridor Analysis",
                description="Farmland and scrubland conversion around Travis County ring road.",
                status="completed",
                progress_percent=100,
                current_step="Analysis complete. Sprawl polygons and LULC matrix generated.",
                created_at="2024-10-18T09:12:00Z",
                completed_at="2024-10-18T09:18:15Z",
                parameters={
                    "modelArch": "BIT-CD (Bitemporal Trans)",
                    "confidenceThreshold": 0.80,
                    "calculateNdvi": True,
                    "calculateNdbi": True,
                    "calculateMndwi": False,
                    "cloudMaskTolerance": 15,
                    "spatialFilteringRadius": 3
                },
                imagery={
                    "preDate": "2019-04-10",
                    "postDate": "2024-04-15",
                    "sensor": "PlanetScope SuperDove",
                    "resolutionMeters": 3,
                    "cloudCoveragePre": 2.1,
                    "cloudCoveragePost": 1.4,
                    "bandsAvailable": ["Red", "Green", "Blue", "NIR"],
                    "preImageUrl": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80",
                    "postImageUrl": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=1200&q=80",
                    "changeMaskUrl": "https://images.unsplash.com/photo-1508739773434-c26b3d09e071?auto=format&fit=crop&w=1200&q=80"
                },
                metrics={
                    "totalUrbanPreKm2": 120.4,
                    "totalUrbanPostKm2": 168.9,
                    "netExpansionKm2": 48.5,
                    "expansionPercentage": 40.28,
                    "annualSprawlRatePercent": 8.05,
                    "vegetationLossKm2": 32.1,
                    "waterEncroachmentKm2": 1.2,
                    "imperviousSurfaceIncreasePercent": 35.6,
                    "sprawlDispersionIndex": 0.81,
                    "compactnessRatio": 0.38
                },
                lulc_breakdown=[
                    {"className": "Urban Built-Up", "color": "#f97316", "preAreaKm2": 120.4, "postAreaKm2": 168.9, "netChangeKm2": 48.5, "percentageChange": 40.3},
                    {"className": "Agricultural Land", "color": "#84cc16", "preAreaKm2": 140.2, "postAreaKm2": 101.4, "netChangeKm2": -38.8, "percentageChange": -27.7},
                    {"className": "Vegetation & Tree Canopy", "color": "#10b981", "preAreaKm2": 82.5, "postAreaKm2": 73.8, "netChangeKm2": -8.7, "percentageChange": -10.5},
                    {"className": "Water Bodies", "color": "#06b6d4", "preAreaKm2": 12.0, "postAreaKm2": 11.2, "netChangeKm2": -0.8, "percentageChange": -6.7},
                    {"className": "Barren & Bare Soil", "color": "#eab308", "preAreaKm2": 7.3, "postAreaKm2": 7.1, "netChangeKm2": -0.2, "percentageChange": -2.7}
                ],
                temporal_trend=[
                    {"year": 2019, "urbanAreaKm2": 120.4, "vegetationAreaKm2": 82.5},
                    {"year": 2020, "urbanAreaKm2": 128.9, "vegetationAreaKm2": 80.2},
                    {"year": 2021, "urbanAreaKm2": 140.1, "vegetationAreaKm2": 78.4},
                    {"year": 2022, "urbanAreaKm2": 151.7, "vegetationAreaKm2": 76.1},
                    {"year": 2023, "urbanAreaKm2": 160.2, "vegetationAreaKm2": 74.9},
                    {"year": 2024, "urbanAreaKm2": 168.9, "vegetationAreaKm2": 73.8}
                ]
            )
        ]
        db.add_all(jobs)

        # 3. Datasets
        datasets = [
            DatasetItem(
                id="ds-spacenet-7",
                title="SpaceNet 7: Multi-Temporal Urban Development Challenge",
                provider="Maxar Technologies & Radiant Earth",
                sensor="WorldView-3",
                resolution="0.5m GSD",
                tile_count=4280,
                total_size_gb=142.5,
                license="CC-BY-SA-4.0 (Open Benchmark)",
                region="Global (24 Distinct Fast-Growing Cities)",
                spatial_extent="Lat [-34.6 to 45.2], Lng [-122.4 to 121.5]",
                spectral_bands=["Red", "Green", "Blue", "Near-Infrared (NIR)"],
                description="Gold-standard benchmark tracks monthly high-resolution satellite imagery over 24 months, recording building footprint polygon appearances.",
                download_url="https://spacenet.ai/sn7-challenge/",
                last_updated="2024-08-14",
                tags=["Building Footprints", "Monthly Time Series", "Sub-meter", "Benchmark"]
            ),
            DatasetItem(
                id="ds-dynamicearthnet",
                title="DynamicEarthNet: Daily PlanetScope Bi-temporal Change",
                provider="European Space Agency (ESA) & Planet Labs",
                sensor="PlanetScope SuperDove",
                resolution="3.0m GSD",
                tile_count=15400,
                total_size_gb=88.0,
                license="ESA Scientific Open License",
                region="Global 75 Metropolitan AOIs",
                spatial_extent="Lat [-40 to 60], Lng [-130 to 140]",
                spectral_bands=["Coastal Blue", "Blue", "Green I", "Green II", "Red", "Red Edge", "NIR"],
                description="High cadence multi-spectral observations with pixel-level ground truth annotations for 7 land-use classes.",
                download_url="https://mediatum.ub.tum.de/1615019",
                last_updated="2024-05-18",
                tags=["Daily Cadence", "Multi-class LULC", "Commercial Satellite", "Change Masks"]
            ),
            DatasetItem(
                id="ds-sentinel2-sprawl",
                title="Sentinel-2 Urban Sprawl Global Benchmark (S2-USG)",
                provider="Copernicus Open Access Hub / UrbanScope Lab",
                sensor="Sentinel-2 MSI",
                resolution="10m - 20m Multi-spectral",
                tile_count=28400,
                total_size_gb=310.2,
                license="Open Access Creative Commons",
                region="North America, South Asia, Latin America, Europe",
                spatial_extent="Global Continental Coverage",
                spectral_bands=["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)", "B11 (SWIR-1)", "B12 (SWIR-2)"],
                description="Carefully curated bi-temporal pairs spanning 2017 to 2024 with verified urban boundary polygon vectors and NDVI/NDBI index masks.",
                download_url="https://copernicus.eu/sentinel-2",
                last_updated="2024-09-30",
                tags=["Free & Open", "Cloud-Optimized GeoTIFF", "SWIR Bands", "Vegetation & Sprawl"]
            ),
            DatasetItem(
                id="ds-urban3d",
                title="Urban3D Challenge Satellite Segmentation Corpus",
                provider="US Geospatial Research / IEEE GRSS",
                sensor="WorldView-3",
                resolution="0.3m Ultra-HD",
                tile_count=1250,
                total_size_gb=65.4,
                license="Research & Non-Commercial",
                region="Jacksonville, Tampa, Omaha, Atlanta",
                spatial_extent="Continental US Metros",
                spectral_bands=["RGB", "Panchromatic", "DSM (Digital Surface Model)"],
                description="Stereo multi-angle imagery paired with LiDAR-derived digital surface models for urban height profile and sprawl boundary detection.",
                download_url="https://ieee-dataport.org/urban3d",
                last_updated="2024-02-10",
                tags=["3D Elevation", "Stereo Pairs", "Sub-meter", "Structural Height"]
            )
        ]
        db.add_all(datasets)

        # 4. Model Benchmarks
        benchmarks = [
            ModelBenchmark(
                id="bm-changeformer",
                model_name="ChangeFormer",
                architecture="Transformer with Siamese Hierarchical Encoders",
                backbone="MiT-B2 Transformer",
                parameters_m=41.2,
                inference_speed_ms=42,
                f1_score=92.4,
                miou=86.8,
                precision=93.1,
                recall=91.7,
                false_positive_rate=2.1,
                memory_footprint_mb=1450
            ),
            ModelBenchmark(
                id="bm-bitcd",
                model_name="BIT-CD",
                architecture="Bitemporal Image Transformer",
                backbone="ResNet-50",
                parameters_m=11.5,
                inference_speed_ms=28,
                f1_score=90.8,
                miou=84.1,
                precision=91.5,
                recall=90.2,
                false_positive_rate=2.8,
                memory_footprint_mb=820
            ),
            ModelBenchmark(
                id="bm-siamunet",
                model_name="Siam-UNet (Concatenation)",
                architecture="Dual-branch Siamese Convolutional UNet",
                backbone="VGG-16 / ResNet-34",
                parameters_m=26.8,
                inference_speed_ms=34,
                f1_score=87.2,
                miou=79.5,
                precision=88.0,
                recall=86.4,
                false_positive_rate=4.6,
                memory_footprint_mb=1100
            ),
            ModelBenchmark(
                id="bm-fcsiamdiff",
                model_name="FC-Siam-diff",
                architecture="Fully Convolutional Siamese Difference",
                backbone="Standard ConvNet",
                parameters_m=1.35,
                inference_speed_ms=14,
                f1_score=82.0,
                miou=72.4,
                precision=83.4,
                recall=80.6,
                false_positive_rate=6.9,
                memory_footprint_mb=350
            )
        ]
        db.add_all(benchmarks)

        # 5. Confusion Matrix
        cm_data = [
            ("Urban Built-Up", "Urban Built-Up", 92.4),
            ("Urban Built-Up", "Barren Soil", 4.8),
            ("Urban Built-Up", "Vegetation", 2.1),
            ("Urban Built-Up", "Water", 0.7),
            ("Barren Soil", "Urban Built-Up", 5.2),
            ("Barren Soil", "Barren Soil", 89.1),
            ("Barren Soil", "Vegetation", 4.9),
            ("Barren Soil", "Water", 0.8),
            ("Vegetation", "Urban Built-Up", 1.8),
            ("Vegetation", "Barren Soil", 3.2),
            ("Vegetation", "Vegetation", 94.2),
            ("Vegetation", "Water", 0.8),
            ("Water", "Urban Built-Up", 0.4),
            ("Water", "Barren Soil", 0.6),
            ("Water", "Vegetation", 1.2),
            ("Water", "Water", 97.8)
        ]
        db.add_all([ConfusionMatrixEntry(actual=a, predicted=p, value=v) for a, p, v in cm_data])

        # 6. Training Session
        training = TrainingSession(
            id="train-run-408",
            name="ChangeFormer-MiT-B2-S2-UrbanSprawl-v2",
            status="running",
            current_epoch=28,
            total_epochs=50,
            best_miou=86.8,
            best_f1_score=92.4,
            estimated_time_remaining_sec=1320,
            config={
                "modelArchitecture": "ChangeFormer (Transformer)",
                "backbone": "MiT-B2",
                "epochs": 50,
                "batchSize": 16,
                "learningRate": 0.0001,
                "optimizer": "AdamW",
                "lossFunction": "Combined (Focal + Dice)",
                "weightDecay": 0.01,
                "validationSplit": 0.2,
                "useDataAugmentation": True,
                "cropSize": 512
            },
            history=[
                {"epoch": 1, "trainLoss": 0.684, "valLoss": 0.641, "mIoU": 52.4, "f1Score": 61.2, "learningRate": 0.0001, "timeSec": 58},
                {"epoch": 4, "trainLoss": 0.512, "valLoss": 0.490, "mIoU": 64.1, "f1Score": 71.8, "learningRate": 0.0001, "timeSec": 57},
                {"epoch": 8, "trainLoss": 0.410, "valLoss": 0.395, "mIoU": 72.8, "f1Score": 79.5, "learningRate": 0.0001, "timeSec": 58},
                {"epoch": 12, "trainLoss": 0.325, "valLoss": 0.312, "mIoU": 78.4, "f1Score": 84.6, "learningRate": 0.0001, "timeSec": 56},
                {"epoch": 16, "trainLoss": 0.264, "valLoss": 0.258, "mIoU": 81.9, "f1Score": 87.8, "learningRate": 0.00008, "timeSec": 57},
                {"epoch": 20, "trainLoss": 0.221, "valLoss": 0.219, "mIoU": 83.7, "f1Score": 89.9, "learningRate": 0.00006, "timeSec": 59},
                {"epoch": 24, "trainLoss": 0.188, "valLoss": 0.191, "mIoU": 85.2, "f1Score": 91.2, "learningRate": 0.00004, "timeSec": 58},
                {"epoch": 28, "trainLoss": 0.165, "valLoss": 0.174, "mIoU": 86.8, "f1Score": 92.4, "learningRate": 0.00003, "timeSec": 57}
            ]
        )
        db.add(training)

        db.commit()
        logger.info("Database seeding completed successfully.")
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()
