import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { AreaOfInterest } from '../../types';
import { CoordinateBar } from './CoordinateBar';
import { MapLayerControls, BaseMapLayer, OverlayToggles } from './MapLayerControls';
import { Maximize2, Minimize2 } from 'lucide-react';

// Fix for default Leaflet icon paths
const customIcon = L.divIcon({
  className: 'custom-gis-pin',
  html: `<div style="
    width: 24px;
    height: 24px;
    background: #06b6d4;
    border: 3px solid #ffffff;
    border-radius: 50%;
    box-shadow: 0 0 15px #06b6d4;
    display: flex;
    align-items: center;
    justify-content: center;
  "><div style="width: 6px; height: 6px; background: #080c14; border-radius: 50%;"></div></div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12]
});

export interface GisMapProps {
  aois: AreaOfInterest[];
  selectedAoi?: AreaOfInterest;
  onSelectAoi?: (aoi: AreaOfInterest) => void;
  className?: string;
  height?: string;
}

export const GisMap: React.FC<GisMapProps> = ({
  aois,
  selectedAoi,
  onSelectAoi,
  className = '',
  height = '500px'
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const baseLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const overlayGroupRef = useRef<L.LayerGroup | null>(null);

  const [mouseCoords, setMouseCoords] = useState<{ lat: number; lng: number }>({
    lat: selectedAoi ? selectedAoi.center[0] : 36.084,
    lng: selectedAoi ? selectedAoi.center[1] : -115.25
  });
  const [currentZoom, setCurrentZoom] = useState<number>(selectedAoi ? selectedAoi.zoom : 11);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [baseMap, setBaseMap] = useState<BaseMapLayer>('dark');
  const [overlays, setOverlays] = useState<OverlayToggles>({
    sprawlMask: true,
    vegetationNdvi: true,
    aoiBoundaries: true,
    heatMap: false
  });
  const [maskOpacity, setMaskOpacity] = useState<number>(0.75);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const initialCenter: L.LatLngTuple = selectedAoi ? [selectedAoi.center[0], selectedAoi.center[1]] : [36.084, -115.25];
      const initialZoom = selectedAoi ? selectedAoi.zoom : 11;

      const map = L.map(mapContainerRef.current, {
        center: initialCenter,
        zoom: initialZoom,
        zoomControl: true,
        attributionControl: true
      });

      mapInstanceRef.current = map;
      baseLayerGroupRef.current = L.layerGroup().addTo(map);
      overlayGroupRef.current = L.layerGroup().addTo(map);

      map.on('mousemove', (e: L.LeafletMouseEvent) => {
        setMouseCoords({ lat: e.latlng.lat, lng: e.latlng.lng });
      });

      map.on('zoomend', () => {
        setCurrentZoom(map.getZoom());
      });
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Handle Base Map Switching
  useEffect(() => {
    if (!mapInstanceRef.current || !baseLayerGroupRef.current) return;

    baseLayerGroupRef.current.clearLayers();

    let tileUrl = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
    let attribution = '&copy; <a href="https://carto.com/">CARTO</a>';

    if (baseMap === 'satellite') {
      tileUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
      attribution = 'Tiles &copy; Esri, Maxar, Earthstar Geographics';
    } else if (baseMap === 'street') {
      tileUrl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
      attribution = '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a>';
    }

    const tileLayer = L.tileLayer(tileUrl, {
      maxZoom: 19,
      attribution
    });

    baseLayerGroupRef.current.addLayer(tileLayer);
  }, [baseMap]);

  // Handle AOIs and Overlays
  useEffect(() => {
    if (!mapInstanceRef.current || !overlayGroupRef.current) return;

    overlayGroupRef.current.clearLayers();

    aois.forEach(aoi => {
      // Bounding box rectangle
      if (overlays.aoiBoundaries) {
        const bounds: L.LatLngBoundsExpression = [
          [aoi.bbox.minLat, aoi.bbox.minLng],
          [aoi.bbox.maxLat, aoi.bbox.maxLng]
        ];

        const isSelected = selectedAoi?.id === aoi.id;
        const rect = L.rectangle(bounds, {
          color: isSelected ? '#06b6d4' : '#f97316',
          weight: isSelected ? 3 : 2,
          fillOpacity: 0.1,
          dashArray: isSelected ? undefined : '5, 5'
        });

        rect.on('click', () => {
          if (onSelectAoi) onSelectAoi(aoi);
        });

        overlayGroupRef.current?.addLayer(rect);
      }

      // Marker
      const marker = L.marker(aoi.center, { icon: customIcon });
      
      const popupHtml = `
        <div style="font-family: inherit; color: #0f172a; min-width: 180px;">
          <h4 style="font-weight: 700; font-size: 13px; margin: 0 0 4px 0; color: #080c14;">${aoi.name}</h4>
          <p style="font-size: 11px; margin: 0 0 6px 0; color: #475569;">${aoi.city}, ${aoi.country}</p>
          <div style="font-size: 10px; background: #f1f5f9; padding: 6px; border-radius: 4px; margin-bottom: 8px;">
            <div>Sprawl Rate: <strong>${aoi.urbanGrowthRateAnnual}% / yr</strong></div>
            <div>Risk: <strong style="color: ${aoi.riskLevel === 'Severe' ? '#dc2626' : '#d97706'}">${aoi.riskLevel}</strong></div>
          </div>
          <a href="/analysis/results/job-vegas-2018-2024" style="
            display: block;
            text-align: center;
            background: #0891b2;
            color: #ffffff;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
            text-decoration: none;
          ">View Sprawl Audit</a>
        </div>
      `;

      marker.bindPopup(popupHtml);
      marker.on('click', () => {
        if (onSelectAoi) onSelectAoi(aoi);
      });

      overlayGroupRef.current?.addLayer(marker);

      // Simulated Sprawl / NDVI raster overlay circle
      if (overlays.sprawlMask) {
        const sprawlCircle = L.circle(aoi.center, {
          color: '#f97316',
          fillColor: '#f97316',
          fillOpacity: maskOpacity * 0.45,
          radius: 4200
        });
        overlayGroupRef.current?.addLayer(sprawlCircle);
      }

      if (overlays.vegetationNdvi) {
        const ndviCircle = L.circle([aoi.center[0] - 0.04, aoi.center[1] + 0.04], {
          color: '#10b981',
          fillColor: '#10b981',
          fillOpacity: maskOpacity * 0.35,
          radius: 3500
        });
        overlayGroupRef.current?.addLayer(ndviCircle);
      }
    });
  }, [aois, selectedAoi, overlays, maskOpacity, onSelectAoi]);

  // Center on selected AOI
  useEffect(() => {
    if (mapInstanceRef.current && selectedAoi) {
      mapInstanceRef.current.flyTo([selectedAoi.center[0], selectedAoi.center[1]], selectedAoi.zoom, {
        duration: 1.2
      });
    }
  }, [selectedAoi]);

  return (
    <div className={`relative rounded-2xl overflow-hidden border border-slate-800 bg-[#080c14] ${className} ${isFullscreen ? 'fixed inset-0 z-50 rounded-none' : ''}`} style={{ height: isFullscreen ? '100vh' : height }}>
      {/* Map container */}
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Floating GIS Layer Controls */}
      <div className="absolute top-4 right-4 z-[400]">
        <MapLayerControls
          activeBaseMap={baseMap}
          onBaseMapChange={setBaseMap}
          overlays={overlays}
          onOverlayToggle={(key) => setOverlays(prev => ({ ...prev, [key]: !prev[key] }))}
          opacity={maskOpacity}
          onOpacityChange={setMaskOpacity}
        />
      </div>

      {/* Fullscreen Button */}
      <button
        onClick={() => setIsFullscreen(!isFullscreen)}
        className="absolute top-4 left-14 z-[400] p-2 rounded-lg bg-[#0d1322]/90 border border-slate-700/80 text-slate-300 hover:text-white hover:bg-slate-800 shadow-xl backdrop-blur-md transition-colors"
        title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Map'}
      >
        {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
      </button>

      {/* Bottom coordinate bar */}
      <div className="absolute bottom-4 left-4 right-4 z-[400]">
        <CoordinateBar
          lat={mouseCoords.lat}
          lng={mouseCoords.lng}
          zoom={currentZoom}
          resolutionMeters={10}
        />
      </div>
    </div>
  );
};
