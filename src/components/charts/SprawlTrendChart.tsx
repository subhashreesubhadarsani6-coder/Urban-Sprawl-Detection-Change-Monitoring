import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

interface TrendPoint {
  year: number;
  urbanAreaKm2: number;
  vegetationAreaKm2: number;
}

interface SprawlTrendChartProps {
  data: TrendPoint[];
  title?: string;
}

export const SprawlTrendChart: React.FC<SprawlTrendChartProps> = ({
  data,
  title = 'Temporal Sprawl & Vegetation Trend (2018 - 2024)'
}) => {
  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-semibold text-slate-200 tracking-wide">{title}</h4>
        <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-500/20">
          Metric: Area (km²)
        </span>
      </div>

      <div className="flex-1 w-full min-h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="urbanGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f97316" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#f97316" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="vegGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293f" vertical={false} />
            <XAxis
              dataKey="year"
              stroke="#64748b"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              tickLine={false}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0d1322',
                borderColor: '#1e293f',
                borderRadius: '8px',
                fontSize: '12px',
                color: '#f8fafc'
              }}
              formatter={(value: any, name: any) => [
                `${value} km²`,
                name === 'urbanAreaKm2' ? 'Built-Up Urban' : 'Vegetation & Canopy'
              ]}
            />
            <Legend
              verticalAlign="top"
              align="right"
              iconType="circle"
              wrapperStyle={{ paddingBottom: '12px', fontSize: '11px' }}
              formatter={(value) => (value === 'urbanAreaKm2' ? 'Urban Built-Up' : 'Vegetation Canopy')}
            />
            <Area
              type="monotone"
              dataKey="urbanAreaKm2"
              stroke="#f97316"
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#urbanGrad)"
            />
            <Area
              type="monotone"
              dataKey="vegetationAreaKm2"
              stroke="#10b981"
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#vegGrad)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
