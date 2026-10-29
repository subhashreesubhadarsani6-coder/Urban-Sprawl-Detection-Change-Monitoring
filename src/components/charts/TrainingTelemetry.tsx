import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { TrainingTelemetryPoint } from '../../types';

export interface TrainingTelemetryProps {
  data: TrainingTelemetryPoint[];
}

export const TrainingTelemetry: React.FC<TrainingTelemetryProps> = ({ data }) => {
  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293f" vertical={false} />
          <XAxis
            dataKey="epoch"
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickLine={false}
            label={{ value: 'Epoch', position: 'insideBottom', offset: -4, fill: '#64748b', fontSize: 10 }}
          />
          {/* Left Y-Axis: Loss */}
          <YAxis
            yAxisId="left"
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickLine={false}
            domain={[0, 'auto']}
          />
          {/* Right Y-Axis: Accuracy metrics (0 - 100%) */}
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickLine={false}
            domain={[40, 100]}
            unit="%"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0d1322',
              borderColor: '#1e293f',
              borderRadius: '8px',
              fontSize: '12px',
              color: '#f8fafc'
            }}
          />
          <Legend
            verticalAlign="top"
            align="right"
            wrapperStyle={{ paddingBottom: '12px', fontSize: '11px' }}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="trainLoss"
            name="Train Loss"
            stroke="#f43f5e"
            strokeWidth={2}
            dot={false}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="valLoss"
            name="Val Loss"
            stroke="#f59e0b"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={false}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="mIoU"
            name="mIoU (%)"
            stroke="#06b6d4"
            strokeWidth={2.5}
            dot={{ r: 3, fill: '#06b6d4' }}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="f1Score"
            name="F1-Score (%)"
            stroke="#10b981"
            strokeWidth={2}
            dot={{ r: 2, fill: '#10b981' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
