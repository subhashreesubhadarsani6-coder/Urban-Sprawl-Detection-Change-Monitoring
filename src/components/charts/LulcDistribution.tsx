import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { LulcClassBreakdown } from '../../types';
import { ArrowUpRight, ArrowDownRight } from 'lucide-react';

export interface LulcDistributionProps {
  data: LulcClassBreakdown[];
}

export const LulcDistribution: React.FC<LulcDistributionProps> = ({ data }) => {
  const chartData = data.map(item => ({
    name: item.className.split(' ')[0], // short name
    fullName: item.className,
    'Pre-Period Area': item.preAreaKm2,
    'Post-Period Area': item.postAreaKm2,
    netChange: item.netChangeKm2,
    percentage: item.percentageChange,
    color: item.color
  }));

  return (
    <div className="space-y-5">
      {/* Chart */}
      <div className="w-full h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293f" vertical={false} />
            <XAxis
              dataKey="name"
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
              formatter={(val: any) => [`${val} km²`, '']}
            />
            <Legend
              verticalAlign="top"
              align="right"
              wrapperStyle={{ paddingBottom: '10px', fontSize: '11px' }}
            />
            <Bar dataKey="Pre-Period Area" fill="#64748b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Post-Period Area" fill="#06b6d4" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Class change stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5">
        {data.map((item) => {
          const isGain = item.netChangeKm2 >= 0;
          return (
            <div
              key={item.className}
              className="p-3 rounded-xl bg-[#090d18] border border-slate-800/80 flex flex-col justify-between"
            >
              <div className="flex items-center gap-1.5 mb-1.5">
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
                <span className="text-[11px] font-medium text-slate-300 truncate" title={item.className}>
                  {item.className}
                </span>
              </div>
              <div>
                <div className="text-base font-bold text-white font-mono">
                  {item.postAreaKm2.toFixed(1)} <span className="text-[10px] text-slate-500 font-normal">km²</span>
                </div>
                <div className={`flex items-center gap-1 text-[11px] font-semibold mt-0.5 ${isGain ? 'text-orange-400' : 'text-emerald-400'}`}>
                  {isGain ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                  <span>{isGain ? '+' : ''}{item.netChangeKm2.toFixed(1)} km² ({item.percentageChange > 0 ? '+' : ''}{item.percentageChange}%)</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
