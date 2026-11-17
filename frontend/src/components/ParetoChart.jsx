import React from 'react';
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';

function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-[#0B1020] border border-slate-700 rounded-xl p-3 text-xs shadow-xl space-y-1">
      <p className="font-mono font-semibold text-white">{d.name}</p>
      <p className="text-slate-400">
        Gen {d.generation} · {d.origin} {d.is_pareto_optimal ? '· Pareto Frontier ★' : ''}
      </p>
      <div className="pt-1 border-t border-slate-800 space-y-0.5 font-mono">
        <p className="text-emerald-400">Accuracy: {d.accuracy}%</p>
        <p className="text-cyan-400">Latency: {d.inference_latency_ms} ms</p>
        <p className="text-violet-400">Model Size: {d.model_size_kb} KB</p>
        <p className="text-amber-300">Fitness: {d.fitness_score}</p>
      </div>
    </div>
  );
}

export default function ParetoChart({ points = [], height = 300 }) {
  if (!points || points.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-sm text-slate-500 border border-dashed border-slate-800 rounded-xl">
        Run an experiment to plot the real multi-objective Pareto frontier.
      </div>
    );
  }

  const optimal = points.filter((p) => p.is_pareto_optimal);
  const dominated = points.filter((p) => !p.is_pareto_optimal);

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 12, right: 20, bottom: 32, left: 4 }}>
          <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="inference_latency_ms"
            name="Latency"
            unit=" ms"
            stroke="#94A3B8"
            tick={{ fontSize: 11 }}
            label={{ value: 'Inference Latency (ms, lower is faster)', position: 'insideBottom', offset: -12, fill: '#64748B', fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="accuracy"
            name="Accuracy"
            unit="%"
            stroke="#94A3B8"
            tick={{ fontSize: 11 }}
            domain={['auto', 'auto']}
          />
          <ZAxis type="number" dataKey="model_size_kb" range={[55, 220]} name="Model Size (KB)" />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: '#475569' }} />
          <Legend verticalAlign="bottom" wrapperStyle={{ fontSize: '12px', paddingTop: '18px' }} />
          <Scatter name="Evaluated Candidates" data={dominated} fill="#475569" fillOpacity={0.75} />
          <Scatter name="Pareto-Optimal Frontier (Rank 1)" data={optimal} fill="#8B5CF6" stroke="#22D3EE" strokeWidth={2} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
