import React from 'react';
import { fmtPct } from '../api/client';

const ORIGIN_COLORS = {
  random: { stroke: '#64748B', fill: '#1E293B', text: '#CBD5E1', label: 'Initial' },
  elitism: { stroke: '#F59E0B', fill: '#2B1F0E', text: '#FCD34D', label: 'Elitism' },
  crossover: { stroke: '#06B6D4', fill: '#0C2533', text: '#67E8F9', label: 'Crossover' },
  mutation: { stroke: '#8B5CF6', fill: '#22153A', text: '#C4B5FD', label: 'Mutation' },
  rl_guided: { stroke: '#10B981', fill: '#0E2920', text: '#6EE7B7', label: 'RL Policy' },
};

export default function EvolutionGraph({ graph, selectedId, onSelectNode }) {
  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-sm text-slate-500 border border-dashed border-slate-800 rounded-xl">
        No evolutionary lineage data available yet.
      </div>
    );
  }

  const maxGen = Math.max(...graph.nodes.map((n) => n.x), 1);
  const maxRow = Math.max(...graph.nodes.map((n) => n.y), 0);

  const colW = 240;
  const rowH = 98;
  const padX = 36;
  const padY = 54;
  const nodeW = 186;
  const nodeH = 66;

  const width = Math.max(640, padX * 2 + maxGen * colW);
  const height = Math.max(260, padY + (maxRow + 1) * rowH + 28);

  const coordMap = {};
  graph.nodes.forEach((n) => {
    const cx = padX + (n.x - 1) * colW + nodeW / 2;
    const cy = padY + n.y * rowH + nodeH / 2;
    coordMap[n.id] = { cx, cy, left: cx - nodeW / 2, right: cx + nodeW / 2, top: cy - nodeH / 2 };
  });

  return (
    <div className="space-y-3">
      {/* Legend */}
      <div className="flex flex-wrap items-center gap-3 text-xs">
        {Object.entries(ORIGIN_COLORS).map(([key, cfg]) => (
          <span key={key} className="inline-flex items-center gap-1.5 text-slate-300">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: cfg.stroke }} />
            {cfg.label}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5 text-amber-300 ml-2">
          ★ Generation Best
        </span>
        <span className="inline-flex items-center gap-1.5 text-cyan-300">
          ◆ Pareto Optimal
        </span>
      </div>

      <div className="w-full overflow-x-auto rounded-xl bg-[#070A12] border border-slate-800 p-3">
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="block select-none">
          {/* Generation column headers */}
          {Array.from({ length: maxGen }, (_, idx) => {
            const gen = idx + 1;
            const cx = padX + idx * colW + nodeW / 2;
            return (
              <g key={`hdr-${gen}`}>
                <line
                  x1={cx}
                  y1={28}
                  x2={cx}
                  y2={height - 16}
                  stroke="#1E293B"
                  strokeDasharray="4 4"
                  strokeWidth="1"
                />
                <rect x={cx - 46} y={6} width={92} height={24} rx={8} fill="#111827" stroke="#334155" />
                <text
                  x={cx}
                  y={22}
                  textAnchor="middle"
                  fill="#E2E8F0"
                  fontSize="11"
                  fontWeight="600"
                  fontFamily="JetBrains Mono, monospace"
                >
                  GEN {gen}
                </text>
              </g>
            );
          })}

          {/* Curved Lineage Edges */}
          {graph.edges.map((edge, i) => {
            const src = coordMap[edge.source];
            const dst = coordMap[edge.target];
            if (!src || !dst) return null;
            const style = ORIGIN_COLORS[edge.operation] || ORIGIN_COLORS.mutation;
            const midX = (src.right + dst.left) / 2;
            const isRelated = selectedId && (edge.source === selectedId || edge.target === selectedId);
            const pathD = `M ${src.right} ${src.cy} C ${midX} ${src.cy}, ${midX} ${dst.cy}, ${dst.left} ${dst.cy}`;
            return (
              <path
                key={`edge-${i}`}
                d={pathD}
                fill="none"
                stroke={style.stroke}
                strokeWidth={isRelated ? 2.4 : 1.4}
                strokeOpacity={selectedId ? (isRelated ? 0.95 : 0.18) : 0.55}
              />
            );
          })}

          {/* Architecture Nodes */}
          {graph.nodes.map((node) => {
            const pos = coordMap[node.id];
            const style = ORIGIN_COLORS[node.origin] || ORIGIN_COLORS.random;
            const isSelected = selectedId === node.id;

            return (
              <g
                key={node.id}
                onClick={() => onSelectNode?.(node)}
                className="cursor-pointer"
              >
                <rect
                  x={pos.left}
                  y={pos.top}
                  width={nodeW}
                  height={nodeH}
                  rx={12}
                  fill={style.fill}
                  stroke={isSelected ? '#FFFFFF' : style.stroke}
                  strokeWidth={isSelected ? 2.2 : node.is_gen_best ? 1.8 : 1.1}
                />
                <text
                  x={pos.left + 10}
                  y={pos.top + 20}
                  fill="#F8FAFC"
                  fontSize="10.5"
                  fontWeight="600"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {node.name.length > 21 ? `${node.name.slice(0, 21)}…` : node.name}
                </text>
                <text
                  x={pos.left + 10}
                  y={pos.top + 38}
                  fill={style.text}
                  fontSize="10"
                >
                  {style.label} · Fit {node.fitness_score != null ? node.fitness_score.toFixed(1) : '—'}
                </text>
                <text
                  x={pos.left + 10}
                  y={pos.top + 54}
                  fill="#94A3B8"
                  fontSize="10"
                  fontFamily="JetBrains Mono, monospace"
                >
                  Acc {fmtPct(node.accuracy, 1)} · {node.inference_latency_ms ? `${node.inference_latency_ms.toFixed(2)}ms` : '—'}
                </text>
                {node.is_gen_best && (
                  <text x={pos.right - 18} y={pos.top + 20} fill="#FBBF24" fontSize="13">
                    ★
                  </text>
                )}
                {node.is_pareto_optimal && (
                  <text x={pos.right - 18} y={pos.top + 54} fill="#22D3EE" fontSize="11">
                    ◆
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
