import React, { useState } from 'react';
import { fmtParams } from '../api/client';

const TYPE_STYLES = {
  Input: {
    fill: '#0E2439',
    stroke: '#06B6D4',
    node: '#22D3EE',
    badge: 'text-cyan-300 bg-cyan-500/15 border-cyan-500/40',
  },
  Dense: {
    fill: '#1E1538',
    stroke: '#8B5CF6',
    node: '#A78BFA',
    badge: 'text-violet-300 bg-violet-500/15 border-violet-500/40',
  },
  BatchNorm: {
    fill: '#13272F',
    stroke: '#14B8A6',
    node: '#2DD4BF',
    badge: 'text-teal-300 bg-teal-500/15 border-teal-500/40',
  },
  Activation: {
    fill: '#2B1F0F',
    stroke: '#F59E0B',
    node: '#FBBF24',
    badge: 'text-amber-300 bg-amber-500/15 border-amber-500/40',
  },
  Dropout: {
    fill: '#2B1220',
    stroke: '#F43F5E',
    node: '#FB7185',
    badge: 'text-rose-300 bg-rose-500/15 border-rose-500/40',
  },
  Output: {
    fill: '#0F2922',
    stroke: '#10B981',
    node: '#34D399',
    badge: 'text-emerald-300 bg-emerald-500/15 border-emerald-500/40',
  },
};

function getVisualNodes(layer) {
  if (layer.type === 'Activation' || layer.type === 'Dropout' || layer.type === 'BatchNorm') {
    return 3;
  }
  const units = Number(layer.out_features || 16);
  if (units >= 256) return 7;
  if (units >= 128) return 6;
  if (units >= 64) return 5;
  if (units >= 32) return 4;
  return Math.max(3, Math.min(5, units));
}

export default function ArchitectureGraph({ layers = [], compact = false }) {
  const [hoveredIdx, setHoveredIdx] = useState(null);

  if (!layers || layers.length === 0) {
    return (
      <div className="h-56 flex items-center justify-center text-sm text-slate-500 border border-dashed border-slate-800 rounded-xl">
        No layer topology available for this architecture.
      </div>
    );
  }

  const colWidth = compact ? 116 : 140;
  const padX = 36;
  const height = compact ? 235 : 285;
  const width = Math.max(580, padX * 2 + layers.length * colWidth);
  const centerY = height / 2 - 12;

  const layerPositions = layers.map((layer, idx) => {
    const x = padX + idx * colWidth + colWidth / 2;
    const count = getVisualNodes(layer);
    const spacing = Math.min(26, (height - 120) / Math.max(1, count));
    const startY = centerY - ((count - 1) * spacing) / 2;
    const nodes = Array.from({ length: count }, (_, i) => ({
      x,
      y: startY + i * spacing,
    }));
    return { layer, x, nodes };
  });

  return (
    <div className="w-full overflow-x-auto rounded-xl bg-[#070A12] border border-slate-800/90 p-3">
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="mx-auto block select-none"
      >
        <defs>
          <linearGradient id="synapseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#06B6D4" stopOpacity="0.35" />
          </linearGradient>
          <filter id="nodeGlow">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Inter-layer synaptic connections */}
        {layerPositions.slice(0, -1).map((curr, idx) => {
          const next = layerPositions[idx + 1];
          const isHighlighted = hoveredIdx === idx || hoveredIdx === idx + 1;
          return (
            <g key={`edges-${idx}`}>
              {curr.nodes.map((n1, i) =>
                next.nodes.map((n2, j) => (
                  <line
                    key={`${i}-${j}`}
                    x1={n1.x}
                    y1={n1.y}
                    x2={n2.x}
                    y2={n2.y}
                    stroke={isHighlighted ? '#8B5CF6' : 'url(#synapseGrad)'}
                    strokeOpacity={isHighlighted ? 0.65 : 0.35}
                    strokeWidth={isHighlighted ? 1.3 : 0.85}
                  />
                ))
              )}
            </g>
          );
        })}

        {/* Layer Columns */}
        {layerPositions.map(({ layer, x, nodes }, idx) => {
          const style = TYPE_STYLES[layer.type] || TYPE_STYLES.Dense;
          const isHovered = hoveredIdx === idx;
          const boxW = compact ? 84 : 102;
          const boxTop = 20;
          const boxH = height - 66;

          return (
            <g
              key={`layer-${idx}`}
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
              className="cursor-pointer"
            >
              <rect
                x={x - boxW / 2}
                y={boxTop}
                width={boxW}
                height={boxH}
                rx={12}
                fill={style.fill}
                fillOpacity={isHovered ? 0.95 : 0.68}
                stroke={style.stroke}
                strokeOpacity={isHovered ? 1 : 0.45}
                strokeWidth={isHovered ? 2 : 1.1}
              />

              {/* Layer Type Header */}
              <text
                x={x}
                y={boxTop + 17}
                textAnchor="middle"
                fill={style.node}
                fontSize="10"
                fontWeight="700"
                fontFamily="JetBrains Mono, monospace"
              >
                {layer.type.toUpperCase()}
              </text>

              {/* Neurons */}
              {nodes.map((n, ni) => (
                <circle
                  key={ni}
                  cx={n.x}
                  cy={n.y}
                  r={layer.type === 'Dense' || layer.type === 'Input' || layer.type === 'Output' ? 5.2 : 4}
                  fill={style.node}
                  fillOpacity={layer.type === 'Dropout' && ni === 1 ? 0.25 : 0.92}
                  stroke="#080B14"
                  strokeWidth="1.5"
                  filter={isHovered ? 'url(#nodeGlow)' : undefined}
                />
              ))}

              {/* Layer Name & Dim Footer */}
              <text
                x={x}
                y={height - 26}
                textAnchor="middle"
                fill="#F8FAFC"
                fontSize="11"
                fontWeight="600"
              >
                {layer.name}
              </text>
              <text
                x={x}
                y={height - 10}
                textAnchor="middle"
                fill="#94A3B8"
                fontSize="9.5"
                fontFamily="JetBrains Mono, monospace"
              >
                {layer.params > 0 ? `${fmtParams(layer.params)} params` : `${layer.out_features} dim`}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Active Layer Inspector Bar */}
      <div className="mt-2.5 pt-2.5 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs px-1">
        {hoveredIdx !== null && layers[hoveredIdx] ? (
          <div className="flex flex-wrap items-center gap-2.5">
            <span
              className={`px-2 py-0.5 rounded-md border font-mono text-[11px] ${
                (TYPE_STYLES[layers[hoveredIdx].type] || TYPE_STYLES.Dense).badge
              }`}
            >
              {layers[hoveredIdx].type}
            </span>
            <span className="text-slate-200 font-medium">{layers[hoveredIdx].details}</span>
            <span className="text-slate-400 font-mono">
              Parameters: {layers[hoveredIdx].params.toLocaleString()}
            </span>
          </div>
        ) : (
          <span className="text-slate-500">Hover any layer block to inspect dimensions and parameter count.</span>
        )}
        <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
          <span>Layers: {layers.length}</span>
          <span>
            Total Params: {layers.reduce((s, l) => s + (l.params || 0), 0).toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}
