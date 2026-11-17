import React, { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Code2, Download, Layers, Play, Scale, Search } from 'lucide-react';
import { api, fmtMs, fmtParams, fmtPct, fmtSizeKb } from '../api/client';
import ArchitectureGraph from '../components/ArchitectureGraph';

export default function ExplorerPage() {
  const [searchParams] = useSearchParams();
  const initialArchId = searchParams.get('arch') || '';

  const [experiments, setExperiments] = useState([]);
  const [selectedExp, setSelectedExp] = useState('');
  const [architectures, setArchitectures] = useState([]);
  const [selectedArch, setSelectedArch] = useState(null);
  const [compareArch, setCompareArch] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listExperiments().then((res) => setExperiments(res.experiments || [])).catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    api
      .listArchitectures(selectedExp || undefined, true)
      .then((res) => {
        const list = res.architectures || [];
        setArchitectures(list);
        if (list.length > 0) {
          const match = initialArchId ? list.find((a) => a.id === initialArchId) : null;
          const sorted = [...list].sort((a, b) => (b.fitness_score || 0) - (a.fitness_score || 0));
          setSelectedArch(match || sorted[0]);
        } else {
          setSelectedArch(null);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [selectedExp, initialArchId]);

  const filtered = architectures.filter((a) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      a.name.toLowerCase().includes(q) ||
      a.origin.toLowerCase().includes(q) ||
      a.id.toLowerCase().includes(q)
    );
  });

  if (!loading && architectures.length === 0) {
    return (
      <div className="glass-card p-12 text-center space-y-4">
        <Layers className="w-10 h-10 text-cyan-400 mx-auto" />
        <h1 className="text-2xl font-bold text-white">Architecture Explorer</h1>
        <p className="text-sm text-slate-400 max-w-md mx-auto">
          No evaluated neural architectures found yet. Run an experiment to visually inspect
          discovered PyTorch layers, activations, dropout blocks, and parameter counts.
        </p>
        <Link to="/configure" className="btn-primary">
          <Play className="w-4 h-4" />
          Start NAS Experiment
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Architecture Explorer
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Inspect connected neural network layers, parameter distributions, and configurations.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <select
            value={selectedExp}
            onChange={(e) => setSelectedExp(e.target.value)}
            className="input-field w-auto min-w-[210px] py-2 text-xs"
          >
            <option value="">All Experiments</option>
            {experiments.map((exp) => (
              <option key={exp.id} value={exp.id}>
                {exp.name} ({exp.dataset})
              </option>
            ))}
          </select>

          <Link to="/compare" className="btn-secondary py-2 px-3.5 text-xs">
            <Scale className="w-3.5 h-3.5 text-cyan-400" />
            Full Comparison
          </Link>
        </div>
      </div>

      <div className="grid xl:grid-cols-12 gap-5 items-start">
        {/* Architecture Sidebar Selector */}
        <div className="xl:col-span-4 min-w-0 glass-card p-4 flex flex-col max-h-[760px]">
          <div className="relative mb-3">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter by name, origin, ID..."
              className="input-field pl-9 py-2 text-xs"
            />
          </div>

          <div className="space-y-2 overflow-y-auto pr-1 flex-1">
            {filtered.map((arch) => {
              const isSelected = selectedArch?.id === arch.id;
              const isCompared = compareArch?.id === arch.id;
              return (
                <div
                  key={arch.id}
                  onClick={() => setSelectedArch(arch)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-violet-600/15 border-violet-500/60 shadow-violet-glow'
                      : 'bg-[#0B1020] border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="font-mono text-xs font-semibold text-white truncate">{arch.name}</p>
                      <p className="text-[11px] text-slate-400 mt-0.5 capitalize">
                        Gen {arch.generation} · {arch.origin} · {arch.config?.hidden_layers?.length || 0} hidden layers
                      </p>
                    </div>
                    {arch.is_pareto_optimal && (
                      <span className="badge-cyan text-[10px] shrink-0">Pareto</span>
                    )}
                  </div>

                  <div className="mt-2.5 flex flex-wrap items-center justify-between gap-1.5 text-[11px] font-mono">
                    <span className="text-emerald-400">Acc {fmtPct(arch.accuracy, 1)}</span>
                    <span className="text-cyan-400">{fmtMs(arch.inference_latency_ms)}</span>
                    <span className="text-slate-400">{fmtParams(arch.parameter_count)}</span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setCompareArch(isCompared ? null : arch);
                      }}
                      className={`px-2 py-0.5 rounded text-[10px] border shrink-0 ${
                        isCompared
                          ? 'bg-cyan-500/20 border-cyan-400 text-cyan-200'
                          : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white'
                      }`}
                    >
                      {isCompared ? 'Comparing' : 'Pin Diff'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Visual Inspector */}
        <div className="xl:col-span-8 min-w-0 space-y-5">
          {selectedArch && (
            <>
              <div className="glass-card p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <h2 className="text-lg font-bold font-mono text-white">{selectedArch.name}</h2>
                      <span className="badge-violet capitalize">{selectedArch.origin}</span>
                      {selectedArch.is_pareto_optimal && <span className="badge-cyan">Pareto Optimal</span>}
                    </div>
                    <p className="text-xs text-slate-400 mt-1">
                      {selectedArch.operation_details || 'Discovered architecture'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <a
                      href={api.getConfigDownloadUrl(selectedArch.id)}
                      className="btn-secondary py-1.5 px-3 text-xs"
                    >
                      <Download className="w-3.5 h-3.5" />
                      JSON
                    </a>
                    <a
                      href={api.getModelDownloadUrl(selectedArch.id)}
                      className="btn-primary py-1.5 px-3 text-xs"
                    >
                      <Download className="w-3.5 h-3.5" />
                      .PT Weights
                    </a>
                  </div>
                </div>

                {/* Quick Metric Pills */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                  <div className="p-3 rounded-xl bg-[#0B1020] border border-slate-800">
                    <p className="text-[10px] text-slate-400 uppercase">Layers</p>
                    <p className="text-base font-bold font-mono text-white mt-0.5">
                      {selectedArch.config?.hidden_layers?.length ?? 0} Hidden
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-[#0B1020] border border-slate-800">
                    <p className="text-[10px] text-slate-400 uppercase">Parameters</p>
                    <p className="text-base font-bold font-mono text-violet-300 mt-0.5">
                      {selectedArch.parameter_count?.toLocaleString() ?? '—'}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-[#0B1020] border border-slate-800">
                    <p className="text-[10px] text-slate-400 uppercase">Model Size</p>
                    <p className="text-base font-bold font-mono text-amber-300 mt-0.5">
                      {fmtSizeKb(selectedArch.model_size_kb)}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-[#0B1020] border border-slate-800">
                    <p className="text-[10px] text-slate-400 uppercase">Accuracy</p>
                    <p className="text-base font-bold font-mono text-emerald-400 mt-0.5">
                      {fmtPct(selectedArch.accuracy)}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-[#0B1020] border border-slate-800">
                    <p className="text-[10px] text-slate-400 uppercase">Latency</p>
                    <p className="text-base font-bold font-mono text-cyan-400 mt-0.5">
                      {fmtMs(selectedArch.inference_latency_ms)}
                    </p>
                  </div>
                </div>

                {/* Interactive Neural Network Diagram */}
                <ArchitectureGraph layers={selectedArch.layer_summary || []} />
              </div>

              {/* Layer-by-Layer Table + JSON Config */}
              <div className="grid md:grid-cols-2 gap-5">
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-violet-400" />
                    Layer Breakdown
                  </h3>
                  <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                    {(selectedArch.layer_summary || []).map((layer, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-2.5 rounded-xl bg-[#0B1020] border border-slate-800 text-xs"
                      >
                        <div>
                          <span className="font-semibold text-white">{layer.name}</span>
                          <span className="ml-2 text-slate-400 font-mono text-[11px]">{layer.details}</span>
                        </div>
                        <span className="font-mono text-[11px] text-cyan-300">
                          {layer.params > 0 ? `${layer.params.toLocaleString()} p` : '0 p'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                    <Code2 className="w-4 h-4 text-cyan-400" />
                    Architecture Genome Configuration
                  </h3>
                  <pre className="p-3.5 rounded-xl bg-[#070A12] border border-slate-800 text-[11px] font-mono text-cyan-200 overflow-x-auto max-h-64">
                    {JSON.stringify(selectedArch.config, null, 2)}
                  </pre>
                </div>
              </div>

              {/* Quick Side-by-Side Comparison if Pinned */}
              {compareArch && compareArch.id !== selectedArch.id && (
                <div className="glass-card p-5 border-cyan-500/40 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">
                      Pinned Comparison: <span className="text-cyan-300 font-mono">{compareArch.name}</span>
                    </h3>
                    <button
                      onClick={() => setCompareArch(null)}
                      className="text-xs text-slate-400 hover:text-white"
                    >
                      Clear Pin
                    </button>
                  </div>
                  <ArchitectureGraph layers={compareArch.layer_summary || []} compact />
                  <div className="grid grid-cols-4 gap-3 text-xs font-mono pt-1">
                    <div>
                      <span className="text-slate-400 block">Accuracy Diff</span>
                      <span className="text-emerald-400">
                        {fmtPct(selectedArch.accuracy)} vs {fmtPct(compareArch.accuracy)}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">Latency Diff</span>
                      <span className="text-cyan-400">
                        {fmtMs(selectedArch.inference_latency_ms)} vs {fmtMs(compareArch.inference_latency_ms)}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">Params Diff</span>
                      <span className="text-violet-300">
                        {fmtParams(selectedArch.parameter_count)} vs {fmtParams(compareArch.parameter_count)}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">Fitness Diff</span>
                      <span className="text-amber-300">
                        {selectedArch.fitness_score?.toFixed(2)} vs {compareArch.fitness_score?.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
