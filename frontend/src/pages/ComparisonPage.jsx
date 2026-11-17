import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { CheckSquare, Play, Scale, Square } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  api,
  fmtMs,
  fmtParams,
  fmtPct,
  fmtSeconds,
  fmtSizeKb,
} from '../api/client';
import ParetoChart from '../components/ParetoChart';

export default function ComparisonPage() {
  const [experiments, setExperiments] = useState([]);
  const [selectedExp, setSelectedExp] = useState('');
  const [architectures, setArchitectures] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listExperiments().then((res) => setExperiments(res.experiments || [])).catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    api
      .listArchitectures(selectedExp || undefined, true)
      .then((res) => {
        const list = (res.architectures || []).sort(
          (a, b) => (b.fitness_score || 0) - (a.fitness_score || 0)
        );
        setArchitectures(list);
        setSelectedIds(list.slice(0, Math.min(4, list.length)).map((a) => a.id));
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [selectedExp]);

  const toggleSelect = (id) => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) {
        return prev.length > 1 ? prev.filter((x) => x !== id) : prev;
      }
      if (prev.length >= 8) return prev;
      return [...prev, id];
    });
  };

  const selectedModels = architectures
    .filter((a) => selectedIds.includes(a.id))
    .sort((a, b) => (b.fitness_score || 0) - (a.fitness_score || 0));

  if (!loading && architectures.length === 0) {
    return (
      <div className="glass-card p-12 text-center space-y-4">
        <Scale className="w-10 h-10 text-cyan-400 mx-auto" />
        <h1 className="text-2xl font-bold text-white">Model Comparison</h1>
        <p className="text-sm text-slate-400 max-w-md mx-auto">
          Evaluate neural architectures in a NAS experiment first to compare accuracy, precision,
          recall, F1-score, latency, memory usage, and parameter count side-by-side.
        </p>
        <Link to="/configure" className="btn-primary">
          <Play className="w-4 h-4" />
          Start NAS Experiment
        </Link>
      </div>
    );
  }

  const qualityChartData = selectedModels.map((m) => ({
    name: m.name.split(' ')[0],
    fullName: m.name,
    Accuracy: Number(((m.accuracy || 0) * 100).toFixed(2)),
    Precision: Number(((m.precision_score || 0) * 100).toFixed(2)),
    Recall: Number(((m.recall_score || 0) * 100).toFixed(2)),
    F1: Number(((m.f1_score || 0) * 100).toFixed(2)),
  }));

  const efficiencyChartData = selectedModels.map((m) => ({
    name: m.name.split(' ')[0],
    LatencyMs: Number((m.inference_latency_ms || 0).toFixed(3)),
    SizeKB: Number((m.model_size_kb || 0).toFixed(1)),
    TrainTimeS: Number((m.training_time_s || 0).toFixed(2)),
  }));

  const paretoPoints = selectedModels.map((m) => ({
    id: m.id,
    name: m.name,
    generation: m.generation,
    accuracy: Number(((m.accuracy || 0) * 100).toFixed(2)),
    inference_latency_ms: Number((m.inference_latency_ms || 0).toFixed(4)),
    model_size_kb: Number((m.model_size_kb || 0).toFixed(2)),
    fitness_score: Number((m.fitness_score || 0).toFixed(2)),
    is_pareto_optimal: m.is_pareto_optimal,
    origin: m.origin,
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Model Comparison Studio
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Select up to 8 evaluated architectures to compare classification quality, speed, and footprint.
          </p>
        </div>

        <select
          value={selectedExp}
          onChange={(e) => setSelectedExp(e.target.value)}
          className="input-field w-auto min-w-[220px] py-2 text-xs"
        >
          <option value="">All Experiments</option>
          {experiments.map((exp) => (
            <option key={exp.id} value={exp.id}>
              {exp.name} ({exp.dataset})
            </option>
          ))}
        </select>
      </div>

      {/* Candidate Multi-Selector */}
      <div className="glass-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">
            Select Architectures ({selectedIds.length} selected)
          </h2>
          <div className="flex gap-2 text-xs">
            <button
              onClick={() => setSelectedIds(architectures.slice(0, 4).map((a) => a.id))}
              className="text-cyan-400 hover:text-cyan-300"
            >
              Select Top 4
            </button>
            <span className="text-slate-600">·</span>
            <button
              onClick={() =>
                setSelectedIds(
                  architectures
                    .filter((a) => a.is_pareto_optimal)
                    .slice(0, 8)
                    .map((a) => a.id)
                )
              }
              className="text-violet-400 hover:text-violet-300"
            >
              Select Pareto Frontier
            </button>
          </div>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2.5 max-h-56 overflow-y-auto pr-1">
          {architectures.map((a) => {
            const checked = selectedIds.includes(a.id);
            return (
              <button
                key={a.id}
                type="button"
                onClick={() => toggleSelect(a.id)}
                className={`text-left p-3 rounded-xl border transition-all flex items-start gap-2.5 ${
                  checked
                    ? 'bg-violet-500/15 border-violet-500/60 text-white'
                    : 'bg-[#0B1020] border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                {checked ? (
                  <CheckSquare className="w-4 h-4 text-violet-400 shrink-0 mt-0.5" />
                ) : (
                  <Square className="w-4 h-4 text-slate-600 shrink-0 mt-0.5" />
                )}
                <div className="min-w-0">
                  <p className="font-mono text-xs font-semibold truncate">{a.name}</p>
                  <p className="text-[11px] font-mono text-slate-400 mt-0.5">
                    Acc {fmtPct(a.accuracy, 1)} · {fmtMs(a.inference_latency_ms)} · Fit {a.fitness_score?.toFixed(1)}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Comparison Charts */}
      <div className="grid lg:grid-cols-2 gap-5">
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold text-white mb-1">
            Classification Metrics (Accuracy, Precision, Recall, F1)
          </h2>
          <p className="text-xs text-slate-400 mb-4">Weighted scikit-learn validation metrics (%)</p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={qualityChartData}>
                <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                <XAxis dataKey="name" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} domain={['auto', 100]} unit="%" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0B1020', borderColor: '#334155', borderRadius: 12, fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="Accuracy" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Precision" fill="#06B6D4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Recall" fill="#10B981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="F1" fill="#F59E0B" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold text-white mb-1">
            Inference Latency & Checkpoint Size
          </h2>
          <p className="text-xs text-slate-400 mb-4">CPU forward-pass speed (ms) and disk size (KB)</p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={efficiencyChartData}>
                <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                <XAxis dataKey="name" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" stroke="#06B6D4" tick={{ fontSize: 11 }} unit="ms" />
                <YAxis yAxisId="right" orientation="right" stroke="#8B5CF6" tick={{ fontSize: 11 }} unit="KB" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0B1020', borderColor: '#334155', borderRadius: 12, fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar yAxisId="left" dataKey="LatencyMs" name="Latency (ms)" fill="#06B6D4" radius={[4, 4, 0, 0]} />
                <Bar yAxisId="right" dataKey="SizeKB" name="Model Size (KB)" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Pareto Tradeoff Scatter */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-semibold text-white mb-1">Selected Models Pareto Tradeoff</h2>
        <p className="text-xs text-slate-400 mb-3">
          Higher & left is better (higher accuracy, lower inference latency).
        </p>
        <ParetoChart points={paretoPoints} height={260} />
      </div>

      {/* Full Ranked Comparison Matrix */}
      <div className="glass-card p-5">
        <h2 className="text-base font-semibold text-white mb-4">Ranked Comparison Matrix</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase">
                <th className="py-3 pr-3">Rank</th>
                <th className="py-3 px-2">Architecture</th>
                <th className="py-3 px-2">Accuracy</th>
                <th className="py-3 px-2">Precision</th>
                <th className="py-3 px-2">Recall</th>
                <th className="py-3 px-2">F1-Score</th>
                <th className="py-3 px-2">Train Time</th>
                <th className="py-3 px-2">Latency</th>
                <th className="py-3 px-2">Params</th>
                <th className="py-3 px-2">Size</th>
                <th className="py-3 px-2">Memory</th>
                <th className="py-3 pl-2 text-right">Fitness</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 font-mono">
              {selectedModels.map((m, idx) => (
                <tr key={m.id} className="hover:bg-slate-800/35">
                  <td className="py-3 pr-3">
                    <span
                      className={`inline-flex items-center justify-center w-6 h-6 rounded-lg font-bold ${
                        idx === 0
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                          : 'bg-slate-800 text-slate-300'
                      }`}
                    >
                      {idx + 1}
                    </span>
                  </td>
                  <td className="py-3 px-2">
                    <Link to={`/explorer?arch=${m.id}`} className="text-white hover:text-cyan-300 font-semibold">
                      {m.name}
                    </Link>
                    <span className="block text-[10px] text-slate-500 font-sans capitalize">
                      Gen {m.generation} · {m.origin} {m.is_pareto_optimal ? '· Pareto ★' : ''}
                    </span>
                  </td>
                  <td className="py-3 px-2 text-emerald-400 font-semibold">{fmtPct(m.accuracy)}</td>
                  <td className="py-3 px-2 text-slate-200">{fmtPct(m.precision_score)}</td>
                  <td className="py-3 px-2 text-slate-200">{fmtPct(m.recall_score)}</td>
                  <td className="py-3 px-2 text-cyan-300">{fmtPct(m.f1_score)}</td>
                  <td className="py-3 px-2 text-slate-300">{fmtSeconds(m.training_time_s)}</td>
                  <td className="py-3 px-2 text-cyan-400">{fmtMs(m.inference_latency_ms)}</td>
                  <td className="py-3 px-2 text-slate-300">{fmtParams(m.parameter_count)}</td>
                  <td className="py-3 px-2 text-amber-300">{fmtSizeKb(m.model_size_kb)}</td>
                  <td className="py-3 px-2 text-slate-300">
                    {m.peak_memory_mb != null ? `${m.peak_memory_mb} MB` : '—'}
                  </td>
                  <td className="py-3 pl-2 text-right font-bold text-violet-300">
                    {m.fitness_score?.toFixed(2) ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
