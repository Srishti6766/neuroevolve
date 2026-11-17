import React, { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  Award,
  Clock,
  Code2,
  Cpu,
  Download,
  HardDrive,
  Layers,
  Play,
  Sparkles,
} from 'lucide-react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
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
import ArchitectureGraph from '../components/ArchitectureGraph';

export default function BestArchitecturePage() {
  const [searchParams] = useSearchParams();
  const queryExp = searchParams.get('exp') || '';

  const [experiments, setExperiments] = useState([]);
  const [selectedExp, setSelectedExp] = useState(queryExp);
  const [bestArch, setBestArch] = useState(null);
  const [expMeta, setExpMeta] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listExperiments().then((res) => setExperiments(res.experiments || [])).catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    api
      .getBestArchitecture(selectedExp || undefined)
      .then((res) => {
        setBestArch(res.architecture);
        setExpMeta(res.experiment);
      })
      .catch(() => {
        setBestArch(null);
      })
      .finally(() => setLoading(false));
  }, [selectedExp]);

  if (!loading && !bestArch) {
    return (
      <div className="glass-card p-12 text-center space-y-4">
        <Award className="w-10 h-10 text-amber-400 mx-auto" />
        <h1 className="text-2xl font-bold text-white">Best Discovered Architecture</h1>
        <p className="text-sm text-slate-400 max-w-md mx-auto">
          No trained architectures are available yet. Run a Neural Architecture Search experiment to
          discover and export your highest-performing PyTorch model.
        </p>
        <Link to="/configure" className="btn-primary">
          <Play className="w-4 h-4" />
          Start NAS Experiment
        </Link>
      </div>
    );
  }

  const historyData = (bestArch?.training_history || []).map((ep) => ({
    epoch: `Ep ${ep.epoch}`,
    train_loss: ep.train_loss,
    val_loss: ep.val_loss,
    train_acc: Number((ep.train_accuracy * 100).toFixed(2)),
    val_acc: Number((ep.val_accuracy * 100).toFixed(2)),
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Best Discovered Architecture
            </h1>
            <span className="badge-emerald">CHAMPION MODEL</span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Highest multi-objective fitness neural network discovered by NEUROEVOLVE.
          </p>
        </div>

        <select
          value={selectedExp}
          onChange={(e) => setSelectedExp(e.target.value)}
          className="input-field w-auto min-w-[230px] py-2 text-xs"
        >
          <option value="">Global Best (All Experiments)</option>
          {experiments.map((exp) => (
            <option key={exp.id} value={exp.id}>
              {exp.name} ({exp.dataset})
            </option>
          ))}
        </select>
      </div>

      {bestArch && (
        <>
          {/* Champion Hero Card */}
          <div className="glass-card p-6 border-violet-500/40 bg-gradient-to-br from-violet-950/35 via-[#111827] to-cyan-950/30">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5">
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <Sparkles className="w-5 h-5 text-amber-400" />
                  <h2 className="text-xl sm:text-2xl font-extrabold font-mono text-white">
                    {bestArch.name}
                  </h2>
                  <span className="badge-violet capitalize">{bestArch.origin}</span>
                  {bestArch.is_pareto_optimal && (
                    <span className="badge-cyan">Pareto Frontier (Rank 1)</span>
                  )}
                </div>
                <p className="text-xs text-slate-300 font-mono">
                  Experiment: {expMeta?.name || bestArch.experiment_id} · Dataset:{' '}
                  <span className="uppercase text-cyan-300">{expMeta?.dataset || '—'}</span> · Generation{' '}
                  {bestArch.generation} · {bestArch.operation_details}
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3 shrink-0">
                <a
                  href={api.getModelDownloadUrl(bestArch.id)}
                  className="btn-primary shadow-violet-glow"
                >
                  <Download className="w-4 h-4" />
                  Download Trained Model (.pt)
                </a>
                <a
                  href={api.getConfigDownloadUrl(bestArch.id)}
                  className="btn-secondary"
                >
                  <Download className="w-4 h-4 text-cyan-400" />
                  Download Config (.json)
                </a>
              </div>
            </div>

            {/* Full Metrics Grid */}
            <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5">
              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400">Validation Accuracy</p>
                <p className="mt-1 text-xl font-bold font-mono text-emerald-400">
                  {fmtPct(bestArch.accuracy)}
                </p>
              </div>
              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400">Weighted F1-Score</p>
                <p className="mt-1 text-xl font-bold font-mono text-cyan-300">
                  {fmtPct(bestArch.f1_score)}
                </p>
              </div>
              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400">Precision / Recall</p>
                <p className="mt-1 text-base font-bold font-mono text-slate-100">
                  {fmtPct(bestArch.precision_score, 1)} / {fmtPct(bestArch.recall_score, 1)}
                </p>
              </div>
              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400">Inference Latency</p>
                <p className="mt-1 text-xl font-bold font-mono text-cyan-400">
                  {fmtMs(bestArch.inference_latency_ms)}
                </p>
              </div>
              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400">Fitness Score</p>
                <p className="mt-1 text-xl font-bold font-mono text-violet-300">
                  {bestArch.fitness_score?.toFixed(2) ?? '—'}
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400 flex items-center gap-1">
                  <Layers className="w-3 h-3" /> Parameters
                </p>
                <p className="mt-1 text-base font-bold font-mono text-white">
                  {bestArch.parameter_count?.toLocaleString() ?? '—'}
                </p>
              </div>
              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400 flex items-center gap-1">
                  <HardDrive className="w-3 h-3" /> Model File Size
                </p>
                <p className="mt-1 text-base font-bold font-mono text-amber-300">
                  {fmtSizeKb(bestArch.model_size_kb)}
                </p>
              </div>
              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400 flex items-center gap-1">
                  <Cpu className="w-3 h-3" /> Peak Memory
                </p>
                <p className="mt-1 text-base font-bold font-mono text-violet-300">
                  {bestArch.peak_memory_mb != null ? `${bestArch.peak_memory_mb} MB` : '—'}
                </p>
              </div>
              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400 flex items-center gap-1">
                  <Clock className="w-3 h-3" /> Training Duration
                </p>
                <p className="mt-1 text-base font-bold font-mono text-slate-200">
                  {fmtSeconds(bestArch.training_time_s)}
                </p>
              </div>
              <div className="p-3.5 rounded-xl bg-[#080B14]/80 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-400">Validation Loss</p>
                <p className="mt-1 text-base font-bold font-mono text-rose-300">
                  {bestArch.val_loss?.toFixed(4) ?? '—'}
                </p>
              </div>
            </div>
          </div>

          {/* Architecture Topology Visualization */}
          <div className="glass-card p-5 space-y-3">
            <h2 className="text-base font-semibold text-white">Neural Network Topology Visualization</h2>
            <ArchitectureGraph layers={bestArch.layer_summary || []} />
          </div>

          {/* Training History & Configuration */}
          <div className="grid lg:grid-cols-2 gap-5">
            <div className="glass-card p-5 space-y-3">
              <h2 className="text-base font-semibold text-white">Training & Validation History</h2>
              <div className="h-72">
                {historyData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={historyData}>
                      <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                      <YAxis yAxisId="acc" stroke="#10B981" tick={{ fontSize: 11 }} unit="%" />
                      <YAxis yAxisId="loss" orientation="right" stroke="#F59E0B" tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0B1020', borderColor: '#334155', borderRadius: 12, fontSize: 12 }}
                      />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <Line yAxisId="acc" type="monotone" dataKey="val_acc" name="Val Accuracy (%)" stroke="#10B981" strokeWidth={2.5} />
                      <Line yAxisId="acc" type="monotone" dataKey="train_acc" name="Train Accuracy (%)" stroke="#06B6D4" strokeWidth={2} strokeDasharray="4 4" />
                      <Line yAxisId="loss" type="monotone" dataKey="val_loss" name="Val Loss" stroke="#F59E0B" strokeWidth={2} />
                      <Line yAxisId="loss" type="monotone" dataKey="train_loss" name="Train Loss" stroke="#8B5CF6" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs text-slate-500">
                    No epoch history recorded.
                  </div>
                )}
              </div>
            </div>

            <div className="glass-card p-5 space-y-3">
              <h2 className="text-base font-semibold text-white flex items-center gap-2">
                <Code2 className="w-4 h-4 text-cyan-400" />
                Model Configuration JSON
              </h2>
              <pre className="p-4 rounded-xl bg-[#070A12] border border-slate-800 text-xs font-mono text-cyan-200 overflow-x-auto max-h-72">
                {JSON.stringify(
                  {
                    architecture_id: bestArch.id,
                    name: bestArch.name,
                    generation: bestArch.generation,
                    origin: bestArch.origin,
                    parameter_count: bestArch.parameter_count,
                    model_size_kb: bestArch.model_size_kb,
                    config: bestArch.config,
                  },
                  null,
                  2
                )}
              </pre>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
