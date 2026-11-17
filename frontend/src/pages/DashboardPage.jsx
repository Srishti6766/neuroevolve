import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  Award,
  Clock,
  Cpu,
  Database,
  FlaskConical,
  GitBranch,
  HardDrive,
  Layers,
  Play,
  RefreshCw,
  Trash2,
  TrendingUp,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api, createExperimentSocket, fmtMs, fmtPct, fmtSizeKb } from '../api/client';
import StatCard from '../components/StatCard';
import ParetoChart from '../components/ParetoChart';

const STATUS_BADGE = {
  running: 'badge-cyan',
  paused: 'badge-amber',
  completed: 'badge-emerald',
  stopped: 'badge-amber',
  failed: 'badge-rose',
  queued: 'badge-violet',
};

export default function DashboardPage() {
  const [experiments, setExperiments] = useState([]);
  const [selectedExpId, setSelectedExpId] = useState('');
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadData = useCallback(async (expId = selectedExpId) => {
    try {
      setError('');
      const [expRes, dashRes] = await Promise.all([
        api.listExperiments(),
        api.getDashboard(expId || undefined),
      ]);
      setExperiments(expRes.experiments || []);
      setSummary(dashRes);
    } catch (err) {
      setError(err.message || 'Failed to load dashboard data from backend.');
    } finally {
      setLoading(false);
    }
  }, [selectedExpId]);

  useEffect(() => {
    loadData(selectedExpId);
  }, [selectedExpId, loadData]);

  // Live refresh on WebSocket events
  useEffect(() => {
    const disconnect = createExperimentSocket(null, (event) => {
      if (
        ['candidate_evaluated', 'generation_completed', 'experiment_completed', 'experiment_started', 'experiment_stopped'].includes(
          event.type
        )
      ) {
        loadData(selectedExpId);
      }
    });
    return disconnect;
  }, [selectedExpId, loadData]);

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this experiment and its saved model weights?')) return;
    try {
      await api.deleteExperiment(id);
      if (selectedExpId === id) setSelectedExpId('');
      loadData('');
    } catch (err) {
      setError(err.message);
    }
  };

  const genChartData = (summary?.generation_metrics || []).map((g) => ({
    ...g,
    label: `Gen ${g.generation}`,
    best_acc_pct: Number((g.best_accuracy * 100).toFixed(2)),
    avg_acc_pct: Number((g.avg_accuracy * 100).toFixed(2)),
  }));

  const activeExp = summary?.active_experiment;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Research Dashboard
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time telemetry from PyTorch Neural Architecture Search runs stored in SQLite.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <select
            value={selectedExpId}
            onChange={(e) => setSelectedExpId(e.target.value)}
            className="input-field w-auto min-w-[210px] py-2 text-xs"
          >
            <option value="">All Experiments (Global View)</option>
            {experiments.map((exp) => (
              <option key={exp.id} value={exp.id}>
                {exp.name} ({exp.dataset} · {exp.status})
              </option>
            ))}
          </select>

          <button onClick={() => loadData(selectedExpId)} className="btn-secondary py-2 px-3.5 text-xs">
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </button>

          <Link to="/configure" className="btn-primary py-2 px-4 text-xs">
            <Play className="w-3.5 h-3.5" />
            New Experiment
          </Link>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/50 border border-rose-700/50 text-rose-200 text-sm">
          {error}
        </div>
      )}

      {/* Active Experiment Banner */}
      {activeExp && (
        <div className="glass-card p-4 border-cyan-500/40 bg-gradient-to-r from-violet-950/40 via-[#111827] to-cyan-950/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className="w-3 h-3 rounded-full bg-cyan-400 animate-ping shrink-0" />
            <div>
              <p className="text-sm font-semibold text-white">
                Active Experiment: {activeExp.name}{' '}
                <span className="badge-cyan ml-2 uppercase">{activeExp.status}</span>
              </p>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Gen {activeExp.current_generation}/{activeExp.total_generations} · Evaluated {activeExp.evaluated_count} models · Dataset: {activeExp.dataset}
              </p>
            </div>
          </div>
          <Link to={`/monitor/${activeExp.id}`} className="btn-primary py-2 px-4 text-xs shrink-0">
            <Activity className="w-3.5 h-3.5" />
            Open Live Monitor
          </Link>
        </div>
      )}

      {/* KPI Metrics Grid */}
      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 7 }).map((_, i) => (
            <div key={i} className="glass-card h-28 animate-pulse bg-slate-900/50" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
          <StatCard
            label="Total Experiments"
            value={summary?.total_experiments ?? 0}
            subtitle={activeExp ? `1 active (${activeExp.status})` : 'No active run'}
            icon={FlaskConical}
            accent="violet"
          />
          <StatCard
            label="Architectures Evaluated"
            value={summary?.total_architectures_evaluated ?? 0}
            subtitle="Real PyTorch models trained"
            icon={Layers}
            accent="cyan"
          />
          <StatCard
            label="Current Generation"
            value={
              summary?.total_generations
                ? `${summary.current_generation} / ${summary.total_generations}`
                : '0'
            }
            subtitle={summary?.selected_experiment ? summary.selected_experiment.name : 'Start a search'}
            icon={GitBranch}
            accent="violet"
          />
          <StatCard
            label="Best Accuracy"
            value={fmtPct(summary?.best_accuracy)}
            subtitle={
              summary?.best_accuracy_architecture
                ? summary.best_accuracy_architecture.name
                : summary?.best_architecture
                ? summary.best_architecture.name
                : 'No models yet'
            }
            icon={Award}
            accent="emerald"
          />
          <StatCard
            label="Best Inference Latency"
            value={fmtMs(summary?.best_latency_ms)}
            subtitle="Single-sample CPU forward pass"
            icon={Clock}
            accent="cyan"
          />
          <StatCard
            label="Best Model Size"
            value={fmtSizeKb(summary?.best_model_size_kb)}
            subtitle="Saved PyTorch checkpoint (.pt)"
            icon={HardDrive}
            accent="amber"
          />
          <StatCard
            label="Active Status"
            value={activeExp ? activeExp.status.toUpperCase() : 'IDLE'}
            subtitle={activeExp ? `Dataset: ${activeExp.dataset}` : 'Ready for next NAS run'}
            icon={Cpu}
            accent={activeExp ? 'cyan' : 'emerald'}
          />
          <StatCard
            label="Best Fitness Score"
            value={
              summary?.best_architecture?.fitness_score != null
                ? `${summary.best_architecture.fitness_score.toFixed(1)} / 100`
                : '—'
            }
            subtitle="Multi-objective composite score"
            icon={TrendingUp}
            accent="violet"
          />
        </div>
      )}

      {/* Empty State Prompt if No Experiments Yet */}
      {!loading && experiments.length === 0 && (
        <div className="glass-card p-10 text-center space-y-4 border-dashed border-violet-500/30">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-violet-500/15 border border-violet-500/30 flex items-center justify-center">
            <Database className="w-7 h-7 text-violet-400" />
          </div>
          <h2 className="text-xl font-bold text-white">No NAS Experiments Recorded Yet</h2>
          <p className="text-sm text-slate-400 max-w-md mx-auto">
            Launch your first real PyTorch Neural Architecture Search experiment to populate accuracy,
            latency, model size, and evolutionary fitness charts.
          </p>
          <Link to="/configure" className="btn-primary">
            <Play className="w-4 h-4" />
            Configure & Start First Experiment
          </Link>
        </div>
      )}

      {/* Charts Grid */}
      {genChartData.length > 0 && (
        <>
          <div className="grid lg:grid-cols-2 gap-5">
            {/* 1. Accuracy across Generations */}
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-sm font-semibold text-white">Accuracy Across Generations</h2>
                  <p className="text-xs text-slate-400">Best & mean validation accuracy (%) per generation</p>
                </div>
                <span className="badge-emerald">Validation %</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={genChartData}>
                    <defs>
                      <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                    <XAxis dataKey="label" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} domain={['auto', 100]} unit="%" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0B1020', borderColor: '#334155', borderRadius: 12, fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Area type="monotone" dataKey="best_acc_pct" name="Best Accuracy (%)" stroke="#8B5CF6" fill="url(#accGrad)" strokeWidth={2.5} />
                    <Line type="monotone" dataKey="avg_acc_pct" name="Mean Accuracy (%)" stroke="#06B6D4" strokeWidth={2} dot={{ r: 3 }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 2. Inference Latency across Generations */}
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-sm font-semibold text-white">Inference Latency Across Generations</h2>
                  <p className="text-xs text-slate-400">Single-sample CPU forward pass (ms, lower is faster)</p>
                </div>
                <span className="badge-cyan">ms / sample</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={genChartData}>
                    <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                    <XAxis dataKey="label" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} unit="ms" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0B1020', borderColor: '#334155', borderRadius: 12, fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="best_latency_ms" name="Fastest Latency (ms)" stroke="#06B6D4" strokeWidth={2.5} dot={{ r: 4 }} />
                    <Line type="monotone" dataKey="avg_latency_ms" name="Mean Latency (ms)" stroke="#F59E0B" strokeWidth={2} strokeDasharray="4 4" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 3. Model Size Comparison */}
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-sm font-semibold text-white">Model Size Comparison</h2>
                  <p className="text-xs text-slate-400">Saved PyTorch checkpoint size (KB) by generation</p>
                </div>
                <span className="badge-amber">KB on Disk</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={genChartData}>
                    <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                    <XAxis dataKey="label" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} unit=" KB" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0B1020', borderColor: '#334155', borderRadius: 12, fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="avg_model_size_kb" name="Avg Model Size (KB)" fill="#8B5CF6" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="min_model_size_kb" name="Smallest Model (KB)" fill="#06B6D4" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 4. Evolutionary Fitness Progress */}
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-sm font-semibold text-white">Evolutionary Fitness Progress</h2>
                  <p className="text-xs text-slate-400">Multi-objective fitness optimization curve (0–100)</p>
                </div>
                <span className="badge-violet">Fitness</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={genChartData}>
                    <defs>
                      <linearGradient id="fitGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#06B6D4" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                    <XAxis dataKey="label" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} domain={['auto', 100]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0B1020', borderColor: '#334155', borderRadius: 12, fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Area type="monotone" dataKey="best_fitness" name="Best Fitness" stroke="#06B6D4" fill="url(#fitGrad)" strokeWidth={2.5} />
                    <Line type="monotone" dataKey="avg_fitness" name="Population Mean Fitness" stroke="#10B981" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Pareto Frontier Section */}
          <div className="glass-card p-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
              <div>
                <h2 className="text-base font-semibold text-white">
                  Multi-Objective Pareto Frontier (Accuracy vs. Latency vs. Size)
                </h2>
                <p className="text-xs text-slate-400">
                  NSGA-II non-dominated solutions highlighted in violet/cyan. Bubble size reflects checkpoint size (KB).
                </p>
              </div>
              <Link to="/compare" className="text-xs text-cyan-400 hover:text-cyan-300 font-medium">
                Compare Models →
              </Link>
            </div>
            <ParetoChart points={summary?.pareto_points || []} height={300} />
          </div>
        </>
      )}

      {/* Experiments History Table */}
      {experiments.length > 0 && (
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-semibold text-white">Saved Experiments History</h2>
            <span className="text-xs text-slate-400 font-mono">{experiments.length} total in SQLite</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider">
                  <th className="py-3 pr-4">Experiment</th>
                  <th className="py-3 px-3">Dataset</th>
                  <th className="py-3 px-3">Mode / Strategy</th>
                  <th className="py-3 px-3">Progress</th>
                  <th className="py-3 px-3">Best Acc</th>
                  <th className="py-3 px-3">Best Lat</th>
                  <th className="py-3 px-3">Status</th>
                  <th className="py-3 pl-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {experiments.map((exp) => (
                  <tr key={exp.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-3 pr-4">
                      <p className="font-semibold text-white">{exp.name}</p>
                      <p className="text-[11px] font-mono text-slate-500">{exp.id}</p>
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300 uppercase">{exp.dataset}</td>
                    <td className="py-3 px-3 text-slate-300">
                      <span className="capitalize">{exp.optimization_mode}</span>
                      <span className="text-slate-500"> · {exp.search_strategy}</span>
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300">
                      Gen {exp.current_generation}/{exp.total_generations} ({exp.evaluated_count} eval)
                    </td>
                    <td className="py-3 px-3 font-mono text-emerald-400">{fmtPct(exp.best_accuracy)}</td>
                    <td className="py-3 px-3 font-mono text-cyan-400">{fmtMs(exp.best_latency_ms)}</td>
                    <td className="py-3 px-3">
                      <span className={STATUS_BADGE[exp.status] || 'badge-violet'}>{exp.status}</span>
                    </td>
                    <td className="py-3 pl-3 text-right">
                      <div className="inline-flex items-center gap-2">
                        <Link
                          to={`/monitor/${exp.id}`}
                          className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors"
                        >
                          Monitor
                        </Link>
                        <Link
                          to={`/evolution?exp=${exp.id}`}
                          className="px-2.5 py-1 rounded-lg bg-violet-500/15 hover:bg-violet-500/25 text-violet-300 transition-colors"
                        >
                          Lineage
                        </Link>
                        {exp.status !== 'running' && exp.status !== 'paused' && (
                          <button
                            onClick={() => handleDelete(exp.id)}
                            className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-950/40 transition-colors"
                            title="Delete experiment"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
