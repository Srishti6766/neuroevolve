import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  Activity,
  Award,
  Clock,
  Cpu,
  HardDrive,
  Pause,
  Play,
  Square,
  Terminal,
  Wifi,
  WifiOff,
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
  createExperimentSocket,
  fmtMs,
  fmtParams,
  fmtPct,
  fmtSizeKb,
} from '../api/client';
import ArchitectureGraph from '../components/ArchitectureGraph';

export default function MonitorPage() {
  const { id: routeId } = useParams();
  const navigate = useNavigate();

  const [experiments, setExperiments] = useState([]);
  const [expId, setExpId] = useState(routeId || '');
  const [experiment, setExperiment] = useState(null);
  const [architectures, setArchitectures] = useState([]);
  const [logs, setLogs] = useState([]);
  const [liveState, setLiveState] = useState(null);
  const [liveEpochCurve, setLiveEpochCurve] = useState([]);
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [actionBusy, setActionBusy] = useState(false);
  const [error, setError] = useState('');
  const logEndRef = useRef(null);

  // Sync route param
  useEffect(() => {
    if (routeId) setExpId(routeId);
  }, [routeId]);

  // Load experiment list and auto-select active or latest
  useEffect(() => {
    api.listExperiments().then((res) => {
      const list = res.experiments || [];
      setExperiments(list);
      if (!routeId && !expId && list.length > 0) {
        const target = res.active_experiment_id || list[0].id;
        setExpId(target);
      }
    }).catch(() => {});
  }, [routeId, expId]);

  const fetchDetail = useCallback(async (targetId) => {
    if (!targetId) return;
    try {
      setError('');
      const res = await api.getExperiment(targetId);
      setExperiment(res.experiment);
      setArchitectures(res.architectures || []);
      setLogs(res.logs || []);
      if (res.live_state) setLiveState(res.live_state);
      // Keep experiment selector dropdown status in sync
      setExperiments((prev) =>
        prev.map((item) => (item.id === res.experiment.id ? { ...item, status: res.experiment.status } : item))
      );
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    if (expId) fetchDetail(expId);
  }, [expId, fetchDetail]);

  // WebSocket streaming for selected experiment
  useEffect(() => {
    if (!expId) return;
    const disconnect = createExperimentSocket(
      expId,
      (event) => {
        if (event.type === 'log') {
          setLogs((prev) => [...prev.slice(-249), event.data]);
        } else if (event.type === 'candidate_started') {
          setLiveEpochCurve([]);
          if (event.data.live) setLiveState(event.data.live);
          fetchDetail(expId);
        } else if (event.type === 'epoch_completed') {
          if (event.data.live) setLiveState(event.data.live);
          const ep = event.data.epoch_metrics;
          if (ep) {
            setLiveEpochCurve((prev) => [
              ...prev.filter((p) => p.epoch !== ep.epoch),
              {
                epoch: ep.epoch,
                train_loss: ep.train_loss,
                val_loss: ep.val_loss,
                val_acc_pct: Number((ep.val_accuracy * 100).toFixed(2)),
                train_acc_pct: Number((ep.train_accuracy * 100).toFixed(2)),
              },
            ]);
          }
        } else if (
          [
            'candidate_evaluated',
            'generation_started',
            'generation_completed',
            'experiment_paused',
            'experiment_resumed',
            'experiment_stopped',
            'experiment_completed',
            'experiment_failed',
          ].includes(event.type)
        ) {
          if (event.data?.live) setLiveState(event.data.live);
          fetchDetail(expId);
        }
      },
      setWsStatus
    );

    // Fallback polling while running/paused (faster when WebSocket is disconnected on serverless hosts)
    const pollMs = wsStatus === 'connected' ? 4000 : 2500;
    const poll = setInterval(() => {
      if (experiment?.status === 'running' || experiment?.status === 'paused' || experiment?.status === 'queued') {
        fetchDetail(expId);
      }
    }, pollMs);

    return () => {
      disconnect();
      clearInterval(poll);
    };
  }, [expId, experiment?.status, wsStatus, fetchDetail]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs.length]);

  const handleControl = async (action) => {
    if (!expId) return;
    setActionBusy(true);
    setError('');
    try {
      if (action === 'pause') await api.pauseExperiment(expId);
      else if (action === 'resume') await api.resumeExperiment(expId);
      else if (action === 'stop') await api.stopExperiment(expId);
      await fetchDetail(expId);
    } catch (err) {
      setError(err.message);
    } finally {
      setActionBusy(false);
    }
  };

  if (!expId && experiments.length === 0) {
    return (
      <div className="glass-card p-12 text-center space-y-4">
        <Activity className="w-10 h-10 text-violet-400 mx-auto" />
        <h1 className="text-2xl font-bold text-white">Live Experiment Monitor</h1>
        <p className="text-sm text-slate-400 max-w-md mx-auto">
          No experiments found yet. Configure and launch a PyTorch NAS experiment to watch candidate
          networks train and evolve live.
        </p>
        <Link to="/configure" className="btn-primary">
          <Play className="w-4 h-4" />
          Start New Experiment
        </Link>
      </div>
    );
  }

  const evaluatedArchs = architectures.filter((a) => a.status === 'evaluated');
  const rankedArchs = [...evaluatedArchs].sort(
    (a, b) => (b.fitness_score || 0) - (a.fitness_score || 0)
  );
  const totalPlanned = (experiment?.total_generations || 1) * (experiment?.population_size || 1);
  const progressPct = Math.min(100, Math.round((evaluatedArchs.length / Math.max(1, totalPlanned)) * 100));

  const currentCand = liveState?.current_candidate;
  const latestEp = liveState?.latest_epoch_metrics;
  const maxEpochs = experiment?.config?.max_epochs || liveState?.max_epochs || 3;
  const epochPct = latestEp ? Math.round((latestEp.epoch / Math.max(1, maxEpochs)) * 100) : 0;

  const displayArch =
    currentCand ||
    rankedArchs[0] ||
    architectures[architectures.length - 1] ||
    null;

  const isRunning = experiment?.status === 'running';
  const isPaused = experiment?.status === 'paused';

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Live Experiment Monitor
            </h1>
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono border ${
                wsStatus === 'connected'
                  ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              {wsStatus === 'connected' ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
              {wsStatus === 'connected' ? 'WS LIVE' : 'POLLING'}
            </span>
            {experiment && (
              <span className="badge-violet uppercase font-mono">{experiment.status}</span>
            )}
          </div>
          {experiment && (
            <p className="text-xs font-mono text-slate-400 mt-1">
              {experiment.name} ({experiment.id}) · Dataset: {experiment.dataset} · Mode: {experiment.optimization_mode} · Strategy: {experiment.search_strategy}
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <select
            value={expId}
            onChange={(e) => {
              setExpId(e.target.value);
              navigate(`/monitor/${e.target.value}`);
            }}
            className="input-field w-auto min-w-[200px] py-2 text-xs"
          >
            {experiments.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name} ({e.status})
              </option>
            ))}
          </select>

          {isRunning && (
            <button
              onClick={() => handleControl('pause')}
              disabled={actionBusy}
              className="btn-secondary py-2 px-3.5 text-xs"
            >
              <Pause className="w-3.5 h-3.5 text-amber-400" />
              Pause
            </button>
          )}

          {isPaused && (
            <button
              onClick={() => handleControl('resume')}
              disabled={actionBusy}
              className="btn-primary py-2 px-3.5 text-xs"
            >
              <Play className="w-3.5 h-3.5" />
              Resume
            </button>
          )}

          {(isRunning || isPaused) && (
            <button
              onClick={() => handleControl('stop')}
              disabled={actionBusy}
              className="btn-danger py-2 px-3.5 text-xs"
            >
              <Square className="w-3.5 h-3.5" />
              Stop Safely
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-700/60 text-rose-200 text-sm">
          {error}
        </div>
      )}

      {experiment?.error_message && experiment.status === 'failed' && (
        <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-700/60 text-rose-200 text-sm">
          Experiment Error: {experiment.error_message}
        </div>
      )}

      {/* Progress & Live Metrics Bar */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3.5">
        <div className="glass-card p-4">
          <p className="text-[11px] text-slate-400 uppercase">Generation</p>
          <p className="mt-1 text-xl font-bold font-mono text-white">
            {experiment?.current_generation ?? 0} / {experiment?.total_generations ?? 0}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[11px] text-slate-400 uppercase">Current Accuracy</p>
          <p className="mt-1 text-xl font-bold font-mono text-emerald-400">
            {latestEp ? fmtPct(latestEp.val_accuracy) : fmtPct(rankedArchs[0]?.accuracy)}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[11px] text-slate-400 uppercase">Validation Loss</p>
          <p className="mt-1 text-xl font-bold font-mono text-amber-300">
            {latestEp
              ? latestEp.val_loss.toFixed(4)
              : rankedArchs[0]?.val_loss != null
              ? rankedArchs[0].val_loss.toFixed(4)
              : '—'}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[11px] text-slate-400 uppercase">Best Latency</p>
          <p className="mt-1 text-xl font-bold font-mono text-cyan-400">
            {fmtMs(experiment?.best_latency_ms)}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[11px] text-slate-400 uppercase">Peak Memory</p>
          <p className="mt-1 text-xl font-bold font-mono text-violet-300">
            {rankedArchs[0]?.peak_memory_mb != null ? `${rankedArchs[0].peak_memory_mb} MB` : '—'}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[11px] text-slate-400 uppercase">Evaluated</p>
          <p className="mt-1 text-xl font-bold font-mono text-white">
            {evaluatedArchs.length} / {totalPlanned}
          </p>
        </div>
      </div>

      {/* Dual Progress Bars */}
      <div className="glass-card p-5 space-y-4">
        <div>
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-slate-300 font-medium">Overall Experiment Progress</span>
            <span className="font-mono text-cyan-300">{progressPct}%</span>
          </div>
          <div className="h-2.5 w-full rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-violet-600 to-cyan-500 transition-all duration-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>

        {currentCand && (
          <div>
            <div className="flex justify-between text-xs mb-1.5">
              <span className="text-slate-300 font-medium">
                Training Candidate: <span className="font-mono text-violet-300">{currentCand.name}</span>{' '}
                ({currentCand.origin})
              </span>
              <span className="font-mono text-emerald-300">
                Epoch {latestEp?.epoch || 0} / {maxEpochs} ({epochPct}%)
              </span>
            </div>
            <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-300"
                style={{ width: `${epochPct}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Live Candidate Visualization & Epoch Curve */}
      <div className="grid lg:grid-cols-2 gap-5">
        <div className="glass-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white">
                {currentCand ? 'Active Candidate Architecture' : 'Top Discovered Architecture'}
              </h2>
              <p className="text-xs text-slate-400 font-mono">
                {displayArch ? `${displayArch.name} · ${displayArch.operation_details || displayArch.origin}` : 'Waiting for candidate...'}
              </p>
            </div>
            {displayArch && (
              <span className="badge-cyan font-mono">
                {displayArch.config?.optimizer?.toUpperCase()} lr={displayArch.config?.learning_rate}
              </span>
            )}
          </div>
          <ArchitectureGraph layers={displayArch?.layer_summary || []} compact />
        </div>

        <div className="glass-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white">Real-Time Epoch Training Curve</h2>
              <p className="text-xs text-slate-400">
                {liveEpochCurve.length > 0
                  ? 'Streaming PyTorch epoch loss & validation accuracy'
                  : 'Showing best candidate training history'}
              </p>
            </div>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>

          <div className="h-64">
            {(() => {
              const curve =
                liveEpochCurve.length > 0
                  ? liveEpochCurve
                  : (rankedArchs[0]?.training_history || []).map((ep) => ({
                      epoch: ep.epoch,
                      train_loss: ep.train_loss,
                      val_loss: ep.val_loss,
                      val_acc_pct: Number((ep.val_accuracy * 100).toFixed(2)),
                    }));
              if (curve.length === 0) {
                return (
                  <div className="h-full flex items-center justify-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
                    Waiting for first epoch to complete...
                  </div>
                );
              }
              return (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={curve}>
                    <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="loss" stroke="#F59E0B" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="acc" orientation="right" stroke="#10B981" tick={{ fontSize: 11 }} unit="%" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0B1020', borderColor: '#334155', borderRadius: 12, fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line yAxisId="loss" type="monotone" dataKey="train_loss" name="Train Loss" stroke="#8B5CF6" strokeWidth={2} />
                    <Line yAxisId="loss" type="monotone" dataKey="val_loss" name="Val Loss" stroke="#F59E0B" strokeWidth={2} />
                    <Line yAxisId="acc" type="monotone" dataKey="val_acc_pct" name="Val Acc (%)" stroke="#10B981" strokeWidth={2.5} />
                  </LineChart>
                </ResponsiveContainer>
              );
            })()}
          </div>
        </div>
      </div>

      {/* Candidate Ranking Table & Live Logs */}
      <div className="grid xl:grid-cols-3 gap-5">
        <div className="xl:col-span-2 glass-card p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Award className="w-4 h-4 text-amber-400" />
              Live Candidate Ranking ({rankedArchs.length} Evaluated)
            </h2>
            <Link to={`/best?exp=${expId}`} className="text-xs text-cyan-400 hover:text-cyan-300">
              Inspect Best Model →
            </Link>
          </div>

          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-[#111827]">
                <tr className="border-b border-slate-800 text-slate-400 uppercase">
                  <th className="py-2.5 pr-2">#</th>
                  <th className="py-2.5 px-2">Architecture</th>
                  <th className="py-2.5 px-2">Origin</th>
                  <th className="py-2.5 px-2">Acc</th>
                  <th className="py-2.5 px-2">Val Loss</th>
                  <th className="py-2.5 px-2">Latency</th>
                  <th className="py-2.5 px-2">Size</th>
                  <th className="py-2.5 px-2">Params</th>
                  <th className="py-2.5 pl-2 text-right">Fitness</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 font-mono">
                {rankedArchs.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="py-8 text-center text-slate-500 font-sans">
                      Candidates will appear here as soon as PyTorch completes evaluation.
                    </td>
                  </tr>
                ) : (
                  rankedArchs.map((a, idx) => (
                    <tr key={a.id} className="hover:bg-slate-800/35">
                      <td className="py-2.5 pr-2 text-slate-400">{idx + 1}</td>
                      <td className="py-2.5 px-2">
                        <Link to={`/explorer?arch=${a.id}`} className="text-white hover:text-cyan-300 font-semibold">
                          {a.name}
                        </Link>
                        {a.is_pareto_optimal && (
                          <span className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300">
                            Pareto
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-2 font-sans capitalize text-slate-300">{a.origin}</td>
                      <td className="py-2.5 px-2 text-emerald-400">{fmtPct(a.accuracy)}</td>
                      <td className="py-2.5 px-2 text-amber-300">{a.val_loss?.toFixed(4) ?? '—'}</td>
                      <td className="py-2.5 px-2 text-cyan-400">{fmtMs(a.inference_latency_ms)}</td>
                      <td className="py-2.5 px-2 text-slate-300">{fmtSizeKb(a.model_size_kb)}</td>
                      <td className="py-2.5 px-2 text-slate-400">{fmtParams(a.parameter_count)}</td>
                      <td className="py-2.5 pl-2 text-right font-bold text-violet-300">
                        {a.fitness_score?.toFixed(2) ?? '—'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Experiment Logs Console */}
        <div className="glass-card p-5 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Terminal className="w-4 h-4 text-violet-400" />
              Experiment Logs
            </h2>
            <span className="text-[11px] font-mono text-slate-500">{logs.length} events</span>
          </div>

          <div className="flex-1 bg-[#070A12] border border-slate-800 rounded-xl p-3 font-mono text-[11px] space-y-1.5 overflow-y-auto max-h-96 min-h-[260px]">
            {logs.length === 0 ? (
              <p className="text-slate-500">Waiting for backend engine logs...</p>
            ) : (
              logs.map((log, i) => {
                const color =
                  log.level === 'success'
                    ? 'text-emerald-300'
                    : log.level === 'warning'
                    ? 'text-amber-300'
                    : log.level === 'error'
                    ? 'text-rose-400'
                    : 'text-slate-300';
                const ts = log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '';
                return (
                  <div key={log.id || i} className="leading-relaxed break-words">
                    <span className="text-slate-600 mr-1.5">[{ts}]</span>
                    <span className={color}>{log.message}</span>
                  </div>
                );
              })
            )}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
