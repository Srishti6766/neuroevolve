import React, { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Brain, GitBranch, Play, Sparkles } from 'lucide-react';
import { api, fmtMs, fmtPct } from '../api/client';
import EvolutionGraph from '../components/EvolutionGraph';

export default function EvolutionPage() {
  const [searchParams] = useSearchParams();
  const queryExp = searchParams.get('exp') || '';

  const [experiments, setExperiments] = useState([]);
  const [selectedExp, setSelectedExp] = useState(queryExp);
  const [evolution, setEvolution] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listExperiments()
      .then((res) => {
        const list = res.experiments || [];
        setExperiments(list);
        if (!queryExp && list.length > 0) {
          setSelectedExp(list[0].id);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [queryExp]);

  useEffect(() => {
    if (!selectedExp) return;
    setLoading(true);
    api
      .getEvolution(selectedExp)
      .then((res) => {
        setEvolution(res);
        const nodes = res.graph?.nodes || [];
        setSelectedNode(nodes[nodes.length - 1] || null);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [selectedExp]);

  if (!loading && experiments.length === 0) {
    return (
      <div className="glass-card p-12 text-center space-y-4">
        <GitBranch className="w-10 h-10 text-violet-400 mx-auto" />
        <h1 className="text-2xl font-bold text-white">Evolution History</h1>
        <p className="text-sm text-slate-400 max-w-md mx-auto">
          Run a NAS experiment to visualize parent-to-child evolutionary lineages, mutations,
          crossovers, and RL policy updates across generations.
        </p>
        <Link to="/configure" className="btn-primary">
          <Play className="w-4 h-4" />
          Start NAS Experiment
        </Link>
      </div>
    );
  }

  const graph = evolution?.graph;
  const rlStats = evolution?.rl_stats;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Evolutionary Lineage & RL Policy
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Interactive NetworkX genealogy graph tracing parents, crossover, mutations, and RL bandit selections.
          </p>
        </div>

        <select
          value={selectedExp}
          onChange={(e) => setSelectedExp(e.target.value)}
          className="input-field w-auto min-w-[230px] py-2 text-xs"
        >
          {experiments.map((exp) => (
            <option key={exp.id} value={exp.id}>
              {exp.name} ({exp.dataset} · {exp.status})
            </option>
          ))}
        </select>
      </div>

      {/* Interactive Lineage Graph */}
      <div className="glass-card p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-white">Architecture Genealogy Graph</h2>
            <p className="text-xs text-slate-400">
              Click any node to highlight its parent/child evolutionary path and inspect exact mutation or crossover details.
            </p>
          </div>
          {graph?.stats && (
            <span className="badge-violet font-mono">
              {graph.stats.total_nodes} Models · {graph.stats.total_edges} Lineage Edges · {graph.stats.total_generations} Gens
            </span>
          )}
        </div>

        <EvolutionGraph
          graph={graph}
          selectedId={selectedNode?.id}
          onSelectNode={setSelectedNode}
        />

        {/* Selected Node Details Bar */}
        {selectedNode && (
          <div className="p-4 rounded-xl bg-[#0B1020] border border-violet-500/30 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono font-bold text-white text-sm">{selectedNode.name}</span>
                <span className="badge-violet capitalize">{selectedNode.origin}</span>
                <span className="badge-cyan font-mono">Gen {selectedNode.generation}</span>
                {selectedNode.is_pareto_optimal && <span className="badge-emerald">Pareto Frontier</span>}
              </div>
              <p className="text-xs text-slate-300 font-mono">
                Operation: {selectedNode.operation_details || 'Initial sampling'}
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs font-mono shrink-0">
              <div>
                <span className="text-slate-500 block">Accuracy</span>
                <span className="text-emerald-400 font-semibold">{fmtPct(selectedNode.accuracy)}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Latency</span>
                <span className="text-cyan-400 font-semibold">{fmtMs(selectedNode.inference_latency_ms)}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Fitness</span>
                <span className="text-violet-300 font-semibold">
                  {selectedNode.fitness_score?.toFixed(2) ?? '—'}
                </span>
              </div>
              <Link
                to={`/explorer?arch=${selectedNode.id}`}
                className="btn-secondary py-1.5 px-3 text-xs font-sans"
              >
                Inspect Layers →
              </Link>
            </div>
          </div>
        )}
      </div>

      {/* Generation Breakdown Cards */}
      {graph?.generations?.length > 0 && (
        <div className="glass-card p-5 space-y-4">
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            Best Architecture & Operators Per Generation
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {graph.generations.map((g) => (
              <div key={g.generation} className="p-4 rounded-xl bg-[#0B1020] border border-slate-800 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm font-bold text-white">Generation {g.generation}</span>
                  <span className="badge-emerald font-mono">Best {fmtPct(g.best_accuracy, 1)}</span>
                </div>
                <p className="text-xs font-mono text-cyan-300 truncate">
                  ★ {g.best_name || 'Pending'}
                </p>
                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400">
                  <span>Best Fitness: {g.best_fitness?.toFixed(2) ?? '—'}</span>
                  <span>Avg Fitness: {g.avg_fitness?.toFixed(2) ?? '—'}</span>
                </div>
                <div className="pt-2 border-t border-slate-800 flex flex-wrap gap-1.5 text-[10px]">
                  {Object.entries(g.origins || {})
                    .filter(([, count]) => count > 0)
                    .map(([op, count]) => (
                      <span
                        key={op}
                        className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 capitalize font-mono"
                      >
                        {op}: {count}
                      </span>
                    ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Reinforcement Learning Policy Insights */}
      {rlStats && (
        <div className="glass-card p-5 space-y-4 border-emerald-500/25">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-white flex items-center gap-2">
                <Brain className="w-4 h-4 text-emerald-400" />
                Reinforcement Learning Controller Insights
              </h2>
              <p className="text-xs text-slate-400">
                {rlStats.algorithm} — Learned Q-values updated from real PyTorch evaluation rewards.
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono">
              <span className="badge-emerald">Updates: {rlStats.total_updates}</span>
              <span className="badge-cyan">Epsilon: {rlStats.current_epsilon}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {Object.entries(rlStats.top_preferences || {}).map(([gene, info]) => (
              <div key={gene} className="p-3.5 rounded-xl bg-[#0B1020] border border-slate-800">
                <p className="text-[10px] uppercase tracking-wider text-slate-400">
                  {gene.replace('_', ' ')}
                </p>
                <p className="mt-1 text-base font-bold font-mono text-emerald-300 uppercase">
                  {info.best_choice}
                </p>
                <p className="mt-1 text-[11px] font-mono text-slate-400">
                  Q = {info.q_value} · {info.trials} trials
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
