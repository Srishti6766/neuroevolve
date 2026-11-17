import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Activity,
  ArrowRight,
  Brain,
  Cpu,
  Database,
  GitBranch,
  Layers,
  Play,
  Scale,
  ShieldCheck,
  Sparkles,
  Target,
  Zap,
} from 'lucide-react';
import NeuralCanvas from '../components/NeuralCanvas';
import { api, fmtMs, fmtPct } from '../api/client';

const FEATURES = [
  {
    icon: GitBranch,
    title: 'Genetic Architecture Evolution',
    desc: 'Evolves neural topologies across generations using tournament selection, elitism, structural crossover, and layer-level mutations.',
    color: 'text-violet-400 bg-violet-500/15 border-violet-500/30',
  },
  {
    icon: Brain,
    title: 'RL-Guided Selection (UCB1 / Q-Learning)',
    desc: 'A lightweight contextual bandit learns which layer depths, widths, activations, and optimizers yield superior real-world rewards.',
    color: 'text-cyan-400 bg-cyan-500/15 border-cyan-500/30',
  },
  {
    icon: Target,
    title: 'Pareto Multi-Objective Optimization',
    desc: 'Simultaneously optimizes validation accuracy, CPU inference latency, and model size via NSGA-II non-dominated sorting.',
    color: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30',
  },
  {
    icon: Cpu,
    title: '100% Real PyTorch Computation',
    desc: 'Every candidate network is dynamically built, trained, benchmarked, and serialized in PyTorch on your local CPU — zero fabricated metrics.',
    color: 'text-amber-400 bg-amber-500/15 border-amber-500/30',
  },
  {
    icon: Layers,
    title: 'Interactive Topology & Lineage Explorer',
    desc: 'Inspect connected dense, normalization, activation, and dropout layers alongside a NetworkX evolutionary genealogy graph.',
    color: 'text-violet-300 bg-violet-500/15 border-violet-500/30',
  },
  {
    icon: ShieldCheck,
    title: '100% Free, Local & Persistent',
    desc: 'No API keys, no cloud GPUs, no paid subscriptions. All experiment history and PyTorch checkpoints persist locally in SQLite.',
    color: 'text-cyan-300 bg-cyan-500/15 border-cyan-500/30',
  },
];

const STEPS = [
  {
    step: '01',
    title: 'Define Search Space & Objective',
    desc: 'Select MNIST, Fashion-MNIST, Digits, or your own CSV. Configure layer bounds, neurons, activations, optimizers, and CPU limits.',
  },
  {
    step: '02',
    title: 'Generate & Train Real PyTorch Models',
    desc: 'NEUROEVOLVE constructs genuine nn.Module networks, trains them on CPU, and measures accuracy, F1, latency, memory, and file size.',
  },
  {
    step: '03',
    title: 'Evolve with GA + RL + Pareto Ranking',
    desc: 'Top architectures survive via elitism, recombine through crossover and mutation, and update a Q-learning UCB1 policy after every run.',
  },
  {
    step: '04',
    title: 'Compare, Inspect & Export Best Model',
    desc: 'Explore the Pareto frontier, compare candidates side-by-side, and download the winning .pt PyTorch weights and JSON configuration.',
  },
];

const TECH_STACK = [
  { name: 'PyTorch', role: 'Dynamic NN Training & Benchmarking' },
  { name: 'FastAPI', role: 'Async REST & WebSocket API' },
  { name: 'React + Vite', role: 'Reactive Research UI' },
  { name: 'Tailwind CSS', role: 'Dark Futuristic Design System' },
  { name: 'Recharts', role: 'Interactive Telemetry & Pareto Charts' },
  { name: 'Framer Motion', role: 'Fluid UI Transitions' },
  { name: 'NetworkX', role: 'Evolutionary Lineage Graph' },
  { name: 'Scikit-learn', role: 'Precision, Recall, F1 & Preprocessing' },
  { name: 'SQLite', role: 'Zero-Config Local Persistence' },
  { name: 'WebSockets', role: 'Live Epoch & Generation Streaming' },
];

export default function LandingPage() {
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    api.getDashboard().then(setSummary).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-[#080B14] text-slate-100 relative overflow-x-hidden">
      <NeuralCanvas />
      <div className="fixed inset-0 bg-grid-pattern bg-grid pointer-events-none z-0 opacity-60" />

      {/* Top Navigation */}
      <header className="relative z-10 border-b border-slate-800/70 bg-[#080B14]/75 backdrop-blur-xl sticky top-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-violet-600/30">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-extrabold tracking-wider text-base text-white">NEUROEVOLVE</span>
              <span className="hidden sm:inline-block ml-2 text-[11px] font-mono px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-300 border border-violet-500/30">
                NAS ENGINE
              </span>
            </div>
          </Link>

          <nav className="hidden lg:flex items-center gap-6 text-sm text-slate-300 whitespace-nowrap">
            <Link to="/dashboard" className="hover:text-white transition-colors">Dashboard</Link>
            <Link to="/configure" className="hover:text-white transition-colors">Configure</Link>
            <Link to="/monitor" className="hover:text-white transition-colors">Live Monitor</Link>
            <Link to="/explorer" className="hover:text-white transition-colors">Explorer</Link>
            <Link to="/evolution" className="hover:text-white transition-colors">Evolution</Link>
            <Link to="/best" className="hover:text-white transition-colors">Best Model</Link>
          </nav>

          <Link to="/configure" className="btn-primary text-xs sm:text-sm py-2 px-4">
            <Play className="w-3.5 h-3.5" />
            Start Experiment
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-20 lg:pt-28 lg:pb-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center max-w-4xl mx-auto"
        >
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-violet-500/10 border border-violet-500/30 text-violet-300 text-xs font-mono mb-6">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            SELF-EVOLVING NEURAL ARCHITECTURE SEARCH · LOCAL PYTORCH ENGINE
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight leading-[1.1] text-white">
            Let AI Discover{' '}
            <span className="bg-gradient-to-r from-violet-400 via-fuchsia-300 to-cyan-400 bg-clip-text text-transparent">
              Better Neural Networks.
            </span>
          </h1>

          <p className="mt-6 text-base sm:text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed">
            NEUROEVOLVE is an autonomous Neural Architecture Search platform that designs, trains,
            benchmarks, and evolves real PyTorch neural networks using genetic algorithms,
            reinforcement learning, and Pareto multi-objective optimization.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link to="/configure" className="btn-primary px-7 py-3.5 text-base shadow-violet-glow">
              <Play className="w-4 h-4" />
              Start Experiment
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link to="/explorer" className="btn-secondary px-7 py-3.5 text-base">
              <Layers className="w-4 h-4 text-cyan-400" />
              Explore Architectures
            </Link>
            <Link to="/dashboard" className="btn-secondary px-6 py-3.5 text-base">
              <Activity className="w-4 h-4 text-violet-400" />
              Open Dashboard
            </Link>
          </div>

          {/* Live Local Telemetry Strip (Strictly Real Backend Data) */}
          <div className="mt-14 grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-3xl mx-auto">
            <div className="glass-card p-4 text-center">
              <p className="text-[11px] uppercase tracking-wider text-slate-400">Experiments Run</p>
              <p className="mt-1 text-xl font-bold font-mono text-white">
                {summary ? summary.total_experiments : '—'}
              </p>
            </div>
            <div className="glass-card p-4 text-center">
              <p className="text-[11px] uppercase tracking-wider text-slate-400">Models Trained</p>
              <p className="mt-1 text-xl font-bold font-mono text-cyan-300">
                {summary ? summary.total_architectures_evaluated : '—'}
              </p>
            </div>
            <div className="glass-card p-4 text-center">
              <p className="text-[11px] uppercase tracking-wider text-slate-400">Best Accuracy</p>
              <p className="mt-1 text-xl font-bold font-mono text-emerald-400">
                {summary?.best_accuracy != null ? fmtPct(summary.best_accuracy) : '—'}
              </p>
            </div>
            <div className="glass-card p-4 text-center">
              <p className="text-[11px] uppercase tracking-wider text-slate-400">Fastest Latency</p>
              <p className="mt-1 text-xl font-bold font-mono text-violet-300">
                {summary?.best_latency_ms != null ? fmtMs(summary.best_latency_ms) : '—'}
              </p>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Feature Cards */}
      <section className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <span className="badge-violet mb-3">CORE CAPABILITIES</span>
          <h2 className="text-2xl sm:text-3xl font-bold text-white">
            Autonomous Research Infrastructure
          </h2>
          <p className="mt-2 text-slate-400 max-w-xl mx-auto text-sm">
            Built end-to-end with real PyTorch training, evolutionary biology operators, and live telemetry.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map((f, idx) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.06 }}
              className="glass-card-hover p-6"
            >
              <div className={`w-11 h-11 rounded-xl border flex items-center justify-center mb-4 ${f.color}`}>
                <f.icon className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-white">{f.title}</h3>
              <p className="mt-2 text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="glass-card p-8 lg:p-12 border-violet-500/20">
          <div className="text-center mb-10">
            <span className="badge-cyan mb-3">EVOLUTIONARY PIPELINE</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-white">How NEUROEVOLVE Works</h2>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {STEPS.map((s) => (
              <div key={s.step} className="relative p-5 rounded-2xl bg-[#080B14]/80 border border-slate-800">
                <span className="font-mono text-2xl font-extrabold bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
                  {s.step}
                </span>
                <h3 className="mt-3 font-semibold text-white text-sm">{s.title}</h3>
                <p className="mt-2 text-xs text-slate-400 leading-relaxed">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-10">
          <span className="badge-emerald mb-3">100% OPEN-SOURCE STACK</span>
          <h2 className="text-2xl sm:text-3xl font-bold text-white">Technology Stack</h2>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5">
          {TECH_STACK.map((t) => (
            <div
              key={t.name}
              className="glass-card p-4 text-center hover:border-cyan-500/40 transition-colors"
            >
              <p className="font-semibold text-sm text-white">{t.name}</p>
              <p className="mt-1 text-[11px] text-slate-400">{t.role}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-slate-800/80 py-8 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-violet-400" />
            <span>NEUROEVOLVE — Local Self-Evolving Neural Architecture Search Platform</span>
          </div>
          <div className="flex items-center gap-5">
            <Link to="/dashboard" className="hover:text-slate-300">Dashboard</Link>
            <Link to="/configure" className="hover:text-slate-300">Configure</Link>
            <Link to="/compare" className="hover:text-slate-300">Compare</Link>
            <Link to="/best" className="hover:text-slate-300">Best Model</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
