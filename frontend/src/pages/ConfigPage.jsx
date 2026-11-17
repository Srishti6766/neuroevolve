import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  Brain,
  CheckCircle2,
  Cpu,
  Database,
  Gauge,
  Play,
  Sliders,
  Sparkles,
  Target,
  Upload,
  Zap,
} from 'lucide-react';
import { api } from '../api/client';

const NEURON_OPTIONS = [32, 64, 128, 256, 384, 512];
const ACTIVATION_OPTIONS = ['relu', 'gelu', 'tanh', 'silu', 'leaky_relu'];
const DROPOUT_OPTIONS = [0.0, 0.1, 0.2, 0.3, 0.4];
const OPTIMIZER_OPTIONS = ['adam', 'adamw', 'sgd'];
const LR_OPTIONS = [0.0005, 0.001, 0.003, 0.005, 0.01];

const OPT_MODES = [
  {
    id: 'accuracy',
    title: 'Accuracy-Focused',
    desc: 'Prioritizes maximum validation accuracy (85% accuracy, 8% speed, 7% compactness).',
    icon: Target,
    accent: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300',
  },
  {
    id: 'speed',
    title: 'Speed-Focused',
    desc: 'Discovers ultra-fast, low-latency, compact architectures (35% acc, 45% speed, 20% size).',
    icon: Zap,
    accent: 'border-cyan-500/50 bg-cyan-500/10 text-cyan-300',
  },
  {
    id: 'balanced',
    title: 'Balanced Multi-Objective',
    desc: 'Pareto-aware optimization balancing accuracy, CPU latency, and model footprint.',
    icon: Gauge,
    accent: 'border-violet-500/50 bg-violet-500/10 text-violet-300',
  },
];

const STRATEGIES = [
  {
    id: 'hybrid_rl',
    title: 'Hybrid GA + RL (Recommended)',
    desc: 'Combines genetic elitism, crossover & mutation with a UCB1/Q-Learning bandit controller.',
  },
  {
    id: 'genetic',
    title: 'Pure Genetic Evolution',
    desc: 'Classic evolutionary NAS with tournament selection, elitism, crossover, and mutation.',
  },
  {
    id: 'rl_guided',
    title: 'RL-Guided Bandit Search',
    desc: 'Reinforcement learning controller proposes candidates directly from learned Q-values.',
  },
];

export default function ConfigPage() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [maxCpuThreads, setMaxCpuThreads] = useState(8);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  // CSV Upload state
  const [csvFile, setCsvFile] = useState(null);
  const [csvName, setCsvName] = useState('');
  const [csvTarget, setCsvTarget] = useState('');
  const [uploadingCsv, setUploadingCsv] = useState(false);
  const [csvSuccess, setCsvSuccess] = useState('');

  // Experiment Form State (CPU-friendly Ryzen 5 defaults)
  const [form, setForm] = useState({
    name: 'MNIST Evolutionary Search',
    dataset: 'mnist',
    generations: 3,
    population_size: 4,
    max_epochs: 3,
    batch_size: 64,
    max_train_samples: 4000,
    max_val_samples: 1000,
    optimization_mode: 'balanced',
    search_strategy: 'hybrid_rl',
    cpu_threads: 4,
    elitism_count: 1,
    search_space: {
      min_layers: 1,
      max_layers: 3,
      neuron_choices: [64, 128, 256],
      activations: ['relu', 'gelu', 'tanh', 'silu'],
      dropout_choices: [0.0, 0.1, 0.2],
      optimizers: ['adam', 'adamw'],
      learning_rates: [0.001, 0.003],
      batch_norm_choices: [false, true],
    },
  });

  useEffect(() => {
    Promise.all([api.getDatasets(), api.getDefaults()])
      .then(([dsRes, defRes]) => {
        setDatasets(dsRes.datasets || []);
        if (defRes.max_cpu_threads) {
          setMaxCpuThreads(defRes.max_cpu_threads);
          setForm((prev) => ({
            ...prev,
            cpu_threads: Math.min(4, defRes.max_cpu_threads),
          }));
        }
      })
      .catch(() => {});
  }, []);

  const toggleArrayChoice = (field, value) => {
    setForm((prev) => {
      const curr = prev.search_space[field];
      const exists = curr.includes(value);
      if (exists && curr.length <= 1) return prev; // keep at least 1 option
      const next = exists ? curr.filter((v) => v !== value) : [...curr, value];
      return {
        ...prev,
        search_space: { ...prev.search_space, [field]: next },
      };
    });
  };

  const applyPreset = (preset) => {
    if (preset === 'quick') {
      setForm((p) => ({
        ...p,
        name: 'Quick CPU Discovery Run',
        generations: 3,
        population_size: 4,
        max_epochs: 2,
        max_train_samples: 2500,
        max_val_samples: 800,
      }));
    } else if (preset === 'balanced') {
      setForm((p) => ({
        ...p,
        name: 'Ryzen 5 Balanced NAS Run',
        generations: 4,
        population_size: 5,
        max_epochs: 4,
        max_train_samples: 5000,
        max_val_samples: 1500,
      }));
    } else if (preset === 'thorough') {
      setForm((p) => ({
        ...p,
        name: 'Deep Architecture Search',
        generations: 6,
        population_size: 6,
        max_epochs: 6,
        max_train_samples: 12000,
        max_val_samples: 2500,
      }));
    }
  };

  const handleCsvUpload = async (e) => {
    e.preventDefault();
    if (!csvFile) return;
    setUploadingCsv(true);
    setError('');
    setCsvSuccess('');
    try {
      const res = await api.uploadDataset(csvFile, csvName, csvTarget);
      const dsRes = await api.getDatasets();
      setDatasets(dsRes.datasets || []);
      const newId = `csv:${res.dataset.id}`;
      setForm((p) => ({ ...p, dataset: newId, name: `${res.dataset.name} NAS Search` }));
      setCsvSuccess(`Uploaded "${res.dataset.name}" (${res.dataset.num_samples} rows, ${res.dataset.num_features} features, ${res.dataset.num_classes} classes).`);
      setCsvFile(null);
      setCsvName('');
      setCsvTarget('');
    } catch (err) {
      setError(err.message);
    } finally {
      setUploadingCsv(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!form.name.trim()) {
      setError('Please enter an experiment name.');
      return;
    }
    if (form.search_space.min_layers > form.search_space.max_layers) {
      setError('Minimum hidden layers cannot exceed maximum hidden layers.');
      return;
    }

    setSubmitting(true);
    try {
      const res = await api.createExperiment(form);
      navigate(`/monitor/${res.experiment.id}`);
    } catch (err) {
      setError(err.message || 'Failed to launch NAS experiment.');
    } finally {
      setSubmitting(false);
    }
  };

  const totalModels = form.generations * form.population_size;

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Configure NAS Experiment
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Define your dataset, evolutionary budget, neural search space, optimization mode, and CPU limits.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={() => applyPreset('quick')} className="btn-secondary text-xs py-1.5 px-3">
            Fast CPU (~15s)
          </button>
          <button type="button" onClick={() => applyPreset('balanced')} className="btn-secondary text-xs py-1.5 px-3">
            Balanced Ryzen 5
          </button>
          <button type="button" onClick={() => applyPreset('thorough')} className="btn-secondary text-xs py-1.5 px-3">
            Deep Search
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-700/60 text-rose-200 text-sm flex items-center gap-2.5">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Section 1: Experiment Identity & Dataset */}
        <div className="glass-card p-6 space-y-5">
          <div className="flex items-center gap-2 text-white font-semibold">
            <Database className="w-4 h-4 text-cyan-400" />
            <h2>1. Experiment & Dataset</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-5">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Experiment Name</label>
              <input
                type="text"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                className="input-field"
                placeholder="e.g., MNIST Balanced Evolution"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Dataset</label>
              <select
                value={form.dataset}
                onChange={(e) => setForm({ ...form, dataset: e.target.value })}
                className="input-field"
              >
                {datasets.length === 0 ? (
                  <>
                    <option value="mnist">MNIST Handwritten Digits (Default)</option>
                    <option value="fashion_mnist">Fashion-MNIST</option>
                    <option value="digits">UCI Handwritten Digits (8x8 Offline)</option>
                  </>
                ) : (
                  datasets.map((ds) => (
                    <option key={ds.id} value={ds.id}>
                      {ds.name} ({ds.input_dim} features · {ds.num_classes} classes)
                    </option>
                  ))
                )}
              </select>
            </div>
          </div>

          {/* Optional Custom CSV Upload */}
          <div className="p-4 rounded-xl bg-[#0B1020] border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-300 flex items-center gap-1.5">
                <Upload className="w-3.5 h-3.5 text-violet-400" />
                Optional: Upload Compatible Classification CSV Dataset
              </span>
              <span className="text-[11px] text-slate-500">Numeric features + target label column</span>
            </div>
            <div className="grid sm:grid-cols-4 gap-2.5 items-center">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setCsvFile(e.target.files?.[0] || null)}
                className="text-xs text-slate-300 file:mr-2 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 sm:col-span-2"
              />
              <input
                type="text"
                value={csvTarget}
                onChange={(e) => setCsvTarget(e.target.value)}
                placeholder="Target col (default: last)"
                className="input-field py-1.5 text-xs"
              />
              <button
                type="button"
                onClick={handleCsvUpload}
                disabled={!csvFile || uploadingCsv}
                className="btn-secondary py-1.5 text-xs"
              >
                {uploadingCsv ? 'Uploading...' : 'Upload CSV'}
              </button>
            </div>
            {csvSuccess && (
              <p className="text-xs text-emerald-400 flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5" />
                {csvSuccess}
              </p>
            )}
          </div>
        </div>

        {/* Section 2: Optimization Objective & Search Strategy */}
        <div className="glass-card p-6 space-y-5">
          <div className="flex items-center gap-2 text-white font-semibold">
            <Sparkles className="w-4 h-4 text-violet-400" />
            <h2>2. Optimization Mode & Search Strategy</h2>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2.5">
              Optimization Objective
            </label>
            <div className="grid md:grid-cols-3 gap-3.5">
              {OPT_MODES.map((m) => {
                const selected = form.optimization_mode === m.id;
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setForm({ ...form, optimization_mode: m.id })}
                    className={`text-left p-4 rounded-xl border transition-all ${
                      selected
                        ? m.accent
                        : 'border-slate-800 bg-[#0B1020] text-slate-300 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center gap-2 font-semibold text-sm">
                      <m.icon className="w-4 h-4" />
                      {m.title}
                    </div>
                    <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">{m.desc}</p>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-2.5">
              Search Algorithm
            </label>
            <div className="grid md:grid-cols-3 gap-3.5">
              {STRATEGIES.map((s) => {
                const selected = form.search_strategy === s.id;
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => setForm({ ...form, search_strategy: s.id })}
                    className={`text-left p-4 rounded-xl border transition-all ${
                      selected
                        ? 'border-cyan-500/50 bg-cyan-500/10 text-white'
                        : 'border-slate-800 bg-[#0B1020] text-slate-300 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center gap-2 font-semibold text-sm">
                      <Brain className="w-4 h-4 text-cyan-400" />
                      {s.title}
                    </div>
                    <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">{s.desc}</p>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Section 3: Evolutionary Budget & CPU Limits */}
        <div className="glass-card p-6 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <h2>3. Evolution Parameters & CPU Resource Limits</h2>
            </div>
            <span className="badge-violet font-mono">
              Total Evaluations: {totalModels} PyTorch Models
            </span>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            <div>
              <label className="flex justify-between text-xs text-slate-300 mb-1.5">
                <span>Generations</span>
                <span className="font-mono text-violet-300">{form.generations}</span>
              </label>
              <input
                type="range"
                min={1}
                max={15}
                value={form.generations}
                onChange={(e) => setForm({ ...form, generations: Number(e.target.value) })}
                className="w-full accent-violet-500"
              />
            </div>

            <div>
              <label className="flex justify-between text-xs text-slate-300 mb-1.5">
                <span>Population Size (per Gen)</span>
                <span className="font-mono text-cyan-300">{form.population_size}</span>
              </label>
              <input
                type="range"
                min={2}
                max={12}
                value={form.population_size}
                onChange={(e) => setForm({ ...form, population_size: Number(e.target.value) })}
                className="w-full accent-cyan-500"
              />
            </div>

            <div>
              <label className="flex justify-between text-xs text-slate-300 mb-1.5">
                <span>Max Training Epochs / Model</span>
                <span className="font-mono text-emerald-300">{form.max_epochs}</span>
              </label>
              <input
                type="range"
                min={1}
                max={15}
                value={form.max_epochs}
                onChange={(e) => setForm({ ...form, max_epochs: Number(e.target.value) })}
                className="w-full accent-emerald-500"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300 mb-1.5">Batch Size</label>
              <select
                value={form.batch_size}
                onChange={(e) => setForm({ ...form, batch_size: Number(e.target.value) })}
                className="input-field"
              >
                {[32, 64, 128, 256].map((b) => (
                  <option key={b} value={b}>{b}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs text-slate-300 mb-1.5">
                Training Samples Cap (CPU Friendly)
              </label>
              <select
                value={form.max_train_samples}
                onChange={(e) =>
                  setForm({
                    ...form,
                    max_train_samples: Number(e.target.value),
                    max_val_samples: Math.max(400, Math.round(Number(e.target.value) * 0.25)),
                  })
                }
                className="input-field"
              >
                <option value={1500}>1,500 train / 400 val (Ultra-fast)</option>
                <option value={4000}>4,000 train / 1,000 val (Recommended CPU)</option>
                <option value={10000}>10,000 train / 2,500 val (High precision)</option>
                <option value={30000}>30,000 train / 7,500 val (Heavy CPU)</option>
                <option value={60000}>60,000 Full Dataset</option>
              </select>
            </div>

            <div>
              <label className="flex justify-between text-xs text-slate-300 mb-1.5">
                <span>PyTorch CPU Threads Limit</span>
                <span className="font-mono text-amber-300">
                  {form.cpu_threads} / {maxCpuThreads} threads
                </span>
              </label>
              <input
                type="range"
                min={1}
                max={Math.max(1, maxCpuThreads)}
                value={form.cpu_threads}
                onChange={(e) => setForm({ ...form, cpu_threads: Number(e.target.value) })}
                className="w-full accent-amber-500"
              />
            </div>
          </div>
        </div>

        {/* Section 4: Neural Network Search Space */}
        <div className="glass-card p-6 space-y-5">
          <div className="flex items-center gap-2 text-white font-semibold">
            <Sliders className="w-4 h-4 text-violet-400" />
            <h2>4. Neural Network Search Space</h2>
          </div>

          <div className="grid sm:grid-cols-2 gap-5">
            <div>
              <label className="flex justify-between text-xs text-slate-300 mb-1.5">
                <span>Minimum Hidden Layers</span>
                <span className="font-mono text-violet-300">{form.search_space.min_layers}</span>
              </label>
              <input
                type="range"
                min={1}
                max={5}
                value={form.search_space.min_layers}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setForm((p) => ({
                    ...p,
                    search_space: {
                      ...p.search_space,
                      min_layers: v,
                      max_layers: Math.max(v, p.search_space.max_layers),
                    },
                  }));
                }}
                className="w-full accent-violet-500"
              />
            </div>

            <div>
              <label className="flex justify-between text-xs text-slate-300 mb-1.5">
                <span>Maximum Hidden Layers</span>
                <span className="font-mono text-cyan-300">{form.search_space.max_layers}</span>
              </label>
              <input
                type="range"
                min={form.search_space.min_layers}
                max={6}
                value={form.search_space.max_layers}
                onChange={(e) =>
                  setForm((p) => ({
                    ...p,
                    search_space: { ...p.search_space, max_layers: Number(e.target.value) },
                  }))
                }
                className="w-full accent-cyan-500"
              />
            </div>
          </div>

          {/* Multi-select Toggle Chips */}
          <div className="grid md:grid-cols-2 gap-5">
            <div>
              <p className="text-xs font-medium text-slate-300 mb-2">Neurons Per Layer</p>
              <div className="flex flex-wrap gap-2">
                {NEURON_OPTIONS.map((n) => {
                  const active = form.search_space.neuron_choices.includes(n);
                  return (
                    <button
                      key={n}
                      type="button"
                      onClick={() => toggleArrayChoice('neuron_choices', n)}
                      className={`px-3 py-1.5 rounded-lg font-mono text-xs border transition-all ${
                        active
                          ? 'bg-violet-500/20 border-violet-500 text-violet-200'
                          : 'bg-[#0B1020] border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      {n}
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-slate-300 mb-2">Activation Functions</p>
              <div className="flex flex-wrap gap-2">
                {ACTIVATION_OPTIONS.map((act) => {
                  const active = form.search_space.activations.includes(act);
                  return (
                    <button
                      key={act}
                      type="button"
                      onClick={() => toggleArrayChoice('activations', act)}
                      className={`px-3 py-1.5 rounded-lg font-mono text-xs uppercase border transition-all ${
                        active
                          ? 'bg-cyan-500/20 border-cyan-500 text-cyan-200'
                          : 'bg-[#0B1020] border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      {act}
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-slate-300 mb-2">Dropout Rates</p>
              <div className="flex flex-wrap gap-2">
                {DROPOUT_OPTIONS.map((d) => {
                  const active = form.search_space.dropout_choices.includes(d);
                  return (
                    <button
                      key={d}
                      type="button"
                      onClick={() => toggleArrayChoice('dropout_choices', d)}
                      className={`px-3 py-1.5 rounded-lg font-mono text-xs border transition-all ${
                        active
                          ? 'bg-rose-500/20 border-rose-500 text-rose-200'
                          : 'bg-[#0B1020] border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      {d === 0 ? '0% (None)' : `${Math.round(d * 100)}%`}
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-slate-300 mb-2">Optimizers & Learning Rates</p>
              <div className="flex flex-wrap gap-2 mb-2.5">
                {OPTIMIZER_OPTIONS.map((opt) => {
                  const active = form.search_space.optimizers.includes(opt);
                  return (
                    <button
                      key={opt}
                      type="button"
                      onClick={() => toggleArrayChoice('optimizers', opt)}
                      className={`px-3 py-1.5 rounded-lg font-mono text-xs uppercase border transition-all ${
                        active
                          ? 'bg-emerald-500/20 border-emerald-500 text-emerald-200'
                          : 'bg-[#0B1020] border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      {opt}
                    </button>
                  );
                })}
              </div>
              <div className="flex flex-wrap gap-2">
                {LR_OPTIONS.map((lr) => {
                  const active = form.search_space.learning_rates.includes(lr);
                  return (
                    <button
                      key={lr}
                      type="button"
                      onClick={() => toggleArrayChoice('learning_rates', lr)}
                      className={`px-2.5 py-1 rounded-lg font-mono text-[11px] border transition-all ${
                        active
                          ? 'bg-amber-500/20 border-amber-500 text-amber-200'
                          : 'bg-[#0B1020] border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      lr={lr}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Submit Action Bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 glass-card p-5 border-violet-500/30">
          <div className="text-xs text-slate-400 space-y-0.5">
            <p className="text-slate-200 font-medium">
              Ready to launch real PyTorch Neural Architecture Search on CPU
            </p>
            <p className="font-mono">
              {form.generations} generations × {form.population_size} candidates × {form.max_epochs} epochs · {form.cpu_threads} threads
            </p>
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="btn-primary px-8 py-3 text-base shadow-violet-glow w-full sm:w-auto"
          >
            <Play className="w-4 h-4" />
            {submitting ? 'Launching Engine...' : 'Start Experiment'}
          </button>
        </div>
      </form>
    </div>
  );
}
