const RAW_API_URL = (import.meta.env.VITE_API_URL || '').trim().replace(/\/+$/, '');
const API_BASE = RAW_API_URL ? `${RAW_API_URL}/api` : '/api';

async function request(path, options = {}) {
  const hasBody = options.body !== undefined && options.body !== null;
  const isFormData = hasBody && options.body instanceof FormData;
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      ...(hasBody && !isFormData ? { 'Content-Type': 'application/json' } : {}),
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const errData = await res.json();
      detail = errData.detail || errData.message || detail;
    } catch {
      // ignore parse failure
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  getHealth: () => request('/health'),
  getDefaults: () => request('/search-space/defaults'),
  getDashboard: (experimentId) =>
    request(experimentId ? `/dashboard?experiment_id=${encodeURIComponent(experimentId)}` : '/dashboard'),
  getDatasets: () => request('/datasets'),
  uploadDataset: (file, name, targetColumn) => {
    const fd = new FormData();
    fd.append('file', file);
    if (name) fd.append('name', name);
    if (targetColumn) fd.append('target_column', targetColumn);
    return request('/datasets/upload', { method: 'POST', body: fd });
  },
  listExperiments: () => request('/experiments'),
  createExperiment: (payload) =>
    request('/experiments', { method: 'POST', body: JSON.stringify(payload) }),
  getExperiment: (id) => request(`/experiments/${encodeURIComponent(id)}`),
  pauseExperiment: (id) => request(`/experiments/${encodeURIComponent(id)}/pause`, { method: 'POST' }),
  resumeExperiment: (id) => request(`/experiments/${encodeURIComponent(id)}/resume`, { method: 'POST' }),
  stopExperiment: (id) => request(`/experiments/${encodeURIComponent(id)}/stop`, { method: 'POST' }),
  deleteExperiment: (id) => request(`/experiments/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  getEvolution: (id) => request(`/experiments/${encodeURIComponent(id)}/evolution`),
  listArchitectures: (experimentId, onlyEvaluated = false) => {
    const params = new URLSearchParams();
    if (experimentId) params.set('experiment_id', experimentId);
    if (onlyEvaluated) params.set('only_evaluated', 'true');
    const q = params.toString();
    return request(`/architectures${q ? `?${q}` : ''}`);
  },
  getBestArchitecture: (experimentId) =>
    request(
      experimentId
        ? `/architectures/best?experiment_id=${encodeURIComponent(experimentId)}`
        : '/architectures/best'
    ),
  getArchitecture: (id) => request(`/architectures/${encodeURIComponent(id)}`),
  compareArchitectures: (architectureIds) =>
    request('/architectures/compare', {
      method: 'POST',
      body: JSON.stringify({ architecture_ids: architectureIds }),
    }),
  getModelDownloadUrl: (archId) => `${API_BASE}/architectures/${encodeURIComponent(archId)}/download/model`,
  getConfigDownloadUrl: (archId) => `${API_BASE}/architectures/${encodeURIComponent(archId)}/download/config`,
};

export function createExperimentSocket(experimentId, onMessage, onStatusChange) {
  const path = experimentId ? `/ws/experiments/${encodeURIComponent(experimentId)}` : '/ws';
  let url;
  if (RAW_API_URL) {
    const wsBase = RAW_API_URL.replace(/^https:/i, 'wss:').replace(/^http:/i, 'ws:');
    url = `${wsBase}${path}`;
  } else {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    url = `${protocol}//${window.location.host}${path}`;
  }

  let ws = null;
  let closedByUser = false;
  let reconnectTimer = null;
  let pingTimer = null;

  function connect() {
    if (closedByUser) return;
    ws = new WebSocket(url);

    ws.onopen = () => {
      onStatusChange?.('connected');
      pingTimer = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 20000);
    };

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        if (parsed.type !== 'pong') {
          onMessage?.(parsed);
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onclose = () => {
      clearInterval(pingTimer);
      onStatusChange?.('disconnected');
      if (!closedByUser) {
        reconnectTimer = setTimeout(connect, 2500);
      }
    };

    ws.onerror = () => {
      onStatusChange?.('error');
    };
  }

  connect();

  return () => {
    closedByUser = true;
    clearInterval(pingTimer);
    clearTimeout(reconnectTimer);
    if (ws) ws.close();
  };
}

export function fmtPct(val, digits = 2) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
  return `${(Number(val) * 100).toFixed(digits)}%`;
}

export function fmtMs(val) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
  return `${Number(val).toFixed(3)} ms`;
}

export function fmtSizeKb(val) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
  const n = Number(val);
  if (n >= 1024) return `${(n / 1024).toFixed(2)} MB`;
  return `${n.toFixed(1)} KB`;
}

export function fmtParams(val) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
  const n = Number(val);
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

export function fmtSeconds(val) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
  return `${Number(val).toFixed(2)}s`;
}
