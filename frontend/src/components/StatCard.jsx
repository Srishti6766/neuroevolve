import React from 'react';
import { motion } from 'framer-motion';

const ACCENT_MAP = {
  violet: 'from-violet-500/20 to-violet-600/5 border-violet-500/30 text-violet-400',
  cyan: 'from-cyan-500/20 to-cyan-600/5 border-cyan-500/30 text-cyan-400',
  emerald: 'from-emerald-500/20 to-emerald-600/5 border-emerald-500/30 text-emerald-400',
  amber: 'from-amber-500/20 to-amber-600/5 border-amber-500/30 text-amber-400',
  rose: 'from-rose-500/20 to-rose-600/5 border-rose-500/30 text-rose-400',
};

export default function StatCard({ label, value, subtitle, icon: Icon, accent = 'violet' }) {
  const accentStyle = ACCENT_MAP[accent] || ACCENT_MAP.violet;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card-hover p-5 relative overflow-hidden"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">{label}</p>
          <p className="mt-2 text-2xl font-bold tracking-tight text-white font-mono truncate">
            {value}
          </p>
          {subtitle && <p className="mt-1.5 text-xs text-slate-400 truncate">{subtitle}</p>}
        </div>
        {Icon && (
          <div className={`p-2.5 rounded-xl bg-gradient-to-br border ${accentStyle} shrink-0`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>
    </motion.div>
  );
}
