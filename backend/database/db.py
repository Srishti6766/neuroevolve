import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.config import DB_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS experiments (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        dataset TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'queued',
                        optimization_mode TEXT NOT NULL DEFAULT 'balanced',
                        search_strategy TEXT NOT NULL DEFAULT 'hybrid_rl',
                        config_json TEXT NOT NULL,
                        current_generation INTEGER NOT NULL DEFAULT 0,
                        total_generations INTEGER NOT NULL DEFAULT 3,
                        population_size INTEGER NOT NULL DEFAULT 4,
                        evaluated_count INTEGER NOT NULL DEFAULT 0,
                        best_architecture_id TEXT,
                        best_accuracy REAL,
                        best_latency_ms REAL,
                        best_model_size_kb REAL,
                        best_fitness REAL,
                        rl_stats_json TEXT,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT
                    );

                    CREATE TABLE IF NOT EXISTS architectures (
                        id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        generation INTEGER NOT NULL,
                        candidate_index INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        origin TEXT NOT NULL DEFAULT 'random',
                        parent_ids_json TEXT NOT NULL DEFAULT '[]',
                        operation_details TEXT DEFAULT '',
                        config_json TEXT NOT NULL,
                        layer_summary_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL DEFAULT 'pending',
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        val_loss REAL,
                        train_loss REAL,
                        training_time_s REAL,
                        inference_latency_ms REAL,
                        parameter_count INTEGER,
                        model_size_kb REAL,
                        peak_memory_mb REAL,
                        fitness_score REAL,
                        pareto_rank INTEGER,
                        crowding_distance REAL,
                        is_pareto_optimal INTEGER NOT NULL DEFAULT 0,
                        rl_reward REAL,
                        training_history_json TEXT NOT NULL DEFAULT '[]',
                        model_weights_path TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS experiment_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL DEFAULT 'info',
                        generation INTEGER,
                        message TEXT NOT NULL,
                        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS custom_datasets (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        num_samples INTEGER NOT NULL,
                        num_features INTEGER NOT NULL,
                        num_classes INTEGER NOT NULL,
                        target_column TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_arch_exp ON architectures(experiment_id);
                    CREATE INDEX IF NOT EXISTS idx_arch_gen ON architectures(experiment_id, generation);
                    CREATE INDEX IF NOT EXISTS idx_logs_exp ON experiment_logs(experiment_id, id);
                    """
                )
                # Mark any stale running/paused experiments from a prior unclean shutdown as stopped
                conn.execute(
                    """
                    UPDATE experiments
                    SET status = 'stopped',
                        error_message = COALESCE(error_message, 'Server restarted while experiment was active'),
                        updated_at = ?
                    WHERE status IN ('running', 'paused', 'queued')
                    """,
                    (utc_now(),),
                )
                conn.commit()
            finally:
                conn.close()

    # ---------- Experiments ----------
    def create_experiment(
        self,
        exp_id: str,
        name: str,
        dataset: str,
        optimization_mode: str,
        search_strategy: str,
        config: Dict[str, Any],
        total_generations: int,
        population_size: int,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO experiments (
                        id, name, dataset, status, optimization_mode, search_strategy,
                        config_json, current_generation, total_generations, population_size,
                        evaluated_count, created_at, updated_at
                    ) VALUES (?, ?, ?, 'queued', ?, ?, ?, 0, ?, ?, 0, ?, ?)
                    """,
                    (
                        exp_id,
                        name,
                        dataset,
                        optimization_mode,
                        search_strategy,
                        json.dumps(config),
                        total_generations,
                        population_size,
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return self.get_experiment(exp_id)

    def update_experiment(self, exp_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_now()
        cols = []
        vals = []
        for k, v in fields.items():
            if k in ("config", "rl_stats"):
                col = f"{k}_json"
                val = json.dumps(v) if v is not None else None
            else:
                col = k
                val = v
            cols.append(f"{col} = ?")
            vals.append(val)
        vals.append(exp_id)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(f"UPDATE experiments SET {', '.join(cols)} WHERE id = ?", vals)
                conn.commit()
            finally:
                conn.close()

    def _serialize_experiment(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json") or "{}")
        rl_raw = d.pop("rl_stats_json", None)
        d["rl_stats"] = json.loads(rl_raw) if rl_raw else None
        return d

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
                return self._serialize_experiment(row) if row else None
            finally:
                conn.close()

    def list_experiments(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
                return [self._serialize_experiment(r) for r in rows]
            finally:
                conn.close()

    def delete_experiment(self, exp_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ---------- Architectures ----------
    def save_architecture(self, arch: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO architectures (
                        id, experiment_id, generation, candidate_index, name, origin,
                        parent_ids_json, operation_details, config_json, layer_summary_json,
                        status, accuracy, precision_score, recall_score, f1_score,
                        val_loss, train_loss, training_time_s, inference_latency_ms,
                        parameter_count, model_size_kb, peak_memory_mb, fitness_score,
                        pareto_rank, crowding_distance, is_pareto_optimal, rl_reward,
                        training_history_json, model_weights_path, created_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        arch["id"],
                        arch["experiment_id"],
                        arch["generation"],
                        arch["candidate_index"],
                        arch["name"],
                        arch.get("origin", "random"),
                        json.dumps(arch.get("parent_ids", [])),
                        arch.get("operation_details", ""),
                        json.dumps(arch.get("config", {})),
                        json.dumps(arch.get("layer_summary", [])),
                        arch.get("status", "pending"),
                        arch.get("accuracy"),
                        arch.get("precision_score"),
                        arch.get("recall_score"),
                        arch.get("f1_score"),
                        arch.get("val_loss"),
                        arch.get("train_loss"),
                        arch.get("training_time_s"),
                        arch.get("inference_latency_ms"),
                        arch.get("parameter_count"),
                        arch.get("model_size_kb"),
                        arch.get("peak_memory_mb"),
                        arch.get("fitness_score"),
                        arch.get("pareto_rank"),
                        arch.get("crowding_distance"),
                        1 if arch.get("is_pareto_optimal") else 0,
                        arch.get("rl_reward"),
                        json.dumps(arch.get("training_history", [])),
                        arch.get("model_weights_path"),
                        arch.get("created_at") or utc_now(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def update_architecture(self, arch_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = []
        vals = []
        json_map = {
            "parent_ids": "parent_ids_json",
            "config": "config_json",
            "layer_summary": "layer_summary_json",
            "training_history": "training_history_json",
        }
        for k, v in fields.items():
            if k in json_map:
                cols.append(f"{json_map[k]} = ?")
                vals.append(json.dumps(v))
            elif k == "is_pareto_optimal":
                cols.append("is_pareto_optimal = ?")
                vals.append(1 if v else 0)
            else:
                cols.append(f"{k} = ?")
                vals.append(v)
        vals.append(arch_id)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(f"UPDATE architectures SET {', '.join(cols)} WHERE id = ?", vals)
                conn.commit()
            finally:
                conn.close()

    def _serialize_arch(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["parent_ids"] = json.loads(d.pop("parent_ids_json") or "[]")
        d["config"] = json.loads(d.pop("config_json") or "{}")
        d["layer_summary"] = json.loads(d.pop("layer_summary_json") or "[]")
        d["training_history"] = json.loads(d.pop("training_history_json") or "[]")
        d["is_pareto_optimal"] = bool(d.get("is_pareto_optimal", 0))
        return d

    def get_architecture(self, arch_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM architectures WHERE id = ?", (arch_id,)).fetchone()
                return self._serialize_arch(row) if row else None
            finally:
                conn.close()

    def list_architectures(
        self, experiment_id: Optional[str] = None, only_evaluated: bool = False
    ) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                clauses = []
                params: List[Any] = []
                if experiment_id:
                    clauses.append("experiment_id = ?")
                    params.append(experiment_id)
                if only_evaluated:
                    clauses.append("status = 'evaluated'")
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                rows = conn.execute(
                    f"SELECT * FROM architectures {where} ORDER BY generation ASC, candidate_index ASC, created_at ASC",
                    params,
                ).fetchall()
                return [self._serialize_arch(r) for r in rows]
            finally:
                conn.close()

    # ---------- Logs ----------
    def add_log(
        self, experiment_id: str, message: str, level: str = "info", generation: Optional[int] = None
    ) -> Dict[str, Any]:
        ts = utc_now()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO experiment_logs (experiment_id, timestamp, level, generation, message) VALUES (?, ?, ?, ?, ?)",
                    (experiment_id, ts, level, generation, message),
                )
                conn.commit()
                return {
                    "id": cur.lastrowid,
                    "experiment_id": experiment_id,
                    "timestamp": ts,
                    "level": level,
                    "generation": generation,
                    "message": message,
                }
            finally:
                conn.close()

    def get_logs(self, experiment_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM experiment_logs WHERE experiment_id = ? ORDER BY id DESC LIMIT ?",
                    (experiment_id, limit),
                ).fetchall()
                return [dict(r) for r in reversed(rows)]
            finally:
                conn.close()

    # ---------- Custom Datasets ----------
    def save_custom_dataset(self, ds: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO custom_datasets
                    (id, name, filename, file_path, num_samples, num_features, num_classes, target_column, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ds["id"],
                        ds["name"],
                        ds["filename"],
                        ds["file_path"],
                        ds["num_samples"],
                        ds["num_features"],
                        ds["num_classes"],
                        ds["target_column"],
                        ds.get("created_at") or utc_now(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return ds

    def list_custom_datasets(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT * FROM custom_datasets ORDER BY created_at DESC").fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_custom_dataset(self, ds_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM custom_datasets WHERE id = ?", (ds_id,)).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    # ---------- Global Dashboard Aggregates ----------
    def get_dashboard_summary(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                total_experiments = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
                active_exp_row = conn.execute(
                    "SELECT * FROM experiments WHERE status IN ('running', 'paused') ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
                active_experiment = self._serialize_experiment(active_exp_row) if active_exp_row else None

                if experiment_id:
                    archs = self.list_architectures(experiment_id=experiment_id, only_evaluated=True)
                    exp = self.get_experiment(experiment_id)
                else:
                    archs = self.list_architectures(only_evaluated=True)
                    exp = active_experiment or (self.list_experiments()[0] if total_experiments > 0 else None)

                total_evaluated = len(archs)
                best_acc = max((a["accuracy"] for a in archs if a["accuracy"] is not None), default=None)
                best_lat = min(
                    (a["inference_latency_ms"] for a in archs if a["inference_latency_ms"] is not None),
                    default=None,
                )
                best_size = min(
                    (a["model_size_kb"] for a in archs if a["model_size_kb"] is not None),
                    default=None,
                )

                best_arch = None
                best_acc_arch = None
                if archs:
                    best_arch = max(
                        archs,
                        key=lambda a: (a["fitness_score"] or 0.0, a["accuracy"] or 0.0),
                    )
                    best_acc_arch = max(
                        archs,
                        key=lambda a: (a["accuracy"] or 0.0, a["fitness_score"] or 0.0),
                    )

                # Generation trends (for selected experiment or latest experiment)
                trend_exp_id = experiment_id or (exp["id"] if exp else None)
                trend_archs = (
                    self.list_architectures(experiment_id=trend_exp_id, only_evaluated=True)
                    if trend_exp_id
                    else []
                )
                gen_map: Dict[int, List[Dict[str, Any]]] = {}
                for a in trend_archs:
                    gen_map.setdefault(a["generation"], []).append(a)

                generation_metrics = []
                for gen in sorted(gen_map.keys()):
                    items = gen_map[gen]
                    accs = [i["accuracy"] for i in items if i["accuracy"] is not None]
                    lats = [i["inference_latency_ms"] for i in items if i["inference_latency_ms"] is not None]
                    sizes = [i["model_size_kb"] for i in items if i["model_size_kb"] is not None]
                    fits = [i["fitness_score"] for i in items if i["fitness_score"] is not None]
                    generation_metrics.append(
                        {
                            "generation": gen,
                            "count": len(items),
                            "best_accuracy": round(max(accs), 4) if accs else 0.0,
                            "avg_accuracy": round(sum(accs) / len(accs), 4) if accs else 0.0,
                            "best_latency_ms": round(min(lats), 4) if lats else 0.0,
                            "avg_latency_ms": round(sum(lats) / len(lats), 4) if lats else 0.0,
                            "avg_model_size_kb": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
                            "min_model_size_kb": round(min(sizes), 2) if sizes else 0.0,
                            "best_fitness": round(max(fits), 2) if fits else 0.0,
                            "avg_fitness": round(sum(fits) / len(fits), 2) if fits else 0.0,
                        }
                    )

                pareto_points = [
                    {
                        "id": a["id"],
                        "name": a["name"],
                        "generation": a["generation"],
                        "accuracy": round((a["accuracy"] or 0.0) * 100.0, 2),
                        "inference_latency_ms": round(a["inference_latency_ms"] or 0.0, 4),
                        "model_size_kb": round(a["model_size_kb"] or 0.0, 2),
                        "fitness_score": round(a["fitness_score"] or 0.0, 2),
                        "pareto_rank": a["pareto_rank"] or 99,
                        "is_pareto_optimal": a["is_pareto_optimal"],
                        "origin": a["origin"],
                    }
                    for a in trend_archs
                ]

                return {
                    "total_experiments": total_experiments,
                    "total_architectures_evaluated": total_evaluated,
                    "current_generation": exp["current_generation"] if exp else 0,
                    "total_generations": exp["total_generations"] if exp else 0,
                    "best_accuracy": best_acc,
                    "best_latency_ms": best_lat,
                    "best_model_size_kb": best_size,
                    "active_experiment": active_experiment,
                    "selected_experiment": exp,
                    "best_architecture": best_arch,
                    "best_accuracy_architecture": best_acc_arch,
                    "generation_metrics": generation_metrics,
                    "pareto_points": pareto_points,
                }
            finally:
                conn.close()


def init_db() -> None:
    db_manager.init_db()


db_manager = DatabaseManager()
