import json
import random
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import torch

from backend.api.websocket import ws_manager
from backend.config import EXPERIMENTS_DIR
from backend.database.db import db_manager, utc_now
from backend.datasets.loader import load_dataset
from backend.engine.search_space import SearchSpace, architecture_display_name
from backend.evaluation.evaluator import evaluate_candidate
from backend.evolution.genetic import create_next_generation, initialize_population
from backend.models.dynamic_net import DynamicNeuralNet
from backend.optimization.pareto import compute_fitness_and_pareto
from backend.reinforcement_learning.rl_selector import RLArchitectureController
from backend.training.trainer import ExperimentStopped, train_candidate


class ExperimentRunnerState:
    def __init__(self, exp_id: str, total_generations: int, population_size: int, max_epochs: int):
        self.exp_id = exp_id
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.live_snapshot: Dict[str, Any] = {
            "experiment_id": exp_id,
            "status": "queued",
            "current_generation": 0,
            "total_generations": total_generations,
            "population_size": population_size,
            "max_epochs": max_epochs,
            "current_candidate": None,
            "current_epoch": 0,
            "evaluated_in_generation": 0,
            "total_evaluated": 0,
            "total_planned": total_generations * population_size,
            "latest_epoch_metrics": None,
            "started_at": utc_now(),
        }


class NASEngineManager:
    """
    Asynchronous Neural Architecture Search execution manager.
    Runs real PyTorch training, evolutionary generations, RL policy updates,
    and Pareto multi-objective ranking in background threads without blocking FastAPI.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runners: Dict[str, ExperimentRunnerState] = {}

    def _emit(self, exp_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        ws_manager.broadcast_threadsafe(
            {
                "type": event_type,
                "experiment_id": exp_id,
                "timestamp": utc_now(),
                "data": payload,
            },
            experiment_id=exp_id,
        )

    def _log(self, exp_id: str, message: str, level: str = "info", generation: Optional[int] = None) -> None:
        entry = db_manager.add_log(exp_id, message, level=level, generation=generation)
        self._emit(exp_id, "log", entry)

    def get_live_state(self, exp_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            runner = self._runners.get(exp_id)
            if runner:
                return dict(runner.live_snapshot)
        return None

    def get_active_experiment_id(self) -> Optional[str]:
        with self._lock:
            for exp_id, runner in self._runners.items():
                if runner.thread and runner.thread.is_alive():
                    return exp_id
        return None

    def start_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            active_id = self.get_active_experiment_id()
            if active_id is not None:
                raise RuntimeError(
                    f"Experiment '{active_id}' is currently active. Stop or wait for it to finish before starting a new CPU search."
                )

            exp_id = f"exp_{uuid.uuid4().hex[:8]}"
            exp = db_manager.create_experiment(
                exp_id=exp_id,
                name=params["name"],
                dataset=params["dataset"],
                optimization_mode=params["optimization_mode"],
                search_strategy=params["search_strategy"],
                config=params,
                total_generations=params["generations"],
                population_size=params["population_size"],
            )

            state = ExperimentRunnerState(
                exp_id=exp_id,
                total_generations=params["generations"],
                population_size=params["population_size"],
                max_epochs=params["max_epochs"],
            )
            t = threading.Thread(
                target=self._run_worker,
                args=(exp_id, params, state),
                daemon=True,
                name=f"nas-{exp_id}",
            )
            state.thread = t
            self._runners[exp_id] = state
            t.start()
            return exp

    def pause_experiment(self, exp_id: str) -> Dict[str, Any]:
        with self._lock:
            runner = self._runners.get(exp_id)
            if not runner or not runner.thread or not runner.thread.is_alive():
                raise ValueError("Experiment is not currently running.")
            runner.pause_event.set()
            runner.live_snapshot["status"] = "paused"
            db_manager.update_experiment(exp_id, status="paused")
            self._log(exp_id, "Experiment paused by user. Training will suspend safely.", level="warning")
            self._emit(exp_id, "experiment_paused", {"status": "paused", "live": dict(runner.live_snapshot)})
            return db_manager.get_experiment(exp_id)

    def resume_experiment(self, exp_id: str) -> Dict[str, Any]:
        with self._lock:
            runner = self._runners.get(exp_id)
            if not runner or not runner.thread or not runner.thread.is_alive():
                raise ValueError("Experiment is not currently active.")
            runner.pause_event.clear()
            runner.live_snapshot["status"] = "running"
            db_manager.update_experiment(exp_id, status="running")
            self._log(exp_id, "Experiment resumed by user.", level="info")
            self._emit(exp_id, "experiment_resumed", {"status": "running", "live": dict(runner.live_snapshot)})
            return db_manager.get_experiment(exp_id)

    def stop_experiment(self, exp_id: str) -> Dict[str, Any]:
        with self._lock:
            runner = self._runners.get(exp_id)
            if not runner or not runner.thread or not runner.thread.is_alive():
                exp = db_manager.get_experiment(exp_id)
                if exp and exp["status"] in ("running", "paused", "queued"):
                    db_manager.update_experiment(exp_id, status="stopped")
                    return db_manager.get_experiment(exp_id)
                raise ValueError("Experiment is not currently running.")
            runner.stop_event.set()
            runner.pause_event.clear()
            runner.live_snapshot["status"] = "stopping"
            self._log(exp_id, "Stop signal received. Halting safely after current step...", level="warning")
            return db_manager.get_experiment(exp_id)

    def _persist_pareto_and_best(
        self,
        exp_id: str,
        all_evaluated: List[Dict[str, Any]],
        optimization_mode: str,
        rl_controller: Optional[RLArchitectureController] = None,
    ) -> Optional[Dict[str, Any]]:
        if not all_evaluated:
            return None
        compute_fitness_and_pareto(all_evaluated, optimization_mode=optimization_mode)
        for arch in all_evaluated:
            db_manager.update_architecture(
                arch["id"],
                fitness_score=arch["fitness_score"],
                pareto_rank=arch["pareto_rank"],
                crowding_distance=arch["crowding_distance"],
                is_pareto_optimal=arch["is_pareto_optimal"],
            )

        best_arch = max(
            all_evaluated,
            key=lambda a: (float(a.get("fitness_score") or 0.0), float(a.get("accuracy") or 0.0)),
        )
        best_acc = max(float(a.get("accuracy") or 0.0) for a in all_evaluated)
        best_lat = min(float(a.get("inference_latency_ms") or 1e6) for a in all_evaluated)
        best_size = min(float(a.get("model_size_kb") or 1e6) for a in all_evaluated)

        db_manager.update_experiment(
            exp_id,
            evaluated_count=len(all_evaluated),
            best_architecture_id=best_arch["id"],
            best_accuracy=round(best_acc, 5),
            best_latency_ms=round(best_lat, 4),
            best_model_size_kb=round(best_size, 2),
            best_fitness=best_arch["fitness_score"],
            rl_stats=rl_controller.get_policy_summary() if rl_controller else None,
        )
        return best_arch

    def _export_experiment_json(self, exp_id: str) -> None:
        exp = db_manager.get_experiment(exp_id)
        archs = db_manager.list_architectures(experiment_id=exp_id)
        if not exp:
            return
        payload = {"experiment": exp, "architectures": archs}
        out_path = EXPERIMENTS_DIR / f"{exp_id}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _run_worker(self, exp_id: str, params: Dict[str, Any], state: ExperimentRunnerState) -> None:
        seed = params.get("seed")
        if seed is None:
            seed = int(time.time() * 1000) % 100000
        rng = random.Random(seed)
        torch.manual_seed(seed)

        generations = int(params["generations"])
        population_size = int(params["population_size"])
        max_epochs = int(params["max_epochs"])
        batch_size = int(params["batch_size"])
        cpu_threads = int(params["cpu_threads"])
        optimization_mode = params["optimization_mode"]
        search_strategy = params["search_strategy"]
        elitism_count = int(params.get("elitism_count", 1))
        search_space = SearchSpace.from_dict(params.get("search_space", {}))
        rl_controller = RLArchitectureController(search_space=search_space)

        all_evaluated: List[Dict[str, Any]] = []

        try:
            state.live_snapshot["status"] = "running"
            db_manager.update_experiment(exp_id, status="running", current_generation=0)
            self._log(
                exp_id,
                f"Starting NEUROEVOLVE search '{params['name']}' | Dataset={params['dataset']} | "
                f"Mode={optimization_mode} | Strategy={search_strategy} | Gens={generations}, Pop={population_size}, Epochs={max_epochs}, CPU threads={cpu_threads}",
                level="info",
                generation=0,
            )
            self._emit(exp_id, "experiment_started", {"experiment": db_manager.get_experiment(exp_id), "live": dict(state.live_snapshot)})

            # Load real dataset
            self._log(exp_id, f"Loading dataset '{params['dataset']}' (train<={params['max_train_samples']}, val<={params['max_val_samples']})...", generation=0)
            train_loader, val_loader, input_dim, num_classes, ds_meta = load_dataset(
                dataset_name=params["dataset"],
                batch_size=batch_size,
                max_train_samples=int(params.get("max_train_samples", 4000)),
                max_val_samples=int(params.get("max_val_samples", 1000)),
            )
            self._log(
                exp_id,
                f"Dataset ready: {ds_meta['display_name']} — {ds_meta['train_samples']} train / {ds_meta['val_samples']} val samples, "
                f"input_dim={input_dim}, classes={num_classes}.",
                level="success",
                generation=0,
            )

            prev_gen_evaluated: List[Dict[str, Any]] = []

            for gen in range(1, generations + 1):
                if state.stop_event.is_set():
                    raise ExperimentStopped("Experiment stopped before generation start.")

                state.live_snapshot["current_generation"] = gen
                state.live_snapshot["evaluated_in_generation"] = 0
                db_manager.update_experiment(exp_id, current_generation=gen)

                if gen == 1:
                    proposals = initialize_population(search_space, population_size, batch_size, rng)
                    self._log(
                        exp_id,
                        f"Generation {gen}/{generations}: Initialized {len(proposals)} diverse candidate architectures.",
                        level="info",
                        generation=gen,
                    )
                else:
                    proposals = create_next_generation(
                        previous_generation=prev_gen_evaluated,
                        search_space=search_space,
                        population_size=population_size,
                        batch_size=batch_size,
                        strategy=search_strategy,
                        rng=rng,
                        rl_controller=rl_controller,
                        elitism_count=elitism_count,
                    )
                    self._log(
                        exp_id,
                        f"Generation {gen}/{generations}: Evolved {len(proposals)} architectures via {search_strategy} "
                        f"(elitism, crossover, mutation, RL UCB1).",
                        level="info",
                        generation=gen,
                    )

                self._emit(
                    exp_id,
                    "generation_started",
                    {"generation": gen, "total_generations": generations, "population_size": len(proposals), "live": dict(state.live_snapshot)},
                )

                current_gen_evaluated: List[Dict[str, Any]] = []

                for cand_idx, prop in enumerate(proposals):
                    if state.stop_event.is_set():
                        raise ExperimentStopped("Experiment stopped between candidates.")

                    arch_id = f"arch_{uuid.uuid4().hex[:8]}"
                    cfg = prop["config"]
                    arch_name = architecture_display_name(cfg, gen, cand_idx)

                    model = DynamicNeuralNet(input_dim=input_dim, num_classes=num_classes, arch_config=cfg)
                    layer_summary = model.describe_layers()

                    arch_record: Dict[str, Any] = {
                        "id": arch_id,
                        "experiment_id": exp_id,
                        "generation": gen,
                        "candidate_index": cand_idx,
                        "name": arch_name,
                        "origin": prop["origin"],
                        "parent_ids": prop.get("parent_ids", []),
                        "operation_details": prop.get("operation_details", ""),
                        "config": cfg,
                        "layer_summary": layer_summary,
                        "status": "training",
                        "created_at": utc_now(),
                    }
                    db_manager.save_architecture(arch_record)

                    state.live_snapshot["current_candidate"] = {
                        "id": arch_id,
                        "name": arch_name,
                        "generation": gen,
                        "candidate_index": cand_idx + 1,
                        "population_size": len(proposals),
                        "origin": prop["origin"],
                        "operation_details": prop.get("operation_details", ""),
                        "config": cfg,
                        "layer_summary": layer_summary,
                    }
                    state.live_snapshot["current_epoch"] = 0
                    state.live_snapshot["latest_epoch_metrics"] = None

                    self._log(
                        exp_id,
                        f"[Gen {gen} | {cand_idx + 1}/{len(proposals)}] Training {arch_name} ({prop['origin']}: {prop.get('operation_details', '')})",
                        level="info",
                        generation=gen,
                    )
                    self._emit(exp_id, "candidate_started", {"candidate": arch_record, "live": dict(state.live_snapshot)})

                    def _on_epoch(ep_rec: Dict[str, Any]) -> None:
                        state.live_snapshot["current_epoch"] = ep_rec["epoch"]
                        state.live_snapshot["latest_epoch_metrics"] = ep_rec
                        self._emit(
                            exp_id,
                            "epoch_completed",
                            {
                                "architecture_id": arch_id,
                                "architecture_name": arch_name,
                                "generation": gen,
                                "epoch_metrics": ep_rec,
                                "live": dict(state.live_snapshot),
                            },
                        )

                    trained_model, history, train_time_s, peak_mem_mb = train_candidate(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        arch_config=cfg,
                        max_epochs=max_epochs,
                        cpu_threads=cpu_threads,
                        pause_event=state.pause_event,
                        stop_event=state.stop_event,
                        on_epoch_end=_on_epoch,
                    )

                    eval_metrics = evaluate_candidate(
                        model=trained_model,
                        val_loader=val_loader,
                        input_dim=input_dim,
                        num_classes=num_classes,
                        exp_id=exp_id,
                        arch_id=arch_id,
                        arch_config=cfg,
                    )

                    last_train_loss = history[-1]["train_loss"] if history else None
                    arch_record.update(
                        {
                            "status": "evaluated",
                            "train_loss": last_train_loss,
                            "training_time_s": train_time_s,
                            "peak_memory_mb": peak_mem_mb,
                            "training_history": history,
                            **eval_metrics,
                        }
                    )

                    all_evaluated.append(arch_record)
                    current_gen_evaluated.append(arch_record)

                    # Recompute Pareto frontier & multi-objective fitness across all evaluated architectures so far
                    best_so_far = self._persist_pareto_and_best(
                        exp_id=exp_id,
                        all_evaluated=all_evaluated,
                        optimization_mode=optimization_mode,
                        rl_controller=rl_controller,
                    )

                    # Update RL Q-learning / UCB1 bandit with real evaluated fitness
                    rl_reward = rl_controller.update(cfg, arch_record["fitness_score"], generation=gen)
                    arch_record["rl_reward"] = rl_reward

                    db_manager.save_architecture(arch_record)
                    db_manager.update_experiment(exp_id, rl_stats=rl_controller.get_policy_summary())

                    state.live_snapshot["evaluated_in_generation"] = cand_idx + 1
                    state.live_snapshot["total_evaluated"] = len(all_evaluated)

                    self._log(
                        exp_id,
                        f"Evaluated {arch_name}: Acc={arch_record['accuracy'] * 100:.2f}%, F1={arch_record['f1_score'] * 100:.2f}%, "
                        f"Lat={arch_record['inference_latency_ms']:.3f}ms, Size={arch_record['model_size_kb']:.1f}KB, "
                        f"Params={arch_record['parameter_count']:,}, Fitness={arch_record['fitness_score']:.2f}, "
                        f"ParetoRank={arch_record['pareto_rank']}",
                        level="success" if arch_record["is_pareto_optimal"] else "info",
                        generation=gen,
                    )
                    self._emit(
                        exp_id,
                        "candidate_evaluated",
                        {
                            "architecture": arch_record,
                            "best_architecture": best_so_far,
                            "rl_stats": rl_controller.get_policy_summary(),
                            "live": dict(state.live_snapshot),
                        },
                    )

                prev_gen_evaluated = current_gen_evaluated
                gen_best = max(current_gen_evaluated, key=lambda a: a.get("fitness_score") or 0.0)
                self._log(
                    exp_id,
                    f"Generation {gen} complete! Best in Gen {gen}: {gen_best['name']} "
                    f"(Acc={gen_best['accuracy'] * 100:.2f}%, Fitness={gen_best['fitness_score']:.2f})",
                    level="success",
                    generation=gen,
                )
                self._emit(
                    exp_id,
                    "generation_completed",
                    {
                        "generation": gen,
                        "best_in_generation": gen_best,
                        "live": dict(state.live_snapshot),
                    },
                )

            # All generations finished!
            state.live_snapshot["status"] = "completed"
            state.live_snapshot["current_candidate"] = None
            db_manager.update_experiment(exp_id, status="completed", completed_at=utc_now())
            self._export_experiment_json(exp_id)
            self._log(
                exp_id,
                f"Experiment completed! Evaluated {len(all_evaluated)} architectures across {generations} generations.",
                level="success",
                generation=generations,
            )
            self._emit(
                exp_id,
                "experiment_completed",
                {"experiment": db_manager.get_experiment(exp_id), "live": dict(state.live_snapshot)},
            )

        except ExperimentStopped as exc:
            state.live_snapshot["status"] = "stopped"
            state.live_snapshot["current_candidate"] = None
            if all_evaluated:
                self._persist_pareto_and_best(exp_id, all_evaluated, optimization_mode, rl_controller)
            db_manager.update_experiment(exp_id, status="stopped", completed_at=utc_now())
            self._export_experiment_json(exp_id)
            self._log(exp_id, f"{exc} Saved {len(all_evaluated)} evaluated architectures.", level="warning")
            self._emit(
                exp_id,
                "experiment_stopped",
                {"experiment": db_manager.get_experiment(exp_id), "live": dict(state.live_snapshot)},
            )

        except Exception as exc:
            tb = traceback.format_exc()
            state.live_snapshot["status"] = "failed"
            state.live_snapshot["current_candidate"] = None
            db_manager.update_experiment(exp_id, status="failed", error_message=str(exc), completed_at=utc_now())
            self._log(exp_id, f"Experiment failed: {exc}", level="error")
            self._emit(
                exp_id,
                "experiment_failed",
                {"error": str(exc), "traceback": tb, "experiment": db_manager.get_experiment(exp_id)},
            )


nas_engine = NASEngineManager()
