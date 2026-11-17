import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.api.schemas import CompareRequest, CreateExperimentRequest
from backend.database.db import db_manager
from backend.datasets.loader import list_available_datasets, register_csv_dataset
from backend.engine.nas_engine import nas_engine
from backend.engine.search_space import DEFAULT_SEARCH_SPACE
from backend.evolution.genetic import build_evolution_graph

router = APIRouter(prefix="/api", tags=["neuroevolve"])


@router.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine": "NEUROEVOLVE NAS Engine",
        "pytorch_version": torch.__version__,
        "device": "cpu",
        "cpu_count": os.cpu_count() or 4,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "platform": platform.platform(),
        "active_experiment_id": nas_engine.get_active_experiment_id(),
    }


@router.get("/search-space/defaults")
def get_default_search_space() -> Dict[str, Any]:
    return {
        "search_space": DEFAULT_SEARCH_SPACE,
        "recommended_cpu_defaults": {
            "dataset": "mnist",
            "generations": 3,
            "population_size": 4,
            "max_epochs": 3,
            "batch_size": 64,
            "max_train_samples": 4000,
            "max_val_samples": 1000,
            "optimization_mode": "balanced",
            "search_strategy": "hybrid_rl",
            "cpu_threads": min(4, os.cpu_count() or 4),
            "elitism_count": 1,
        },
        "max_cpu_threads": os.cpu_count() or 4,
    }


@router.get("/dashboard")
def get_dashboard(experiment_id: Optional[str] = None) -> Dict[str, Any]:
    summary = db_manager.get_dashboard_summary(experiment_id=experiment_id)
    active = summary.get("active_experiment")
    summary["live_state"] = nas_engine.get_live_state(active["id"]) if active else None
    return summary


@router.get("/datasets")
def get_datasets() -> Dict[str, Any]:
    return {"datasets": list_available_datasets()}


@router.post("/datasets/upload")
async def upload_csv_dataset(
    file: UploadFile = File(...),
    name: str = Form(default=""),
    target_column: str = Form(default=""),
) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid .csv file.")
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="CSV file exceeds 25 MB limit.")
    try:
        ds = register_csv_dataset(
            name=name or Path(file.filename).stem,
            filename=file.filename,
            content=content,
            target_column=target_column,
        )
        return {"dataset": ds}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/experiments")
def list_experiments() -> Dict[str, Any]:
    exps = db_manager.list_experiments()
    return {
        "experiments": exps,
        "active_experiment_id": nas_engine.get_active_experiment_id(),
    }


@router.post("/experiments")
def create_and_start_experiment(req: CreateExperimentRequest) -> Dict[str, Any]:
    try:
        params = req.model_dump()
        exp = nas_engine.start_experiment(params)
        return {"experiment": exp}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/experiments/{exp_id}")
def get_experiment_detail(exp_id: str) -> Dict[str, Any]:
    exp = db_manager.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' not found.")
    archs = db_manager.list_architectures(experiment_id=exp_id)
    logs = db_manager.get_logs(exp_id, limit=250)
    live = nas_engine.get_live_state(exp_id)
    return {
        "experiment": exp,
        "architectures": archs,
        "logs": logs,
        "live_state": live,
    }


@router.post("/experiments/{exp_id}/pause")
def pause_experiment(exp_id: str) -> Dict[str, Any]:
    try:
        exp = nas_engine.pause_experiment(exp_id)
        return {"experiment": exp}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/experiments/{exp_id}/resume")
def resume_experiment(exp_id: str) -> Dict[str, Any]:
    try:
        exp = nas_engine.resume_experiment(exp_id)
        return {"experiment": exp}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/experiments/{exp_id}/stop")
def stop_experiment(exp_id: str) -> Dict[str, Any]:
    try:
        exp = nas_engine.stop_experiment(exp_id)
        return {"experiment": exp}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/experiments/{exp_id}")
def delete_experiment(exp_id: str) -> Dict[str, Any]:
    if nas_engine.get_active_experiment_id() == exp_id:
        raise HTTPException(status_code=409, detail="Stop the experiment before deleting it.")
    # Remove saved model weights for this experiment
    archs = db_manager.list_architectures(experiment_id=exp_id)
    for a in archs:
        wp = a.get("model_weights_path")
        if wp and os.path.exists(wp):
            try:
                os.remove(wp)
            except OSError:
                pass
    deleted = db_manager.delete_experiment(exp_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return {"deleted": True, "id": exp_id}


@router.get("/experiments/{exp_id}/evolution")
def get_evolution_history(exp_id: str) -> Dict[str, Any]:
    exp = db_manager.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' not found.")
    archs = db_manager.list_architectures(experiment_id=exp_id)
    graph = build_evolution_graph(archs)
    return {
        "experiment": exp,
        "graph": graph,
        "rl_stats": exp.get("rl_stats"),
    }


@router.get("/architectures")
def list_architectures(
    experiment_id: Optional[str] = None, only_evaluated: bool = False
) -> Dict[str, Any]:
    archs = db_manager.list_architectures(experiment_id=experiment_id, only_evaluated=only_evaluated)
    return {"architectures": archs}


@router.get("/architectures/best")
def get_best_architecture(experiment_id: Optional[str] = None) -> Dict[str, Any]:
    archs = db_manager.list_architectures(experiment_id=experiment_id, only_evaluated=True)
    if not archs:
        return {"architecture": None, "experiment": db_manager.get_experiment(experiment_id) if experiment_id else None}
    best = max(archs, key=lambda a: (float(a.get("fitness_score") or 0.0), float(a.get("accuracy") or 0.0)))
    exp = db_manager.get_experiment(best["experiment_id"])
    return {"architecture": best, "experiment": exp}


@router.get("/architectures/{arch_id}")
def get_architecture(arch_id: str) -> Dict[str, Any]:
    arch = db_manager.get_architecture(arch_id)
    if not arch:
        raise HTTPException(status_code=404, detail=f"Architecture '{arch_id}' not found.")
    exp = db_manager.get_experiment(arch["experiment_id"])
    return {"architecture": arch, "experiment": exp}


@router.post("/architectures/compare")
def compare_architectures(req: CompareRequest) -> Dict[str, Any]:
    found: List[Dict[str, Any]] = []
    for aid in req.architecture_ids:
        a = db_manager.get_architecture(aid)
        if a:
            found.append(a)
    if not found:
        raise HTTPException(status_code=404, detail="No matching architectures found.")
    ranked = sorted(
        found,
        key=lambda a: (float(a.get("fitness_score") or 0.0), float(a.get("accuracy") or 0.0)),
        reverse=True,
    )
    return {
        "architectures": ranked,
        "count": len(ranked),
    }


@router.get("/architectures/{arch_id}/download/model")
def download_model_weights(arch_id: str):
    arch = db_manager.get_architecture(arch_id)
    if not arch:
        raise HTTPException(status_code=404, detail="Architecture not found.")
    weights_path = arch.get("model_weights_path")
    if not weights_path or not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Trained model weights file not found on disk.")
    filename = f"neuroevolve_{arch['id']}_gen{arch['generation']}.pt"
    return FileResponse(path=weights_path, media_type="application/octet-stream", filename=filename)


@router.get("/architectures/{arch_id}/download/config")
def download_architecture_config(arch_id: str):
    arch = db_manager.get_architecture(arch_id)
    if not arch:
        raise HTTPException(status_code=404, detail="Architecture not found.")
    exp = db_manager.get_experiment(arch["experiment_id"])
    export_payload = {
        "platform": "NEUROEVOLVE",
        "architecture_id": arch["id"],
        "name": arch["name"],
        "experiment_id": arch["experiment_id"],
        "dataset": exp["dataset"] if exp else None,
        "generation": arch["generation"],
        "origin": arch["origin"],
        "operation_details": arch["operation_details"],
        "parent_ids": arch["parent_ids"],
        "architecture_config": arch["config"],
        "layers": arch["layer_summary"],
        "metrics": {
            "accuracy": arch["accuracy"],
            "precision": arch["precision_score"],
            "recall": arch["recall_score"],
            "f1_score": arch["f1_score"],
            "val_loss": arch["val_loss"],
            "train_loss": arch["train_loss"],
            "training_time_s": arch["training_time_s"],
            "inference_latency_ms": arch["inference_latency_ms"],
            "parameter_count": arch["parameter_count"],
            "model_size_kb": arch["model_size_kb"],
            "peak_memory_mb": arch["peak_memory_mb"],
            "fitness_score": arch["fitness_score"],
            "pareto_rank": arch["pareto_rank"],
            "is_pareto_optimal": arch["is_pareto_optimal"],
        },
        "training_history": arch["training_history"],
    }
    filename = f"neuroevolve_{arch['id']}_config.json"
    return JSONResponse(
        content=export_payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
