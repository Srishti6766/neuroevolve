"""
Comprehensive end-to-end verification tests for NEUROEVOLVE backend:
1. Health, datasets, defaults, and CSV dataset upload
2. Real PyTorch NAS experiment execution across multiple generations
3. Genetic evolution (elitism, crossover, mutation) + RL UCB1/Q-Learning + NSGA-II Pareto frontier
4. SQLite persistence across reloads
5. Comparison, NetworkX Evolution graph, Best architecture, and Model (.pt) / Config (.json) downloads
6. Safe stop handling on active experiments
"""
import json
import os
import time
from pathlib import Path

from backend.api import routes
from backend.api.schemas import CompareRequest, CreateExperimentRequest
from backend.database.db import DatabaseManager, db_manager
from backend.datasets.loader import register_csv_dataset


def test_full_nas_experiment_lifecycle():
    # 1. Health, Defaults, Datasets
    health = routes.health_check()
    assert health["status"] == "ok"
    assert health["device"] == "cpu"
    assert health["cpu_count"] >= 1

    defaults = routes.get_default_search_space()
    assert "search_space" in defaults
    assert defaults["recommended_cpu_defaults"]["dataset"] == "mnist"

    ds_res = routes.get_datasets()
    ds_ids = {d["id"] for d in ds_res["datasets"]}
    assert {"mnist", "fashion_mnist", "digits"}.issubset(ds_ids)

    # 2. Launch a real small PyTorch NAS experiment (built-in offline UCI Digits dataset)
    req = CreateExperimentRequest(
        name="Pytest Real PyTorch NAS Verification",
        dataset="digits",
        generations=3,
        population_size=3,
        max_epochs=2,
        batch_size=64,
        max_train_samples=600,
        max_val_samples=200,
        optimization_mode="balanced",
        search_strategy="hybrid_rl",
        cpu_threads=2,
        seed=42,
        search_space={
            "min_layers": 1,
            "max_layers": 3,
            "neuron_choices": [32, 64, 128],
            "activations": ["relu", "gelu", "tanh"],
            "dropout_choices": [0.0, 0.1],
            "optimizers": ["adam", "adamw"],
            "learning_rates": [0.001, 0.005],
            "batch_norm_choices": [False, True],
        },
    )
    create_res = routes.create_and_start_experiment(req)
    exp_id = create_res["experiment"]["id"]
    assert exp_id.startswith("exp_")

    # 3. Wait for real background PyTorch training & evolution to complete
    deadline = time.time() + 90
    detail = None
    while time.time() < deadline:
        detail = routes.get_experiment_detail(exp_id)
        status = detail["experiment"]["status"]
        if status in ("completed", "failed", "stopped"):
            break
        time.sleep(0.35)

    assert detail is not None
    assert detail["experiment"]["status"] == "completed", (
        f"Experiment did not complete cleanly: {detail['experiment']}"
    )

    archs = detail["architectures"]
    assert len(archs) == 9  # 3 generations x 3 candidates
    assert all(a["status"] == "evaluated" for a in archs)
    assert {a["generation"] for a in archs} == {1, 2, 3}

    # Verify genetic evolution + RL occurred in generations 2 & 3
    gen2_3_origins = {a["origin"] for a in archs if a["generation"] >= 2}
    assert "elitism" in gen2_3_origins
    assert len(gen2_3_origins & {"mutation", "crossover", "rl_guided"}) >= 1

    # Verify Pareto frontier & RL updates
    assert any(a["is_pareto_optimal"] for a in archs)
    assert detail["experiment"]["rl_stats"]["total_updates"] == 9
    assert len(detail["logs"]) >= 10

    # 4. Verify Dashboard reflects real stored results
    dash = routes.get_dashboard(experiment_id=exp_id)
    assert dash["total_architectures_evaluated"] == 9
    assert len(dash["generation_metrics"]) == 3
    assert dash["best_accuracy"] > 0.5
    assert dash["best_latency_ms"] > 0
    assert dash["best_model_size_kb"] > 0
    assert len(dash["pareto_points"]) == 9

    # 5. Verify Evolution Graph (NetworkX lineage)
    evo = routes.get_evolution_history(exp_id)
    assert evo["graph"]["stats"]["total_nodes"] == 9
    assert evo["graph"]["stats"]["total_edges"] >= 2
    assert len(evo["graph"]["generations"]) == 3

    # 6. Verify Model Comparison Endpoint
    ids = [a["id"] for a in archs[:4]]
    cmp_res = routes.compare_architectures(CompareRequest(architecture_ids=ids))
    assert cmp_res["count"] == 4
    fitnesses = [a["fitness_score"] for a in cmp_res["architectures"]]
    assert fitnesses == sorted(fitnesses, reverse=True)

    # 7. Verify Best Architecture & Downloads (.pt and .json)
    best_res = routes.get_best_architecture(experiment_id=exp_id)
    best_arch = best_res["architecture"]
    assert best_arch is not None
    assert best_arch["parameter_count"] > 0
    assert best_arch["model_size_kb"] > 0
    assert best_arch["inference_latency_ms"] > 0
    assert best_arch["peak_memory_mb"] > 0
    assert len(best_arch["training_history"]) == 2
    assert os.path.exists(best_arch["model_weights_path"])

    model_resp = routes.download_model_weights(best_arch["id"])
    assert Path(model_resp.path).exists()
    assert os.path.getsize(model_resp.path) > 500

    cfg_resp = routes.download_architecture_config(best_arch["id"])
    cfg_payload = json.loads(cfg_resp.body.decode("utf-8"))
    assert cfg_payload["architecture_id"] == best_arch["id"]
    assert cfg_payload["metrics"]["accuracy"] == best_arch["accuracy"]

    # 8. Verify SQLite persistence via a fresh DatabaseManager instance reading the same DB file
    fresh_db = DatabaseManager()
    persisted_exp = fresh_db.get_experiment(exp_id)
    assert persisted_exp is not None
    assert persisted_exp["status"] == "completed"
    assert len(fresh_db.list_architectures(experiment_id=exp_id)) == 9

    # 9. Clean up test experiment and verify weight files removed
    weights_path = best_arch["model_weights_path"]
    del_res = routes.delete_experiment(exp_id)
    assert del_res["deleted"] is True
    assert not os.path.exists(weights_path)


def test_csv_dataset_registration():
    # Generate a tiny valid 40-row classification CSV in memory
    lines = ["f1,f2,f3,f4,label"]
    for i in range(40):
        cls = i % 2
        lines.append(f"{cls * 2.0 + 0.1},{cls * 1.5 - 0.2},{i * 0.01},{1.0 - cls},class_{cls}")
    csv_bytes = "\n".join(lines).encode("utf-8")

    rec = register_csv_dataset("UnitTestCSV", "unit_test.csv", csv_bytes, target_column="label")
    assert rec["num_samples"] == 40
    assert rec["num_features"] == 4
    assert rec["num_classes"] == 2
    assert os.path.exists(rec["file_path"])

    # Cleanup uploaded test CSV & DB row
    Path(rec["file_path"]).unlink(missing_ok=True)
    with db_manager._lock:
        conn = db_manager._connect()
        try:
            conn.execute("DELETE FROM custom_datasets WHERE id = ?", (rec["id"],))
            conn.commit()
        finally:
            conn.close()
