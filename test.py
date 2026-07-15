import argparse
from pathlib import Path

from train import (
    build_model,
    build_output_paths,
    create_collator,
    create_dataset,
    create_loader,
    evaluate_detailed,
    load_config,
    load_model_state,
    load_torch_checkpoint,
    print_config_summary,
    resolve_data_paths,
    resolve_device,
    set_reproducibility,
    setup_logger,
    write_json,
    _expand_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TRUST with the same YAML configuration used for training."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and print the configuration without loading data or a model",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    cfg, config_path = load_config(cli.config)
    print_config_summary(cfg, config_path)
    if cli.check_config:
        return

    config_dir = config_path.parent
    output_paths = build_output_paths(cfg, config_dir)
    log = setup_logger(output_paths["dir"], cfg["run"]["name"], "test")
    device = resolve_device(cfg)
    set_reproducibility(cfg["run"]["seed"])
    data_paths = resolve_data_paths(cfg, config_dir)
    test_dataset = create_dataset("test", cfg, data_paths)
    if len(test_dataset) == 0:
        raise RuntimeError("test dataset is empty")
    collator = create_collator(cfg)
    raw_probe = test_dataset[0]
    fixed_probe = collator.fit_sample(raw_probe)
    log.info("Object feature guard: %s -> %s", tuple(raw_probe[1].shape), tuple(fixed_probe[1].shape))
    test_loader = create_loader(test_dataset, cfg, False, collator)

    model = build_model(cfg, device)
    configured_checkpoint = cfg["test"]["checkpoint_path"]
    checkpoint_path = (
        _expand_path(configured_checkpoint, config_dir)
        if configured_checkpoint
        else output_paths["best"]
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Test checkpoint not found: {checkpoint_path}")
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if cfg["test"]["use_ema_weights"] and checkpoint.get("ema_model_state_dict") is not None:
            model_state = checkpoint["ema_model_state_dict"]
        else:
            model_state = checkpoint["model_state_dict"]
        best_val_acc = float(checkpoint.get("best_acc", 0.0))
    else:
        model_state = checkpoint
        best_val_acc = 0.0
    load_model_state(model, model_state, strict=True)
    model.eval()

    configured_predictions = cfg["test"]["predictions_csv"]
    predictions_path = (
        _expand_path(configured_predictions, config_dir)
        if configured_predictions
        else output_paths["predictions"]
    )
    configured_metrics = cfg["test"]["metrics_json"]
    metrics_path = (
        _expand_path(configured_metrics, config_dir)
        if configured_metrics
        else output_paths["metrics"]
    )
    metrics, _ = evaluate_detailed(
        cfg, model, test_loader, device, best_val_acc, Path(predictions_path)
    )
    write_json(Path(metrics_path), metrics)
    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Predictions: %s", predictions_path)
    log.info("Metrics: %s", metrics_path)
    log.info("PAR=%.2f | CAR=%.2f | Acc_ALL=%.2f", metrics["PAR"], metrics["CAR"], metrics["Acc_ALL"])


if __name__ == "__main__":
    main()
