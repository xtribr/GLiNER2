#!/usr/bin/env python3
"""Retreina modelos preditivos e permite promocao controlada por gates."""

import argparse
import json
import shutil
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from ml.prediction_model import AUDIT_REPORT_NAME, ENEMPredictionModel, TARGETS
from ml.preprocessor import ENEMPreprocessor


def gates_pass(metrics: dict) -> bool:
    """Confere os gates de promocao ja calculados pelo modelo."""
    for target in TARGETS:
        info = metrics.get(target, {})
        gates = info.get("metadata", {}).get("promotion_gates", {})
        required = [
            gates.get("pass_top10_signed_error_gate"),
            gates.get("pass_top20_negative_share_gate"),
            gates.get("pass_rmse_gate"),
        ]
        if not all(required):
            print(f"Gate reprovado para {target}: {gates}")
            return False
    return True


def promote_artifacts(source_dir: Path, target_dir: Path) -> None:
    """Copia artefatos aprovados para o diretorio oficial de modelos."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        name = f"prediction_{target.replace('nota_', '')}.joblib"
        shutil.copy2(source_dir / name, target_dir / name)
    shutil.copy2(source_dir / AUDIT_REPORT_NAME, target_dir / AUDIT_REPORT_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retreina modelos de predicao ENEM.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=BACKEND_ROOT / "models",
        help="Diretorio onde os modelos serao treinados.",
    )
    parser.add_argument(
        "--promote-to",
        type=Path,
        help="Diretorio oficial para promover artefatos aprovados.",
    )
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="Falha se qualquer target reprovar os gates de promocao.",
    )
    parser.add_argument(
        "--allow-source-year-mismatch",
        action="store_true",
        help="Permite treino com fontes ausentes/de anos distintos e registra o override.",
    )
    parser.add_argument(
        "--validate-sources-only",
        action="store_true",
        help="Valida e imprime a proveniência sem treinar nem gravar artefatos.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preprocessor = ENEMPreprocessor()
    preprocessor.validate_training_sources(
        allow_year_mismatch=args.allow_source_year_mismatch
    )
    print(json.dumps(
        preprocessor.get_data_provenance(include_hashes=True),
        ensure_ascii=False,
        indent=2,
    ))
    if args.validate_sources_only:
        print("\nFontes validadas; nenhum modelo foi treinado ou alterado.")
        return 0

    model = ENEMPredictionModel(model_dir=args.model_dir)
    model.preprocessor = preprocessor
    metrics = model.train_all(
        allow_source_year_mismatch=args.allow_source_year_mismatch
    )

    print("\nRetraining complete.")
    for target, info in metrics.items():
        public_metrics = info.get("public_metrics", {})
        print(
            f"- {target}: "
            f"rmse_public={public_metrics.get('rmse')}, "
            f"r2_public={public_metrics.get('r2')}, "
            f"version={info.get('metadata', {}).get('model_version')}"
        )

    passed = gates_pass(metrics)
    if args.require_gates and not passed:
        print("\nModelos mantidos em staging; promocao bloqueada pelos gates.")
        return 2

    if args.promote_to:
        if args.require_gates and not passed:
            print("\nPromocao nao executada porque os gates falharam.")
            return 2
        promote_artifacts(args.model_dir, args.promote_to)
        print(f"\nArtefatos promovidos para: {args.promote_to}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
