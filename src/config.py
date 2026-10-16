"""Central configuration for the RagBenchmarkPlatform pipeline.

Step 1 of the implementation plan. This module owns:

- ``CorpusConfig`` / ``IngestionConfig`` / ``EvalConfig`` dataclasses
  (YAML-serializable).
- Deterministic sha256 hashing:
    * ``sha256_file()`` -- sha256 of the source PDF bytes, recorded in the
      corpus manifest (``corpora/<name>/manifest.yaml``).
    * ``IngestionConfig.config_hash()`` -- sha256 over the canonical JSON of
      the fields that determine chunk identity (parser, chunker settings,
      embedding model, corpus name + corpus sha256). The golden set's
      ``ingestion.config_hash`` MUST equal this value; ``validate_golden.py``
      (step 6) refuses to build indexes on mismatch, so a changed
      parser/chunker/PDF invalidates ground truth loudly, never silently.
- ``RunManifest`` -- stamps config_hash + corpus_hash + code_hash (git HEAD)
  for every run under ``runs/``.

Stdlib + pyyaml only. No API keys here; those stay in ``.env``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# hashing primitives
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """sha256 of a file's bytes, streamed so large PDFs stay memory-light."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    """sha256 of a unicode string (utf-8)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_json(obj: object) -> str:
    """Deterministic JSON: sorted keys, compact separators.

    This is what makes the config hash stable across machines, Python
    versions, and dict insertion orders.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# corpus + ingestion + eval configs
# ---------------------------------------------------------------------------

@dataclass
class CorpusConfig:
    """Identifies the exact source-document bytes a corpus was built from."""
    name: str = "postgresql-18-manual"
    pdf_filename: str = "postgresql-18-A4.pdf"
    sha256: str = ""          # fill via: python -m src.config --hash-pdf <pdf>
    pages: int = 3156
    source_url: str = ""      # where the PDF was downloaded from
    retrieved_at: str = ""    # ISO date of download; metadata only, not hashed

    def manifest_dict(self) -> dict:
        return dataclasses.asdict(self)

    def write_manifest(self, path: str | Path) -> Path:
        """Write corpora/<name>/manifest.yaml. Commit the manifest, not the PDF."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.manifest_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_pdf(cls, pdf_path: str | Path, **kwargs) -> "CorpusConfig":
        """Build a CorpusConfig by hashing a local PDF."""
        pdf_path = Path(pdf_path)
        return cls(pdf_filename=pdf_path.name, sha256=sha256_file(pdf_path), **kwargs)

    @classmethod
    def from_manifest(cls, path: str | Path) -> "CorpusConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**d)


@dataclass
class IngestionConfig:
    """Everything that determines chunk identity.

    Change any hashed field and chunk ids shift -> the golden set's ground
    truth must be rebuilt (step 5), never reused.
    """
    parser: str = "docling"                     # docling | pymupdf | unstructured
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-small"
    corpus: CorpusConfig = field(default_factory=CorpusConfig)

    def hashed_fields(self) -> dict:
        """Fields covered by config_hash().

        Deliberately curated, not "hash everything": metadata such as
        retrieved_at or source_url is excluded, so re-downloading byte-identical
        content does not invalidate ground truth.
        """
        return {
            "parser": self.parser,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "corpus_name": self.corpus.name,
            "corpus_sha256": self.corpus.sha256,
        }

    def config_hash(self) -> str:
        """The pinning hash. Golden sets record this in ingestion.config_hash."""
        return sha256_str(canonical_json(self.hashed_fields()))

    def check_against_golden(self, golden_ingestion: dict) -> None:
        """Raise if a golden set was built for a different ingestion config.

        Called by validate_golden.py (step 6) before any index is built.
        """
        expected = golden_ingestion.get("config_hash", "")
        actual = self.config_hash()
        if expected != actual:
            raise ValueError(
                f"golden set config_hash mismatch: "
                f"golden={expected[:12]}... vs current={actual[:12]}... -- "
                "ground truth chunk ids are stale; rebuild them (step 5), "
                "do not evaluate against shifted chunks."
            )


@dataclass
class EvalConfig:
    """Retrieval-eval settings (Phase 2). Stamped on run manifests but never
    part of the config hash -- eval settings don't change chunk identity."""
    k_values: list = field(default_factory=lambda: [3, 5, 10])
    rerank_candidate_ns: list = field(default_factory=lambda: [10, 20, 50])
    judge_model: str = "gpt-4o-mini"
    question_quotas: dict = field(
        default_factory=lambda: {"exact_term": 0.4, "paraphrase": 0.4, "multi_hop": 0.2}
    )


# ---------------------------------------------------------------------------
# run manifests: config hash + corpus hash + code hash, stamped per run
# ---------------------------------------------------------------------------

def code_hash(repo_root: str | Path | None = None) -> str:
    """Short git HEAD sha of the repo, or 'unknown' outside a git checkout."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


@dataclass
class RunManifest:
    """Provenance record for one pipeline/eval run. Written to runs/<run_id>/."""
    run_id: str = ""
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    config_hash: str = ""
    corpus_hash: str = ""
    code_hash_: str = ""   # trailing underscore: `code_hash` is the function name
    eval: dict = field(default_factory=dict)
    notes: str = ""

    def manifest_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["code_hash"] = d.pop("code_hash_")
        return d

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.manifest_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_configs(
        cls,
        ingestion: IngestionConfig,
        eval_cfg: EvalConfig,
        run_id: str,
        notes: str = "",
    ) -> "RunManifest":
        return cls(
            run_id=run_id,
            config_hash=ingestion.config_hash(),
            corpus_hash=ingestion.corpus.sha256,
            code_hash_=code_hash(),
            eval=dataclasses.asdict(eval_cfg),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# YAML io
# ---------------------------------------------------------------------------

def default_ingestion_config() -> IngestionConfig:
    return IngestionConfig()


def save_ingestion_config(cfg: IngestionConfig, path: str | Path) -> Path:
    """Write config YAML. The file also stamps _config_hash for inspection --
    informational only; the hash is always recomputed, never trusted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = dataclasses.asdict(cfg)
    d["_config_hash"] = cfg.config_hash()
    path.write_text(
        yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def load_ingestion_config(path: str | Path) -> IngestionConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    d.pop("_config_hash", None)  # stamped value is never trusted
    corpus = d.pop("corpus", {})
    return IngestionConfig(corpus=CorpusConfig(**corpus), **d)


# ---------------------------------------------------------------------------
# CLI: the concrete "hash your PDF" workflow
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="config hashing and corpus manifest helpers (step 1)"
    )
    ap.add_argument(
        "--hash-pdf", metavar="PDF",
        help="hash a PDF and write corpora/<name>/manifest.yaml",
    )
    ap.add_argument("--corpus-name", default="postgresql-18-manual")
    ap.add_argument(
        "--init-config", metavar="YAML",
        help="write a default ingestion config YAML (edit corpus sha256 after --hash-pdf)",
    )
    ap.add_argument(
        "--config-hash", metavar="YAML",
        help="print the config_hash of an ingestion config YAML",
    )
    args = ap.parse_args(argv)

    if args.hash_pdf:
        corpus = CorpusConfig.from_pdf(
            args.hash_pdf,
            name=args.corpus_name,
            retrieved_at=datetime.now(timezone.utc).date().isoformat(),
        )
        manifest = corpus.write_manifest(
            Path("corpora") / args.corpus_name / "manifest.yaml"
        )
        print(f"sha256: {corpus.sha256}")
        print(f"manifest: {manifest}")

    if args.init_config:
        cfg = default_ingestion_config()
        out = save_ingestion_config(cfg, args.init_config)
        print(f"config: {out}")
        print(f"config_hash: {cfg.config_hash()}  (fill corpus.sha256 first if empty)")

    if args.config_hash:
        cfg = load_ingestion_config(args.config_hash)
        print(cfg.config_hash())

    if not (args.hash_pdf or args.init_config or args.config_hash):
        ap.print_help()


if __name__ == "__main__":
    main()
