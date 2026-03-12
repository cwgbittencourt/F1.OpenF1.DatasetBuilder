from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable


VALID_SUBDIRS = {"bronze", "silver", "gold"}


def should_sync_data_lake(env: dict[str, str] | None = None) -> bool:
    source = env or os.environ
    raw = source.get("SYNC_DATA_LAKE", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def should_download_data_lake(env: dict[str, str] | None = None) -> bool:
    source = env or os.environ
    raw = source.get("DOWNLOAD_DATA_LAKE", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def should_cleanup_data_lake(env: dict[str, str] | None = None) -> bool:
    source = env or os.environ
    raw = source.get("CLEANUP_LOCAL_DATA", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_subdirs(raw: str | None, default: Iterable[str]) -> list[str]:
    if not raw:
        return [s for s in default]
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return [p for p in parts if p in VALID_SUBDIRS]


def _resolve_subdirs(
    subdirs: Iterable[str] | None, env: dict[str, str], mode: str
) -> list[str]:
    if subdirs is not None:
        resolved = [str(s).strip().lower() for s in subdirs if str(s).strip()]
    else:
        if mode == "download":
            resolved = _parse_subdirs(env.get("DATA_LAKE_DOWNLOAD_SUBDIRS"), ["gold"])
        else:
            resolved = _parse_subdirs(env.get("DATA_LAKE_SUBDIRS"), VALID_SUBDIRS)
    resolved = [s for s in resolved if s in VALID_SUBDIRS]
    if not resolved:
        raise RuntimeError("Subdirs do data lake nao configurados.")
    return resolved


def _get_s3_client(env: dict[str, str], create_bucket: bool) -> tuple[object, str, str]:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("boto3 nao disponivel para sync do data lake.") from exc

    endpoint = env.get("DATA_LAKE_S3_ENDPOINT") or env.get("MLFLOW_S3_ENDPOINT_URL")
    access_key = env.get("AWS_ACCESS_KEY_ID")
    secret_key = env.get("AWS_SECRET_ACCESS_KEY")
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("Credenciais S3 ou endpoint do data lake nao configurados.")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    bucket = env.get("DATA_LAKE_BUCKET", "openf1-datalake")
    prefix = env.get("DATA_LAKE_PREFIX", "openf1")

    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        if create_bucket:
            client.create_bucket(Bucket=bucket)
        else:
            raise RuntimeError(f"Bucket do data lake nao encontrado: {bucket}")

    return client, bucket, prefix


def _upload_dir(client: object, bucket: str, prefix: str, root: Path, rel_prefix: str) -> int:
    count = 0
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root).as_posix()
        parts = [part for part in [prefix, rel_prefix, rel] if part]
        key = "/".join(parts)
        client.upload_file(str(file_path), bucket, key)
        count += 1
    return count


def _download_prefix(
    client: object, bucket: str, prefix: str, dest_root: Path
) -> int:
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key") or ""
            if not key or key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/")
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            count += 1
    return count


def _dir_has_entries(path: Path) -> bool:
    try:
        return path.exists() and any(path.iterdir())
    except Exception:
        return False


def sync_data_lake(
    data_dir: Path,
    env: dict[str, str] | None = None,
    subdirs: Iterable[str] | None = None,
) -> dict[str, int]:
    source = env or os.environ
    if not should_sync_data_lake(source):
        return {}
    client, bucket, prefix = _get_s3_client(source, create_bucket=True)
    logger = logging.getLogger(__name__)
    results: dict[str, int] = {}
    for subdir in _resolve_subdirs(subdirs, source, mode="upload"):
        root = data_dir / subdir
        if not root.exists():
            continue
        files = _upload_dir(client, bucket, prefix, root, subdir)
        logger.info("Data lake sync %s: %s arquivos", root, files)
        results[subdir] = files
    return results


def download_data_lake(
    data_dir: Path,
    env: dict[str, str] | None = None,
    subdirs: Iterable[str] | None = None,
    only_if_missing: bool = True,
) -> dict[str, int]:
    source = env or os.environ
    if not should_download_data_lake(source):
        return {}
    create_bucket = source.get("DATA_LAKE_CREATE_BUCKET", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    client, bucket, prefix = _get_s3_client(source, create_bucket=create_bucket)
    logger = logging.getLogger(__name__)
    results: dict[str, int] = {}
    for subdir in _resolve_subdirs(subdirs, source, mode="download"):
        root = data_dir / subdir
        if only_if_missing and _dir_has_entries(root):
            continue
        root.mkdir(parents=True, exist_ok=True)
        prefix_key = f"{prefix.rstrip('/')}/{subdir}/"
        files = _download_prefix(client, bucket, prefix_key, root)
        logger.info("Data lake download %s: %s arquivos", root, files)
        results[subdir] = files
    return results
