"""Cache-first provider downloads with explicit time and credential boundaries."""

from __future__ import annotations

from contextlib import nullcontext

from dataclasses import asdict, dataclass, field
import base64
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
import pandas as pd
import requests


class DataAPIError(ValueError):
    """An API request cannot be completed without inventing or exposing inputs."""


_SECRET_KEYS = {"securitytoken", "token", "access_token", "client_secret", "password", "authorization", "api_key"}
_OFFICIAL_HOSTS = {"web-api.tp.entsoe.eu", "api.terna.it", "api.mercatoelettrico.org"}


def _http_error_hint(payload: bytes) -> str:
    """Return fixed categories only; provider body strings never enter diagnostics."""
    sample=payload[:65536].decode("utf-8",errors="replace").lower()
    categories=(
        ("rate_limit_or_quota",("rate limit","ratelimit","quota","too many requests","throttl","spike arrest","limit exceeded")),
        ("invalid_or_expired_token",("invalid_token","invalidaccesstoken","invalid access token","expired_token","access_token_expired","access token expired","expired access token")),
        ("insufficient_scope_or_subscription",("insufficient_scope","insufficient scope","not subscribed","subscription required","not authorized to access")),
        ("gateway_or_policy_block",("request blocked","request was blocked","request rejected","access denied")),
    )
    return next((category for category,patterns in categories if any(pattern in sample for pattern in patterns)),"unspecified")


def _check_nonsecret(value: Any) -> None:
    """Reject credential-bearing mappings before creating cache identities."""
    if isinstance(value,Mapping):
        for key,child in value.items():
            folded=re.sub(r"[^a-z0-9]","",str(key).lower())
            if folded in {"securitytoken","token","accesstoken","clientsecret","clientid","password","authorization","apikey","secret","login","username","userid"}:
                raise DataAPIError("Credentials must be supplied through configured environment variables, outside request options")
            _check_nonsecret(child)
    elif isinstance(value,(list,tuple)):
        for child in value:
            _check_nonsecret(child)


@dataclass(frozen=True)
class ApiRequest:
    """A nonsecret request specification, safe to include in cache provenance."""

    url: str
    params: Mapping[str, Any] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        address = urlsplit(self.url)
        if address.scheme != "https" or address.hostname not in _OFFICIAL_HOSTS or address.query or address.fragment or address.username:
            raise DataAPIError("Provider requests require an official HTTPS endpoint without embedded credentials/query")
        if any(str(key).lower() in _SECRET_KEYS for key in (*self.params, *self.headers)):
            raise DataAPIError("Credentials must remain transport-only, outside request specifications")
        _check_nonsecret(self.params)
        _check_nonsecret(self.headers)


class SecureTransport:
    """Bounded retries without logging credential-bearing URLs or response text."""

    def __init__(self, *, session=None, attempts: int = 3, connect_timeout: float = 10,
                 read_timeout: float = 60, retry_delay_seconds: float = 1,
                 minimum_request_interval_seconds: float = 0.25,
                 terna_minimum_request_interval_seconds: float | None = None,
                 max_retry_delay_seconds: float = 10,
                 retry_callback: Callable[[str],None] | None = None):
        values = (connect_timeout, read_timeout, retry_delay_seconds, minimum_request_interval_seconds,max_retry_delay_seconds)
        if terna_minimum_request_interval_seconds is not None:
            values += (terna_minimum_request_interval_seconds,)
        if not isinstance(attempts, int) or not 1 <= attempts <= 10 or any(not np.isfinite(v) or v < 0 for v in values) or not connect_timeout or not read_timeout:
            raise DataAPIError("Invalid API timeout/retry settings")
        self.session = session or requests.Session()
        self.attempts = attempts
        self.timeout = (connect_timeout, read_timeout)
        self.retry_delay = retry_delay_seconds
        self.max_retry_delay = max_retry_delay_seconds
        self.retry_callback = retry_callback
        self.minimum_interval = minimum_request_interval_seconds
        self.terna_minimum_interval = (minimum_request_interval_seconds if terna_minimum_request_interval_seconds is None
                                       else max(minimum_request_interval_seconds,terna_minimum_request_interval_seconds))
        self.last_request = 0.0

    def _retry(self, provider: str, attempt: int, reason: str, delay: float) -> None:
        """Expose bounded retry progress using fixed nonsecret context only."""
        delay=min(self.max_retry_delay,delay)
        if self.retry_callback:
            self.retry_callback(f"[retry {attempt+2}/{self.attempts}] {provider}: {reason}; waiting {delay:g} seconds")
        time.sleep(delay)

    def _request(self, method: str, specification: ApiRequest, *, query_secrets=None,
                 bearer_token: str | None = None, authorization_scheme: str = "bearer", form=None,
                 json_payload=None) -> bytes:
        if authorization_scheme not in {"bearer", "raw"}:
            raise DataAPIError("Authorization scheme must be bearer or raw")
        if authorization_scheme == "raw" and urlsplit(specification.url).hostname != "api.terna.it":
            raise DataAPIError("Raw authorization is supported only for documented Terna requests")
        params = dict(specification.params)
        params.update(query_secrets or {})
        headers = dict(specification.headers)
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}" if authorization_scheme == "bearer" else bearer_token
        provider = urlsplit(specification.url).hostname
        # Match the successful diagnostic's documented support header. Generate it
        # only in transport so correlation IDs never change reusable cache keys.
        if provider == "api.terna.it" and method == "GET" and not any(
                str(name).lower() == "businessid" for name in headers):
            headers["businessID"] = str(uuid4())
        business_id = next((str(value) for name, value in headers.items()
                           if str(name).lower() == "businessid"), "")
        support_reference = (f" businessID={business_id}." if re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", business_id) else "")
        request_interval=self.terna_minimum_interval if provider=="api.terna.it" else self.minimum_interval
        for attempt in range(self.attempts):
            delay = request_interval-(time.monotonic()-self.last_request)
            if delay > 0:
                time.sleep(delay)
            self.last_request = time.monotonic()
            try:
                body_options = {} if json_payload is None else {"json": json_payload}
                response = self.session.request(method, specification.url, params=params,
                    headers=headers, data=form, timeout=self.timeout, allow_redirects=False, **body_options)
            except requests.RequestException as error:
                if attempt+1 == self.attempts:
                    raise DataAPIError(f"{provider} request failed ({type(error).__name__}) after {self.attempts} attempts; credentials and URL queries omitted") from None
                self._retry(provider,attempt,type(error).__name__,self.retry_delay*2**attempt)
                continue
            try:
                status = response.status_code
                if 200 <= status < 300:
                    return bytes(response.content)
                if status==400 and provider=="web-api.tp.entsoe.eu":
                    # ENTSO-E also encodes explicit absence as HTTP 400. Let its
                    # strict parser identify no-data, without exposing server text.
                    try:
                        document=ET.fromstring(response.content)
                        for element in document.iter():
                            element.tag=element.tag.split("}")[-1]
                        absent=any(reason.findtext("code")=="999" and
                            (reason.findtext("text") or "").strip().lower().startswith("no matching data")
                            for reason in document.iter("Reason"))
                    except ET.ParseError:
                        absent=False
                    if absent:
                        return bytes(response.content)
                retryable = status in {429,500,502,503,504}
                if retryable and attempt+1 < self.attempts:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = float(retry_after) if re.fullmatch(r"\d+(\.\d+)?", retry_after) else self.retry_delay*2**attempt
                    self._retry(provider,attempt,f"HTTP {status}",delay)
                    continue
                suffix = "check provider credentials/subscriptions" if status in {401,403} else "verify dataset/options and provider status"
                hint=_http_error_hint(response.content)
                attempts_note=f" after {attempt+1} attempts" if attempt else ""
                raise DataAPIError(f"{provider} returned HTTP {status}{attempts_note}; {suffix}.{support_reference} provider_hint={hint} (heuristic). Response body and URL queries omitted")
            finally:
                response.close()
        raise DataAPIError("API retry budget exhausted")

    def get(self, specification: ApiRequest, *, query_secrets=None, bearer_token=None,
            authorization_scheme: str = "bearer") -> bytes:
        """Send ephemeral query/Bearer credentials only to the verified provider."""
        return self._request("GET",specification,query_secrets=query_secrets,bearer_token=bearer_token,
                             authorization_scheme=authorization_scheme)

    def post_form(self, url: str, fields: Mapping[str,str]) -> bytes:
        """Authenticate without persisting the form, token response, or exceptions."""
        return self._request("POST",ApiRequest(url),form=dict(fields))

    def post_json(self, specification: ApiRequest | str, fields: Mapping[str, Any] | None = None,
                  *, bearer_token: str | None = None) -> bytes:
        """Send public request fields or private login fields only in the body.

        Existing request specifications stay unchanged, preserving daily cache
        identities. Authentication fields belong in this ephemeral argument,
        never in the public specification used for cache provenance.
        """
        original = ApiRequest(specification) if isinstance(specification, str) else specification
        body = dict(original.params) if fields is None else dict(fields)
        request = ApiRequest(original.url, headers=original.headers)
        return self._request("POST", request, json_payload=body, bearer_token=bearer_token)


def _json_value(value):
    if isinstance(value,(datetime,pd.Timestamp,Path)):
        return str(value)
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _json_bytes(value) -> bytes:
    return (json.dumps(value,indent=2,sort_keys=True,default=_json_value,allow_nan=False)+"\n").encode()


def _atomic_bytes(path: Path, value: bytes) -> None:
    """Publish complete cache files; a manifest is written only after its payload."""
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=".download-",delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    temporary.replace(path)


class DownloadCache:
    """Small per-provider/dataset/day caches with raw and typed normalized data."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def location(self, identity: Mapping[str,Any]) -> Path:
        for name in ("provider","dataset","zone"):
            if not re.fullmatch(r"[A-Za-z0-9_-]+",str(identity[name])):
                raise DataAPIError("Unsafe provider/dataset cache component")
        digest = hashlib.sha256(_json_bytes(dict(identity))).hexdigest()
        day = identity.get("local_day",pd.Timestamp(identity["start"]).date().isoformat())
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}",day):
            raise DataAPIError("Invalid cache day")
        return self.directory/identity["provider"]/identity["dataset"]/identity["zone"]/day/digest[:20]

    def load(self, identity: Mapping[str,Any]) -> tuple[pd.DataFrame,dict] | None:
        directory = self.location(identity)
        if not (directory/"manifest.json").is_file():
            return None
        manifest = json.loads((directory/"manifest.json").read_text())
        if manifest.get("request") != dict(identity):
            raise DataAPIError(f"Cache identity mismatch at {directory}")
        expected_raw="response.xml" if identity["provider"]=="entsoe" else "response.json"
        if manifest.get("raw_file") != expected_raw:
            raise DataAPIError(f"Unexpected raw payload filename in cache at {directory}")
        try:
            raw = (directory/manifest["raw_file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=manifest["raw_sha256"]:
                raise DataAPIError(f"Cached API payload checksum mismatch at {directory}")
            normalized=(directory/"records.parquet").read_bytes()
            if hashlib.sha256(normalized).hexdigest()!=manifest["normalized_sha256"]:
                raise DataAPIError(f"Cached normalized records checksum mismatch at {directory}")
            frame = pd.read_parquet(directory/"records.parquet")
        except (KeyError,OSError,ValueError) as error:
            if isinstance(error,DataAPIError):
                raise
            raise DataAPIError(f"Incomplete/corrupt API cache at {directory}; refresh this request explicitly") from None
        manifest["directory"]=str(directory)
        return frame,manifest

    def load_raw(self, identity: Mapping[str,Any]) -> bytes:
        """Read the source payload after the caller has verified its snapshot."""
        directory = self.location(identity)
        raw_file = "response.xml" if identity["provider"] == "entsoe" else "response.json"
        return (directory / raw_file).read_bytes()

    def load_no_data(self, identity: Mapping[str,Any], max_age_hours: float) -> dict | None:
        """Explicit absence has a bounded lifetime; it never becomes zero values."""
        receipt_path=self.location(identity)/"no_data.json"
        if not receipt_path.is_file() or max_age_hours<=0:
            return None
        receipt=json.loads(receipt_path.read_text())
        if receipt.get("request")!=dict(identity):
            raise DataAPIError("No-data receipt identity mismatch")
        age=(pd.Timestamp.now(tz="UTC")-pd.Timestamp(receipt["retrieved_at"])).total_seconds()/3600
        return receipt if 0<=age<max_age_hours else None

    def load_unparsed(self, identity: Mapping[str,Any]) -> tuple[bytes,dict] | None:
        """Reuse a retained data response after a parser fix, without credentials."""
        directory=self.location(identity)
        receipt_path=directory/"schema_error.json"
        if not receipt_path.is_file():
            return None
        try:
            receipt=json.loads(receipt_path.read_text())
            payload=(directory/"unparsed_response.json").read_bytes()
            if receipt["request"]!=dict(identity) or hashlib.sha256(payload).hexdigest()!=receipt["raw_sha256"]:
                raise ValueError
        except (KeyError,ValueError,OSError):
            raise DataAPIError(f"Invalid retained API response at {directory}; refresh this request explicitly") from None
        return payload,receipt

    def save_unparsed(self, identity: Mapping[str,Any], payload: bytes, retrieved_at,
                      sensitive_values=()) -> Path:
        """Retain malformed data only after checking for in-memory credentials."""
        secrets=[str(value) for value in sensitive_values if value]
        def contains_secret(value):
            if isinstance(value,str):
                return any(secret in value for secret in secrets)
            if isinstance(value,dict):
                return any(contains_secret(key) or contains_secret(child) for key,child in value.items())
            if isinstance(value,list):
                return any(contains_secret(child) for child in value)
            return False
        try:
            decoded_contains_secret=contains_secret(json.loads(payload))
        except (ValueError,UnicodeError):
            decoded_contains_secret=False
        if decoded_contains_secret or any(value.encode() in payload for value in secrets):
            raise DataAPIError("Terna returned authentication material in its data response; the response was not retained")
        directory=self.location(identity)
        directory.mkdir(parents=True,exist_ok=True)
        _atomic_bytes(directory/"unparsed_response.json",payload)
        receipt={"request":dict(identity),"status":"unparsed-data", "retrieved_at":str(retrieved_at),
                 "raw_sha256":hashlib.sha256(payload).hexdigest(),
                 "interpretation":"Retained provider data awaiting a schema correction; not a successful normalized cache or a no-data receipt"}
        _atomic_bytes(directory/"schema_error.json",_json_bytes(receipt))
        return directory/"unparsed_response.json"

    def save_no_data(self, identity: Mapping[str,Any]) -> dict:
        """Record a sanitized absence receipt, preserving any prior valid snapshot."""
        directory=self.location(identity)
        directory.mkdir(parents=True,exist_ok=True)
        receipt={"request":dict(identity),"status":"no-data","records":0,
            "retrieved_at":pd.Timestamp.now(tz="UTC").isoformat(),"directory":str(directory),
            "interpretation":"no rows for the requested series after provider parsing/filtering; not zero values or inferred historical coverage"}
        _atomic_bytes(directory/"no_data.json",_json_bytes(receipt))
        return receipt

    def save(self, identity: Mapping[str,Any], payload: bytes, frame: pd.DataFrame,
             metadata: Mapping[str,Any]) -> dict:
        directory = self.location(identity)
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/"manifest.json").is_file():
            previous=json.loads((directory/"manifest.json").read_text())
            previous_time=pd.Timestamp(previous["cached_at"]).strftime("%Y%m%dT%H%M%S%fZ")
            archive=directory/"revisions"/previous_time
            archive.mkdir(parents=True,exist_ok=True)
            for name in ("response.xml","response.json","records.parquet","records.csv","values.png"):
                if (directory/name).is_file():
                    shutil.copy2(directory/name,archive/name)
            previous["directory"]=str(archive)
            _atomic_bytes(archive/"manifest.json",_json_bytes(previous))
        raw_file = "response.xml" if identity["provider"]=="entsoe" else "response.json"
        _atomic_bytes(directory/raw_file,payload)
        normalized=frame.to_parquet(index=False)
        _atomic_bytes(directory/"records.parquet",normalized)
        _atomic_bytes(directory/"records.csv",frame.to_csv(index=False).encode())
        manifest = {"schema_version":"api-cache-v1","request":dict(identity),
            "raw_file":raw_file,"raw_sha256":hashlib.sha256(payload).hexdigest(),
            "normalized_sha256":hashlib.sha256(normalized).hexdigest(),
            "records":len(frame),"cached_at":datetime.now(timezone.utc).isoformat(),
            "directory":str(directory),"metadata":dict(metadata),
            "historical_availability":"source provenance only; download/cache time does not establish historical publication"}
        _atomic_bytes(directory/"manifest.json",_json_bytes(manifest))
        (directory/"no_data.json").unlink(missing_ok=True)
        (directory/"values.png").unlink(missing_ok=True)
        return manifest


def cache_from_location(location: str | Path, *, threads: int = 4):
    """Select the consolidated database or legacy directory without opening it."""
    path = Path(location)
    if path.suffix.lower() == ".duckdb":
        from .duckdb_store import DuckDBCache
        return DuckDBCache(path, threads=threads)
    return DownloadCache(path)


def cache_from_settings(settings: Mapping[str, Any]):
    """Resolve explicit acquisition storage while retaining old configurations."""
    backend = settings.get("storage_backend", "files")
    if backend == "duckdb":
        path = Path(settings.get("database_path", "data/api/market.duckdb"))
        if path.suffix.lower() != ".duckdb":
            raise DataAPIError("api.database_path must end in .duckdb")
        return cache_from_location(path, threads=settings.get("database_threads", 4))
    if backend != "files":
        raise DataAPIError("api.storage_backend must be files or duckdb")
    return cache_from_location(settings.get("cache_directory", "data/api"))


def _secret(environment_name: str, *, interactive: bool) -> str:
    """Cached requests never reach this function; unattended misses never prompt."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",environment_name):
        raise DataAPIError("Credential environment variable name is invalid")
    secret = os.environ.get(environment_name)
    if secret:
        return secret
    if interactive and sys.stdin.isatty():
        secret = getpass.getpass(f"{environment_name}: ")
        if secret:
            return secret
    raise DataAPIError(f"Missing credential environment variable {environment_name}; set it or run download-data --interactive in a terminal")


def _utc(value: Any, local_timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise DataAPIError("API dates must not be missing")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(local_timezone,ambiguous="raise",nonexistent="raise")
    return timestamp.tz_convert("UTC")


def _daily_windows(start, end, local_timezone):
    """Split at local midnights, preserving 23/25-hour DST days."""
    current = _utc(start,local_timezone)
    stop = _utc(end,local_timezone)
    if current >= stop:
        raise DataAPIError("API start must precede exclusive end")
    while current < stop:
        next_day = current.tz_convert(local_timezone).date()+timedelta(days=1)
        boundary = pd.Timestamp(next_day,tz=local_timezone).tz_convert("UTC")
        yield current,min(stop,boundary)
        current = min(stop,boundary)


def _token_refresh_age(token: str, maximum_age_seconds: float) -> float:
    """Refresh before an optional JWT expiry without trusting it as authentication.

    GME verifies its token on data requests. Decoding here merely bounds the
    local refresh clock; missing or malformed claims use a configured limit.
    """
    try:
        encoded = token.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if not isinstance(payload, dict):
            return maximum_age_seconds
        expiry = payload.get("exp")
        if isinstance(expiry, bool) or not isinstance(expiry, (int, float)) or not np.isfinite(expiry):
            return maximum_age_seconds
        return max(0., min(maximum_age_seconds, expiry - time.time() - 30.))
    except (IndexError, ValueError, TypeError, UnicodeError):
        return maximum_age_seconds


def plan_downloads(config: Mapping[str,Any]) -> list[dict]:
    """Resolve daily requests, completing every dataset in each local month first."""
    settings = config.get("api",{})
    if not all(name in settings for name in ("start","end","requests")):
        raise DataAPIError("Configure api.start, api.end (exclusive), and api.requests")
    download_order = settings.get("download_order","month_first")
    if download_order not in {"month_first","dataset_first"}:
        raise DataAPIError("api.download_order must be month_first or dataset_first")
    if settings.get("terna_authorization_scheme", "bearer") not in {"bearer", "raw"}:
        raise DataAPIError("api.terna_authorization_scheme must be bearer or raw")
    local_timezone = settings.get("timezone","Europe/Rome")
    plans=[]
    for task in settings["requests"]:
        provider,dataset = task["provider"],task["dataset"]
        options = dict(task.get("options",{}))
        _check_nonsecret(options)
        zone = str(options.get("zone","NORD"))
        for start,end in _daily_windows(settings["start"],settings["end"],local_timezone):
            if provider=="entsoe":
                if set(options)-{"zone","zone_eic"}:
                    raise DataAPIError("Unknown ENTSO-E request options; only zone and zone_eic are supported")
                from .entsoe import build_request, dataset_metadata
                url,params = build_request(dataset,options.get("zone_eic","10Y1001A1001A73I"),start,end)
                specification = ApiRequest(url,params)
                metadata = dataset_metadata(dataset)
            elif provider=="terna":
                from .terna import request_spec, dataset_metadata
                if any(value.tz_convert("Europe/Rome") != value.tz_convert("Europe/Rome").normalize()
                       for value in (start,end)):
                    raise DataAPIError("Terna requests require complete Europe/Rome calendar days; use api.timezone Europe/Rome and date-only bounds")
                first_day=start.tz_convert("Europe/Rome").date()
                last_day=(end-pd.Timedelta(nanoseconds=1)).tz_convert("Europe/Rome").date()
                specification = request_spec(dataset,first_day,last_day,options)
                metadata = dataset_metadata(dataset)
            elif provider=="gme":
                if set(options) - {"zone", "granularity_type"}:
                    raise DataAPIError("Unknown GME request options; only zone and granularity_type are supported")
                from .gme import ZONES, build_request, dataset_metadata
                if zone not in ZONES or (zone == "PUN" and dataset != "day_ahead_prices"):
                    raise DataAPIError("GME requires a supported zonal code; PUN is available only for MGP")
                url, body = build_request(dataset, start, end, granularity_type=options.get("granularity_type"))
                specification = ApiRequest(url, body)
                metadata = dataset_metadata(dataset)
            else:
                raise DataAPIError(f"Unsupported API provider {provider!r}")
            identity = {"provider":provider,"dataset":dataset,"zone":zone,
                "start":start.isoformat(),"end":end.isoformat(),"options":options,
                "local_day":start.tz_convert(local_timezone).date().isoformat(),"timezone":local_timezone,
                "request":asdict(specification),"schema_version":"api-cache-v1",
                "vintage_revision":task.get("vintage_revision","latest_at_retrieval")}
            plans.append({"identity":identity,"specification":specification,"metadata":metadata})
    if not plans:
        raise DataAPIError("No API datasets requested")
    # Stable ordering preserves configured dataset order within each local month.
    # Scheduling is deliberately outside the identity so existing daily caches reuse.
    if download_order=="month_first":
        plans.sort(key=lambda plan:plan["identity"]["local_day"][:7])
    return plans


def _plot_records(frame: pd.DataFrame, path: Path, title: str) -> None:
    """Plot native returned values without manufacturing sub-hourly observations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clock = next((name for name in ("product_start","reference_time")
                  if name in frame and frame[name].notna().any()),None)
    quantity = next((name for name in ("value","price_eur_mwh","value_mw") if name in frame),None)
    if clock is None or quantity is None or frame.empty:
        return
    fig,axis=plt.subplots(figsize=(11,4),layout="constrained")
    dimensions=[name for name in ("zone","psr_type","direction","session_type","energy_source")
                if name in frame and frame[name].notna().any()]
    groups = frame.groupby(dimensions,dropna=False) if dimensions else [("returned data",frame)]
    for label,group in groups:
        axis.plot(pd.to_datetime(group[clock],utc=True),group[quantity],marker=".",label=str(label))
    axis.set_title(title+"; native source values")
    axis.set_xlabel("UTC source time"); axis.set_ylabel(str(frame.unit.iloc[0]) if "unit" in frame else quantity)
    axis.legend(); fig.savefig(path,dpi=120); plt.close(fig)


def download_from_config(config: Mapping[str,Any], *, interactive: bool = False,
                         dry_run: bool = False, refresh_cache: bool = False,
                         transport: SecureTransport | None = None,
                         progress: Callable[[str],None] | None = None) -> dict:
    """Fetch only missing daily requests and export normalized source/provenance."""
    plans = plan_downloads(config)
    settings = config["api"]
    cache = cache_from_settings(settings)
    consolidated = cache.directory.suffix.lower() == ".duckdb"
    plot_groups = {}
    no_data_hours=float(settings.get("no_data_cache_hours",24))
    if not np.isfinite(no_data_hours) or no_data_hours<0:
        raise DataAPIError("api.no_data_cache_hours must be finite and nonnegative")
    if dry_run:
        with cache.read_session() if consolidated else nullcontext():
            return {"status":"planned","requests":len(plans),
                "download_order":settings.get("download_order","month_first"),"items":[
                {"request":item["identity"],"cached":cache.load(item["identity"]) is not None,
                 "no_data_cached":cache.load_no_data(item["identity"],no_data_hours) is not None,
                 "dataset_metadata":item["metadata"]} for item in plans]}
    transport = transport or SecureTransport(**{**settings.get("http",{}),"retry_callback":progress})
    credentials = settings.get("credentials",{})
    secret_values={}
    def credential(name):
        if name not in secret_values:
            secret_values[name]=_secret(name,interactive=interactive)
        return secret_values[name]
    terna_token=None
    terna_token_expires=0.0
    gme_token=None
    gme_token_expires=0.0
    gme_refresh_seconds=settings.get("gme_token_refresh_seconds",900.)
    if isinstance(gme_refresh_seconds,bool) or not np.isfinite(float(gme_refresh_seconds)) or float(gme_refresh_seconds)<=0:
        raise DataAPIError("api.gme_token_refresh_seconds must be finite and positive")
    raw_responses={}
    results=[]
    http_gets,auth_requests=0,0
    progress_every=int(settings.get("progress_every",25))
    if progress_every<1:
        raise DataAPIError("api.progress_every must be a positive integer")
    progress_group=None
    def announce(identity):
        nonlocal progress_group
        group=(identity["local_day"][:7],identity["provider"],identity["dataset"])
        if progress and (group!=progress_group or len(results)%progress_every==0 or len(results)==len(plans)):
            item=results[-1]
            progress(f"[{len(results)}/{len(plans)}] month {group[0]} {item['provider']}/{item['dataset']} {identity['local_day']}: {item['status']}, {item['records']} rows")
        progress_group=group
    for plan in plans:
        identity,specification=plan["identity"],plan["specification"]
        cached = None if refresh_cache else cache.load(identity)
        no_data = None if refresh_cache or cached is not None else cache.load_no_data(identity,no_data_hours)
        if no_data is not None:
            results.append({"provider":identity["provider"],"dataset":identity["dataset"],
                "start":identity["start"],"end":identity["end"],"status":"no-data-cached",
                "records":0,"historically_eligible_records":0,"directory":no_data["directory"]})
            announce(identity)
            continue
        if cached is not None:
            frame,manifest=cached
            raw_key=_json_bytes({"provider":identity["provider"],"specification":asdict(specification),
                                 "vintage_revision":identity["vintage_revision"],"schema_version":identity["schema_version"]})
            raw_responses[raw_key]=(cache.load_raw(identity),
                                    pd.Timestamp(manifest["metadata"]["retrieved_at"]))
            status="cached"
        else:
            provider=identity["provider"]
            raw_key=_json_bytes({"provider":provider,"specification":asdict(specification),
                                 "vintage_revision":identity["vintage_revision"],"schema_version":identity["schema_version"]})
            reused=raw_responses.get(raw_key)
            if provider in {"terna", "gme"} and reused is None and not refresh_cache:
                retained=cache.load_unparsed(identity)
                if retained is not None:
                    reused=(retained[0],pd.Timestamp(retained[1]["retrieved_at"]))
            if provider=="entsoe":
                from .entsoe import EntsoeNoDataError,parse_entsoe_xml
                if reused is None:
                    token=credential(credentials.get("entsoe_token_env","ENTSOE_API_TOKEN"))
                    payload=transport.get(specification,query_secrets={"securityToken":token})
                    http_gets+=1
                    retrieved_at=pd.Timestamp.now(tz="UTC")
                else:
                    payload,retrieved_at=reused
                try:
                    frame=parse_entsoe_xml(payload,dataset=identity["dataset"],zone=identity["zone"],
                        zone_eic=identity["options"].get("zone_eic","10Y1001A1001A73I"),retrieved_at=retrieved_at)
                except EntsoeNoDataError:
                    frame=pd.DataFrame()
            elif provider=="terna":
                from .terna import TOKEN_URL, parse_token, parse_terna_json
                if reused is None and (terna_token is None or time.monotonic() >= terna_token_expires):
                    client_id=credential(credentials.get("terna_client_id_env","TERNA_CLIENT_ID"))
                    client_secret=credential(credentials.get("terna_client_secret_env","TERNA_CLIENT_SECRET"))
                    try:
                        token_payload=transport.post_form(TOKEN_URL,{"grant_type":"client_credentials", "client_id":client_id,"client_secret":client_secret})
                    except DataAPIError as error:
                        raise DataAPIError(f"Terna authentication/token request failed (application Key and Secret): {error}") from None
                    auth_requests+=1
                    terna_token,expires_seconds=parse_token(token_payload)
                    terna_token_expires=time.monotonic()+max(0,expires_seconds-30)
                if reused is None:
                    try:
                        auth_options = ({"authorization_scheme": "raw"}
                                        if settings.get("terna_authorization_scheme", "bearer") == "raw" else {})
                        payload=transport.get(specification,bearer_token=terna_token,**auth_options)
                    except DataAPIError as error:
                        raise DataAPIError(f"Terna data request for {identity['dataset']} on {identity['local_day']} failed after obtaining an access token: {error}") from None
                    http_gets+=1
                    retrieved_at=pd.Timestamp.now(tz="UTC")
                else:
                    payload,retrieved_at=reused
                try:
                    frame=parse_terna_json(payload,dataset=identity["dataset"],options=identity["options"],retrieved_at=retrieved_at,
                        requested_day=identity["local_day"])
                except ValueError as error:
                    retained_path=cache.save_unparsed(identity,payload,retrieved_at,
                        sensitive_values=[*secret_values.values(),terna_token])
                    raise DataAPIError(f"Terna {identity['dataset']} on {identity['local_day']}: {error}. Unparsed data retained at {retained_path}; a parser fix can reuse it without downloading again") from None
            else:
                from .gme import TOKEN_URL, GMENoDataError, parse_token_response, parse_gme_response, validate_sanitized_response
                if reused is None and (gme_token is None or time.monotonic() >= gme_token_expires):
                    login=credential(credentials.get("gme_username_env","GME_USERNAME"))
                    password=credential(credentials.get("gme_password_env","GME_PASSWORD"))
                    try:
                        token_payload=transport.post_json(TOKEN_URL,{"Login":login,"Password":password})
                        gme_token=parse_token_response(token_payload)
                    except ValueError as error:
                        raise DataAPIError(f"GME authentication failed (GME-issued username/password): {error}") from None
                    auth_requests+=1
                    gme_token_expires=time.monotonic()+_token_refresh_age(gme_token,float(gme_refresh_seconds))
                if reused is None:
                    try:
                        payload=transport.post_json(specification,bearer_token=gme_token)
                    except DataAPIError as error:
                        raise DataAPIError(f"GME data request for {identity['dataset']} on {identity['local_day']} failed: {error}") from None
                    http_gets+=1
                    retrieved_at=pd.Timestamp.now(tz="UTC")
                else:
                    payload,retrieved_at=reused
                private_values=[*secret_values.values(),gme_token]
                try:
                    frame=parse_gme_response(payload,dataset=identity["dataset"],zone=identity["zone"],
                        retrieved_at=retrieved_at,start=identity["start"],end=identity["end"],
                        granularity_type=identity["options"].get("granularity_type"),sensitive_values=private_values)
                except GMENoDataError:
                    frame=pd.DataFrame()
                except ValueError as error:
                    # Successful HTTP data may be a schema variant. Keep it only
                    # after inspecting the decoded archive for credential echoes.
                    try:
                        validate_sanitized_response(payload,sensitive_values=private_values)
                    except ValueError:
                        raise DataAPIError("GME data response could not be safely decoded for retention; no response was cached") from None
                    retained_path=cache.save_unparsed(identity,payload,retrieved_at,sensitive_values=private_values)
                    raise DataAPIError(f"GME {identity['dataset']} on {identity['local_day']}: {error}. Unparsed data retained at {retained_path}; a parser fix can reuse it without downloading again") from None
            raw_responses[raw_key]=(payload,retrieved_at)
            if frame.empty:
                receipt=cache.save_no_data(identity)
                results.append({"provider":provider,"dataset":identity["dataset"],
                    "start":identity["start"],"end":identity["end"],"status":"no-data",
                    "records":0,"historically_eligible_records":0,"directory":receipt["directory"]})
                announce(identity)
                continue
            manifest=cache.save(identity,payload,frame,{**plan["metadata"],"frame_provenance":frame.attrs,
                "acquisition":"reused_verified_raw_snapshot" if reused is not None else "http",
                "retrieved_at":retrieved_at.isoformat()})
            status="downloaded"
        directory=Path(manifest["directory"])
        if settings.get("plots",True):
            if consolidated:
                # Database records share one plot per dataset and local month.
                key = tuple(identity[name] for name in ("provider", "dataset", "zone")) + (identity["local_day"][:7],)
                plot_groups.setdefault(key, []).append(frame)
            elif not (directory/"values.png").is_file():
                _plot_records(frame,directory/"values.png",f"{identity['provider']} {identity['dataset']} {identity['start'][:10]}")
        eligible=int(frame.historical_feature_eligible.fillna(False).sum()) if "historical_feature_eligible" in frame else 0
        results.append({"provider":identity["provider"],"dataset":identity["dataset"],
            "start":identity["start"],"end":identity["end"],"status":status,
            "records":len(frame),"historically_eligible_records":eligible,"directory":str(directory)})
        announce(identity)
    output=Path(settings.get("output_directory","outputs/api"))
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination=output/run_id
    destination.mkdir(parents=True,exist_ok=False)
    plot_paths = []
    for (provider, dataset, zone, month), frames in sorted(plot_groups.items()):
        path = destination / "plots" / provider / dataset / zone / f"{month}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        _plot_records(pd.concat(frames, ignore_index=True), path, f"{provider} {dataset} {zone} {month}")
        plot_paths.append(str(path))
    gaps=sum(item["status"].startswith("no-data") for item in results)
    summary={"status":"complete-with-gaps" if gaps else "complete","run_id":run_id,"requested_chunks":len(plans),
        "storage_backend":"duckdb" if consolidated else "files", "cache_location":str(cache.directory),
        "plots":plot_paths,
        "download_order":settings.get("download_order","month_first"),
        "downloaded_chunks":sum(item["status"]=="downloaded" for item in results),
        "cached_chunks":sum(item["status"]=="cached" for item in results),"items":results,
        "no_data_chunks":gaps,
        "data_http_operations":http_gets,"authentication_operations":auth_requests,
        "http_operation_convention":"Each operation may include bounded retry attempts",
        "publication_policy":"Unverified source creation or retrieval times are not historical publication times",
        "manifest":str(destination/"manifest.json")}
    _atomic_bytes(destination/"manifest.json",_json_bytes(summary))
    return summary
