"""Compare the two authorization formats shown in Terna's official examples."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from .api import ApiRequest, DataAPIError, SecureTransport, _secret, plan_downloads
from .terna import TOKEN_URL, parse_terna_json, parse_token


def diagnose_terna(config: Mapping[str, Any], *, interactive: bool = False,
                   transport: SecureTransport | None = None) -> dict:
    """Use one token exchange and two one-day GETs, with no caches or body output.

    This explicit diagnostic compares the general guide's Bearer format with
    the Total Load page's raw-token example. It changes no downloader setting,
    follows no redirects and makes no automatic retries on denied requests.
    """
    settings = config.get("api", {})
    plan = next((item for item in plan_downloads(config)
                 if item["identity"]["provider"] == "terna"), None)
    if plan is None:
        raise DataAPIError("Configure at least one Terna request for diagnose-terna")
    http_settings = {**settings.get("http", {}), "attempts": 1}
    transport = transport or SecureTransport(**http_settings)
    credentials = settings.get("credentials", {})
    client_id = _secret(credentials.get("terna_client_id_env", "TERNA_CLIENT_ID"), interactive=interactive)
    client_secret = _secret(credentials.get("terna_client_secret_env", "TERNA_CLIENT_SECRET"), interactive=interactive)
    try:
        payload = transport.post_form(TOKEN_URL, {"grant_type": "client_credentials",
            "client_id": client_id, "client_secret": client_secret})
    except DataAPIError as error:
        raise DataAPIError(f"Terna authentication/token diagnostic failed: {error}") from None
    token, _ = parse_token(payload)
    specification: ApiRequest = plan["specification"]
    checks = []
    # Both variants use the same host, endpoint, date, zone and issued token.
    for scheme in ("bearer", "raw"):
        business_id = str(uuid4())
        request = replace(specification, headers={**specification.headers, "businessID": business_id})
        result = {"authorization_scheme": scheme, "business_id": business_id}
        try:
            response = transport.get(request, bearer_token=token, authorization_scheme=scheme)
            frame = parse_terna_json(response, dataset=plan["identity"]["dataset"],
                options=plan["identity"]["options"], retrieved_at=pd.Timestamp.now(tz="UTC"))
            result.update(status="valid_response", records=len(frame))
        except ValueError as error:
            match = re.search(r"HTTP (\d{3})", str(error)) if isinstance(error, DataAPIError) else None
            # Only our transport/parser's sanitized messages are exposed.
            result.update(status="failed", http_status=int(match[1]) if match else None,
                          error=str(error) if isinstance(error, DataAPIError) else "Response did not match the documented Terna data schema")
        checks.append(result)
    working = [check["authorization_scheme"] for check in checks if check["status"] == "valid_response"]
    return {"status": "checked", "token_exchange": "successful", "endpoint": specification.url,
            "dataset": plan["identity"]["dataset"], "local_day": plan["identity"]["local_day"],
            "query": dict(specification.params), "checks": checks,
            "working_authorization_schemes": working,
            "interpretation": "Validated responses establish a working format for this request only; both failures leave provider policy or other request requirements unresolved",
            "secrets": "Credentials, tokens, request authorization headers and response bodies are omitted; no cache or configuration was changed"}
