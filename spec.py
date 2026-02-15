"""Pydantic models describing the AbletonOSC API spec."""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel


class ParamSpec(BaseModel):
    name: str
    type: Literal["int", "float", "str", "bool", "any"]
    description: str = ""
    optional: bool = False


class EndpointSpec(BaseModel):
    address: str  # "/live/song/set/tempo"
    domain: str  # "song"
    kind: Literal["method", "get", "set", "listen_start", "listen_stop", "custom"]
    params: list[ParamSpec] = []
    returns: list[ParamSpec] = []
    description: str = ""
    wildcard: bool = False  # supports track_id=*


class DomainSpec(BaseModel):
    name: str  # "song"
    description: str
    base_address: str  # "/live/song"
    index_params: list[ParamSpec] = []
    endpoints: list[EndpointSpec]


class AbletonOSCSpec(BaseModel):
    version: str = "1.0"
    source: str = "AbletonOSC (remix-mcp fork)"
    domains: list[DomainSpec]

    def search(self, query: str) -> list[EndpointSpec]:
        """Case-insensitive substring match across address, description, domain name."""
        q = query.lower()
        results = []
        for domain in self.domains:
            for ep in domain.endpoints:
                if (
                    q in ep.address.lower()
                    or q in ep.description.lower()
                    or q in domain.name.lower()
                ):
                    results.append(ep)
        return results

    def get_domain(self, name: str) -> Optional[DomainSpec]:
        """Return a single domain by name."""
        for d in self.domains:
            if d.name == name:
                return d
        return None

    def get_endpoint(self, address: str) -> Optional[EndpointSpec]:
        """Exact address lookup."""
        for d in self.domains:
            for ep in d.endpoints:
                if ep.address == address:
                    return ep
        return None

    def list_domains(self) -> list[str]:
        """Return domain names."""
        return [d.name for d in self.domains]

    def by_kind(self, kind: str) -> list[EndpointSpec]:
        """Filter all endpoints by kind (method/get/set/listen_start/listen_stop/custom)."""
        return [ep for d in self.domains for ep in d.endpoints if ep.kind == kind]

    def for_domain(self, name: str) -> list[EndpointSpec]:
        """All endpoints in a domain."""
        d = self.get_domain(name)
        return d.endpoints if d else []

    def to_json(self) -> str:
        """Serialize the populated spec to JSON."""
        return json.dumps(self.model_dump(), indent=2)
