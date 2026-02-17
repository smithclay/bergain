"""Pydantic models describing the AbletonOSC API spec."""

from __future__ import annotations

from typing import Literal

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
