"""IPFS through Pinata: pinning job descriptions and files, and reading them back.

Reading goes through the gateway named by ``IPFS_GATEWAY`` (with
``PINATA_GATEWAY_TOKEN`` if it has access controls) and then public gateways.
"""

import asyncio
import logging
import json
import time
import aiohttp
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

try:
    from ..config import config
except ImportError:
    from config import Config
    config = Config()

logger = logging.getLogger(__name__)

PINATA_API_URL = "https://api.pinata.cloud"
PINATA_PIN_JSON_URL = f"{PINATA_API_URL}/pinning/pinJSONToIPFS"
PINATA_PIN_FILE_URL = f"{PINATA_API_URL}/pinning/pinFileToIPFS"

# Public IPFS gateways, tried after the configured one (IPFS_GATEWAY). Pinata's
# public gateway no longer serves content for reading, so a dedicated gateway
# (https://<name>.mypinata.cloud/ipfs/) is the reliable choice.
IPFS_GATEWAYS = [
    "https://ipfs.io/ipfs/",
    "https://dweb.link/ipfs/",
]

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_FILE_COUNT = 5


@dataclass
class IPFSUploadResult:
    success: bool
    cid: Optional[str] = None
    error: Optional[str] = None
    size: Optional[int] = None
    name: Optional[str] = None


@dataclass
class ServiceDescription:
    """A job description as published for an auction."""
    title: str
    description: str
    prompts: List[str]
    input_files_cid: Optional[str] = None  # CID of the input files (e.g. PDFs)
    requirements: Optional[Dict[str, Any]] = None
    complexity: str = "medium"  # low, medium or high
    deadline: Optional[str] = None  # ISO 8601
    customer_endpoint: Optional[str] = None
    attachment_cids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "title": self.title,
            "description": self.description,
            "prompts": self.prompts,
            "complexity": self.complexity,
        }
        if self.input_files_cid:
            data["input_files_cid"] = self.input_files_cid
        if self.requirements:
            data["requirements"] = self.requirements
        if self.deadline:
            data["deadline"] = self.deadline
        if self.customer_endpoint:
            data["customer_endpoint"] = self.customer_endpoint
        if self.attachment_cids:
            data["attachments"] = self.attachment_cids
        return data


class IPFSClient:
    """Pin content to IPFS through Pinata and read it back through gateways."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        jwt: Optional[str] = None
    ):
        """Credentials default to PINATA_JWT (preferred) or PINATA_API_KEY and PINATA_API_SECRET."""
        import os

        self.jwt = jwt or os.getenv("PINATA_JWT")
        self.api_key = api_key or os.getenv("PINATA_API_KEY")
        self.api_secret = api_secret or os.getenv("PINATA_API_SECRET")
        # Gateway for reading, and its access key if it has access controls.
        gateway = (os.getenv("IPFS_GATEWAY") or "").strip()
        if gateway and not gateway.endswith("/"):
            gateway += "/"
        self.gateway = gateway or None
        self.gateway_token = (os.getenv("PINATA_GATEWAY_TOKEN") or "").strip() or None

        if not self.jwt and not (self.api_key and self.api_secret):
            logger.warning("Pinata credentials not configured. Upload operations will fail.")

        logger.info("IPFSClient initialized")

    def _get_auth_headers(self) -> Dict[str, str]:
        if self.jwt:
            return {"Authorization": f"Bearer {self.jwt}"}
        elif self.api_key and self.api_secret:
            return {
                "pinata_api_key": self.api_key,
                "pinata_secret_api_key": self.api_secret
            }
        else:
            raise ValueError("Pinata credentials not configured")

    async def pin_json(
        self,
        data: Dict[str, Any],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> IPFSUploadResult:
        """Pin a JSON object; returns the result with its CID."""
        try:
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"

            payload = {
                "pinataContent": data
            }

            if name or metadata:
                payload["pinataMetadata"] = {}
                if name:
                    payload["pinataMetadata"]["name"] = name
                if metadata:
                    payload["pinataMetadata"]["keyvalues"] = metadata

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    PINATA_PIN_JSON_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        cid = result.get("IpfsHash")
                        size = result.get("PinSize", 0)

                        logger.info(f"✅ Pinned JSON to IPFS: {cid}")
                        return IPFSUploadResult(
                            success=True,
                            cid=cid,
                            size=size,
                            name=name
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"Pinata error: {response.status} - {error_text}")
                        return IPFSUploadResult(
                            success=False,
                            error=f"Pinata API error: {response.status}"
                        )

        except Exception as e:
            logger.error(f"Error pinning JSON: {e}")
            return IPFSUploadResult(success=False, error=str(e))

    async def pin_file(
        self,
        file_path: Union[str, Path],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> IPFSUploadResult:
        """Pin a file; the pin is named after the file unless ``name`` is given."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return IPFSUploadResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )

            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                return IPFSUploadResult(
                    success=False,
                    error=f"File too large: {file_size} bytes (max {MAX_FILE_SIZE})"
                )

            file_name = name or file_path.name
            headers = self._get_auth_headers()

            data = aiohttp.FormData()
            file_handle = open(file_path, 'rb')
            data.add_field(
                'file',
                file_handle,
                filename=file_name,
                content_type='application/octet-stream'
            )

            pinata_metadata = {"name": file_name}
            if metadata:
                pinata_metadata["keyvalues"] = metadata
            data.add_field(
                'pinataMetadata',
                json.dumps(pinata_metadata),
                content_type='application/json'
            )

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        PINATA_PIN_FILE_URL,
                        headers=headers,
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            cid = result.get("IpfsHash")
                            size = result.get("PinSize", file_size)

                            logger.info(f"✅ Pinned file to IPFS: {file_name} -> {cid}")
                            return IPFSUploadResult(
                                success=True,
                                cid=cid,
                                size=size,
                                name=file_name
                            )
                        else:
                            error_text = await response.text()
                            logger.error(f"Pinata error: {response.status} - {error_text}")
                            return IPFSUploadResult(
                                success=False,
                                error=f"Pinata API error: {response.status}"
                            )
            finally:
                file_handle.close()

        except Exception as e:
            logger.error(f"Error pinning file: {e}")
            return IPFSUploadResult(success=False, error=str(e))

    async def pin_files(
        self,
        file_paths: List[Union[str, Path]],
        metadata: Optional[Dict[str, str]] = None
    ) -> List[IPFSUploadResult]:
        """Pin up to MAX_FILE_COUNT files, each with the same metadata."""
        if len(file_paths) > MAX_FILE_COUNT:
            logger.warning(f"Too many files ({len(file_paths)}), limiting to {MAX_FILE_COUNT}")
            file_paths = file_paths[:MAX_FILE_COUNT]

        results = []
        for file_path in file_paths:
            result = await self.pin_file(file_path, metadata=metadata)
            results.append(result)

        return results

    async def fetch(
        self,
        cid: str,
        as_json: bool = True,
        timeout: int = 30,
        retry_for: float = 0.0,
    ) -> Optional[Union[Dict[str, Any], bytes]]:
        """Fetch content by CID: the configured gateway first, then the public ones.

        ``timeout`` applies per gateway. ``retry_for`` keeps retrying all gateways for that
        many seconds, because freshly pinned content can take a while to become readable.
        Returns parsed JSON (``as_json``) or raw bytes, or None.
        """
        gateways = ([self.gateway] if self.gateway else []) + IPFS_GATEWAYS
        deadline = time.monotonic() + max(0.0, retry_for)
        while True:
            outcomes = []
            for gateway in gateways:
                url = f"{gateway}{cid}"
                headers = (
                    {"x-pinata-gateway-token": self.gateway_token}
                    if gateway == self.gateway and self.gateway_token else None
                )
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            url,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=timeout)
                        ) as response:
                            if response.status == 200:
                                if as_json:
                                    data = await response.json(content_type=None)
                                    logger.debug(f"Fetched JSON from IPFS: {cid}")
                                    return data
                                data = await response.read()
                                logger.debug(f"Fetched {len(data)} bytes from IPFS: {cid}")
                                return data
                            outcomes.append(f"HTTP {response.status}")
                except Exception as e:
                    outcomes.append(type(e).__name__)
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(3)

        logger.error(f"Failed to fetch {cid} from all gateways ({', '.join(outcomes)})")
        return None

    async def fetch_json(self, cid: str, timeout: int = 30, retry_for: float = 0.0) -> Optional[Dict[str, Any]]:
        return await self.fetch(cid, as_json=True, timeout=timeout, retry_for=retry_for)

    async def fetch_file(self, cid: str, timeout: int = 60, retry_for: float = 0.0) -> Optional[bytes]:
        return await self.fetch(cid, as_json=False, timeout=timeout, retry_for=retry_for)

    async def download_file(
        self, cid: str, output_path: Union[str, Path], timeout: int = 60, retry_for: float = 0.0
    ) -> bool:
        """Download a file by CID to ``output_path``; False on failure."""
        try:
            data = await self.fetch_file(cid, timeout=timeout, retry_for=retry_for)
            if data:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'wb') as f:
                    f.write(data)

                logger.info(f"✅ Downloaded {len(data)} bytes from {cid} to {output_path}")
                return True
            else:
                logger.error(f"Failed to fetch file from IPFS: {cid}")
                return False
        except Exception as e:
            logger.error(f"Error downloading file {cid}: {e}")
            return False

def create_ipfs_client() -> IPFSClient:
    return IPFSClient()
