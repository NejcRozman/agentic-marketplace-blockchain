"""
IPFS Client using Pinata for decentralized storage.

Provides functionality for:
- Uploading JSON metadata to IPFS
- Uploading files (PDFs, etc.) to IPFS
- Fetching content from IPFS via public gateways
- Creating service description bundles with attachments
"""

import logging
import json
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

# Pinata API endpoints
PINATA_API_URL = "https://api.pinata.cloud"
PINATA_PIN_JSON_URL = f"{PINATA_API_URL}/pinning/pinJSONToIPFS"
PINATA_PIN_FILE_URL = f"{PINATA_API_URL}/pinning/pinFileToIPFS"

# Public IPFS gateways for reading (in order of preference)
IPFS_GATEWAYS = [
    "https://gateway.pinata.cloud/ipfs/",
    "https://ipfs.io/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://dweb.link/ipfs/",
]

# Maximum file size (10MB) and count limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILE_COUNT = 5


@dataclass
class IPFSUploadResult:
    """Result of an IPFS upload operation."""
    success: bool
    cid: Optional[str] = None
    error: Optional[str] = None
    size: Optional[int] = None
    name: Optional[str] = None


@dataclass
class ServiceDescription:
    """Service description structure for auction creation."""
    title: str
    description: str
    prompts: List[str]  # Questions/prompts for the agent to answer
    input_files_cid: Optional[str] = None  # CID of input files directory (e.g., PDFs)
    requirements: Optional[Dict[str, Any]] = None  # Optional technical requirements
    complexity: str = "medium"  # low, medium, high
    deadline: Optional[str] = None  # ISO 8601 format
    customer_endpoint: Optional[str] = None  # A2A endpoint for result delivery
    attachment_cids: Optional[List[str]] = None  # CIDs of additional files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
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
    """
    IPFS client using Pinata for pinning content.
    
    Supports uploading JSON metadata and files, and fetching content
    from IPFS via public gateways.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        jwt: Optional[str] = None
    ):
        """
        Initialize the IPFS client.
        
        Args:
            api_key: Pinata API key (or from env PINATA_API_KEY)
            api_secret: Pinata API secret (or from env PINATA_API_SECRET)
            jwt: Pinata JWT token (or from env PINATA_JWT) - preferred auth method
        """
        import os
        
        self.jwt = jwt or os.getenv("PINATA_JWT")
        self.api_key = api_key or os.getenv("PINATA_API_KEY")
        self.api_secret = api_secret or os.getenv("PINATA_API_SECRET")
        
        if not self.jwt and not (self.api_key and self.api_secret):
            logger.warning("Pinata credentials not configured. Upload operations will fail.")
        
        logger.info("IPFSClient initialized")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Pinata API."""
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
        """
        Pin JSON data to IPFS via Pinata.
        
        Args:
            data: JSON-serializable dictionary to upload
            name: Optional name for the pin
            metadata: Optional key-value metadata for the pin
            
        Returns:
            IPFSUploadResult with CID if successful
        """
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
        """
        Pin a file to IPFS via Pinata.
        
        Args:
            file_path: Path to the file to upload
            name: Optional name for the pin (defaults to filename)
            metadata: Optional key-value metadata for the pin
            
        Returns:
            IPFSUploadResult with CID if successful
        """
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
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            file_handle = open(file_path, 'rb')
            data.add_field(
                'file',
                file_handle,
                filename=file_name,
                content_type='application/octet-stream'
            )
            
            # Add metadata if provided
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
        """
        Pin multiple files to IPFS.
        
        Args:
            file_paths: List of file paths to upload (max 5)
            metadata: Optional metadata to apply to all files
            
        Returns:
            List of IPFSUploadResult for each file
        """
        if len(file_paths) > MAX_FILE_COUNT:
            logger.warning(f"Too many files ({len(file_paths)}), limiting to {MAX_FILE_COUNT}")
            file_paths = file_paths[:MAX_FILE_COUNT]
        
        results = []
        for file_path in file_paths:
            result = await self.pin_file(file_path, metadata=metadata)
            results.append(result)
        
        return results
    
    async def create_service_description(
        self,
        service: ServiceDescription,
        attachment_files: Optional[List[Union[str, Path]]] = None
    ) -> IPFSUploadResult:
        """
        Create and pin a complete service description with optional attachments.
        
        This is the main method for consumers creating auctions:
        1. Uploads any attachment files first
        2. Adds attachment CIDs to service description
        3. Uploads the complete service description JSON
        
        Args:
            service: ServiceDescription object with service details
            attachment_files: Optional list of files to attach (max 5)
            
        Returns:
            IPFSUploadResult with CID of the service description JSON
        """
        try:
            # Upload attachments first if provided
            if attachment_files:
                logger.info(f"Uploading {len(attachment_files)} attachments...")
                attachment_results = await self.pin_files(attachment_files)
                
                # Collect successful CIDs
                attachment_cids = []
                for result in attachment_results:
                    if result.success and result.cid:
                        attachment_cids.append(result.cid)
                    else:
                        logger.warning(f"Failed to upload attachment: {result.error}")
                
                if attachment_cids:
                    service.attachment_cids = attachment_cids
            
            # Convert service description to dict and upload
            service_data = service.to_dict()
            
            result = await self.pin_json(
                data=service_data,
                name=f"service-{service.title[:30]}",
                metadata={"type": "service_description", "complexity": service.complexity}
            )
            
            if result.success:
                logger.info(f"✅ Service description created: {result.cid}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating service description: {e}")
            return IPFSUploadResult(success=False, error=str(e))
    
    async def fetch(
        self,
        cid: str,
        as_json: bool = True,
        timeout: int = 30
    ) -> Optional[Union[Dict[str, Any], bytes]]:
        """
        Fetch content from IPFS via public gateways.
        
        Args:
            cid: The IPFS content identifier
            as_json: If True, parse response as JSON; if False, return raw bytes
            timeout: Request timeout in seconds
            
        Returns:
            Parsed JSON dict, raw bytes, or None if fetch failed
        """
        for gateway in IPFS_GATEWAYS:
            url = f"{gateway}{cid}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status == 200:
                            if as_json:
                                data = await response.json()
                                logger.debug(f"Fetched JSON from IPFS: {cid}")
                                return data
                            else:
                                data = await response.read()
                                logger.debug(f"Fetched {len(data)} bytes from IPFS: {cid}")
                                return data
                        else:
                            logger.debug(f"Gateway {gateway} returned {response.status}")
                            continue
                            
            except Exception as e:
                logger.debug(f"Gateway {gateway} failed: {e}")
                continue
        
        logger.error(f"Failed to fetch {cid} from all gateways")
        return None
    
    async def fetch_json(self, cid: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Fetch and parse JSON content from IPFS.
        
        Args:
            cid: The IPFS content identifier
            timeout: Request timeout in seconds
            
        Returns:
            Parsed JSON dict or None if fetch failed
        """
        return await self.fetch(cid, as_json=True, timeout=timeout)
    
    async def fetch_file(self, cid: str, timeout: int = 60) -> Optional[bytes]:
        """
        Fetch raw file content from IPFS.
        
        Args:
            cid: The IPFS content identifier
            timeout: Request timeout in seconds
            
        Returns:
            Raw bytes or None if fetch failed
        """
        return await self.fetch(cid, as_json=False, timeout=timeout)
    
    async def download_file(self, cid: str, output_path: Union[str, Path], timeout: int = 60) -> bool:
        """
        Download a file from IPFS and save it to local filesystem.
        
        Args:
            cid: The IPFS content identifier
            output_path: Local path to save the file
            timeout: Request timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = await self.fetch_file(cid, timeout=timeout)
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
    
    def get_gateway_url(self, cid: str, gateway_index: int = 0) -> str:
        """
        Get a public gateway URL for a CID.
        
        Args:
            cid: The IPFS content identifier
            gateway_index: Which gateway to use (0 = Pinata, preferred)
            
        Returns:
            Full URL to access the content
        """
        gateway = IPFS_GATEWAYS[gateway_index % len(IPFS_GATEWAYS)]
        return f"{gateway}{cid}"


# Convenience function for quick access
def create_ipfs_client() -> IPFSClient:
    """Create an IPFSClient with credentials from environment."""
    return IPFSClient()
