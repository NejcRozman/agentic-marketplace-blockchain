"""Blockchain client for interacting with smart contracts."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from aiohttp import ClientTimeout
from web3 import AsyncWeb3
from eth_account import Account
from eth_utils import to_checksum_address

# web3.py's AsyncHTTPProvider defaults to a 30s per-request HTTP timeout
# (DEFAULT_HTTP_TIMEOUT). aiohttp.ServerTimeoutError is a subclass of
# asyncio.TimeoutError, and wait_for_transaction_receipt's outer
# asyncio.wait_for(..., timeout=...) only catches asyncio.TimeoutError to
# raise TimeExhausted - so a single slow HTTP poll past 30s gets misreported
# as "not in the chain after {timeout} seconds" even when the real elapsed
# time is far short of that. Raising the per-request timeout well past
# observed local-node latency spikes lets a single slow poll succeed instead
# of prematurely aborting the whole wait.
HTTP_REQUEST_TIMEOUT_SECONDS = 90

try:
    from ..config import config
except ImportError:
    from config import Config
    config = Config()

logger = logging.getLogger(__name__)


class BlockchainClient:
    """
    Asynchronous blockchain client for interacting with Ethereum-compatible networks.
    
    Provides methods for:
    - Contract interaction
    - Transaction management
    - Event monitoring
    - Account management
    """
    
    def __init__(self, blockchain_config = None):
        self.config = blockchain_config or config
        self.w3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.contracts: Dict[str, Any] = {}
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False
    
    async def _initialize(self):
        """Initialize the blockchain connection."""
        if self._initialized:
            return
        
        try:
            # Create Web3 instance
            self.w3 = AsyncWeb3(
                AsyncWeb3.AsyncHTTPProvider(
                    self.config.rpc_url,
                    request_kwargs={"timeout": ClientTimeout(total=HTTP_REQUEST_TIMEOUT_SECONDS)},
                )
            )
                       
            # Set up account if private key is provided
            if self.config.private_key:
                self.account = Account.from_key(self.config.private_key)
                logger.info(f"Initialized account: {self.account.address}")
            
            # Verify connection
            if await self.is_connected():
                chain_id = await self.w3.eth.chain_id
                logger.info(f"Connected to blockchain - Chain ID: {chain_id}")
                self._initialized = True
            else:
                logger.error("Failed to connect to blockchain")
                
        except Exception as e:
            logger.error(f"Error initializing blockchain client: {e}")
            raise
    
    async def is_connected(self) -> bool:
        """Check if connected to the blockchain."""
        try:
            if not self.w3:
                return False
            await self.w3.eth.block_number
            return True
        except Exception:
            return False
    
    async def get_balance(self, address: Optional[str] = None) -> int:
        """Get ETH/native token balance for an address."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        addr = address or (self.account.address if self.account else None)
        if not addr:
            raise ValueError("No address provided and no account configured")
        
        balance = await self.w3.eth.get_balance(to_checksum_address(addr))
        return balance
    
    async def get_gas_price(self) -> int:
        """Get current gas price."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        # Use configured gas_price if available
        gas_price = getattr(self.config, 'gas_price', None)
        if gas_price:
            return gas_price
        
        return await self.w3.eth.gas_price
    
    async def load_contract(self, name: str, address: str, abi: List[Dict]) -> Any:
        """Load a smart contract."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        contract = self.w3.eth.contract(
            address=to_checksum_address(address),
            abi=abi
        )
        
        self.contracts[name] = contract
        logger.info(f"Loaded contract {name} at {address}")
        return contract
    
    async def call_contract_method(
        self,
        contract_name: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Call a read-only contract method.

        Retried on failure since a `.call()` has no side effects — forked RPC
        endpoints occasionally fail to fetch a storage slot from the upstream chain
        for a transient reason, and this is the most frequently hit read path in the
        whole client.
        """
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")

        contract = self.contracts[contract_name]
        method = getattr(contract.functions, method_name)

        max_attempts = 3
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                return await method(*args, **kwargs).call()
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Transient error calling {contract_name}.{method_name} "
                        f"(attempt {attempt + 1}/{max_attempts}): {e}. Retrying..."
                    )
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                logger.error(f"Error calling {contract_name}.{method_name}: {e}")

        raise last_error
    
    async def send_transaction(
        self,
        contract_name: str,
        method_name: str,
        *args,
        value: int = 0,
        gas_limit: Optional[int] = None,
        gas_price: Optional[int] = None,
        **kwargs
    ) -> str:
        """Send a transaction to a contract method."""
        if not self.account:
            raise RuntimeError("No account configured for sending transactions")
        
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")
        
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        contract = self.contracts[contract_name]
        method = getattr(contract.functions, method_name)

        # Retry nonce-related races (common when same key is used across processes)
        last_error = None
        for attempt in range(3):
            try:
                # Build transaction
                transaction = await method(*args, **kwargs).build_transaction({
                    'from': self.account.address,
                    'value': value,
                    'gas': gas_limit or self.config.gas_limit,
                    'gasPrice': gas_price or await self.get_gas_price(),
                    'nonce': await self.w3.eth.get_transaction_count(self.account.address, 'pending'),
                    'chainId': self.config.chain_id,
                })

                # Sign transaction
                signed_txn = self.account.sign_transaction(transaction)

                # Send transaction
                tx_hash = await self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)

                logger.info(f"Sent transaction: {tx_hash.hex()}")
                return tx_hash.hex()
            except Exception as e:
                last_error = e
                err_text = str(e).lower()
                if "nonce too low" in err_text or "already known" in err_text or "replacement transaction underpriced" in err_text:
                    logger.warning(f"Transaction nonce race on attempt {attempt + 1}/3: {e}")
                    await asyncio.sleep(0.2)
                    continue
                raise

        raise last_error
    
    async def wait_for_transaction(self, tx_hash: str, timeout: int = 120) -> Dict[str, Any]:
        """Wait for transaction confirmation.

        The transaction was already broadcast before this is called, so retrying the
        wait itself (not resubmitting) is safe. Forked RPC endpoints occasionally hit
        a transient upstream error while proxying to the real chain (e.g. an Infura
        request timeout surfaced as a "Fork Error: Transport(...)" JSON-RPC error) —
        a single such blip during one poll shouldn't abort the whole wait when the
        transaction has very likely already confirmed.
        """
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")

        max_attempts = 3
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)

                return {
                    'transactionHash': receipt['transactionHash'].hex(),
                    'blockNumber': receipt['blockNumber'],
                    'gasUsed': receipt['gasUsed'],
                    'effectiveGasPrice': receipt.get('effectiveGasPrice', 0),
                    'status': receipt['status'],
                    'logs': [dict(log) for log in receipt['logs']]
                }
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Transient error waiting for transaction {tx_hash} "
                        f"(attempt {attempt + 1}/{max_attempts}): {e}. Retrying..."
                    )
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                logger.error(f"Error waiting for transaction {tx_hash}: {e}")

        raise last_error
    
    async def get_contract_events(
        self,
        contract_name: str,
        event_name: str,
        from_block: Union[int, str] = 'latest',
        to_block: Union[int, str] = 'latest',
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Get contract events using eth_getLogs."""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")
        
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        contract = self.contracts[contract_name]
        event = getattr(contract.events, event_name)
        
        # Use get_logs instead of create_filter for async web3
        logs = await event.get_logs(
            from_block=from_block,
            to_block=to_block,
            argument_filters=filters or {}
        )
        
        return [dict(log) for log in logs]
    
    async def estimate_gas(
        self,
        contract_name: str,
        method_name: str,
        *args,
        value: int = 0,
        **kwargs
    ) -> int:
        """Estimate gas for a transaction.

        This is a read-only simulation (no state mutation), so retrying it is always
        safe. Forked RPC endpoints occasionally fail to fetch a storage slot from the
        upstream chain for a transient reason (e.g. "failed to get storage for ...:
        Internal error") — a single such blip shouldn't abort the whole call.
        """
        if not self.account:
            raise RuntimeError("No account configured")

        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")

        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")

        contract = self.contracts[contract_name]
        method = getattr(contract.functions, method_name)

        max_attempts = 3
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                return await method(*args, **kwargs).estimate_gas({
                    'from': self.account.address,
                    'value': value
                })
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Transient error estimating gas for {contract_name}.{method_name} "
                        f"(attempt {attempt + 1}/{max_attempts}): {e}. Retrying..."
                    )
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                logger.error(f"Error estimating gas for {contract_name}.{method_name}: {e}")

        raise last_error
    
    async def get_block(self, block_identifier: Union[int, str] = 'latest') -> Dict[str, Any]:
        """Get block information."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        block = await self.w3.eth.get_block(block_identifier)
        return dict(block)
    
    async def get_block_number(self) -> int:
        """Get the current block number."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        return await self.w3.eth.block_number
    
    async def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction information."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        tx = await self.w3.eth.get_transaction(tx_hash)
        return dict(tx)
    
    async def close(self):
        """Close the blockchain client and release resources."""
        if self.w3 and self.w3.provider:
            try:
                await self.w3.provider.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting provider: {e}")
        self._initialized = False
        logger.info("BlockchainClient closed")