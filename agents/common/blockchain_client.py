"""Asynchronous client for the chain: contract calls, transactions, receipts and events."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from aiohttp import ClientTimeout
from web3 import AsyncWeb3
from eth_account import Account
from eth_utils import to_checksum_address

# Per-request HTTP timeout. web3's default of 30 s misreports a single slow poll
# of wait_for_transaction_receipt as "not in the chain after <timeout> seconds",
# because aiohttp's timeout is an asyncio.TimeoutError.
HTTP_REQUEST_TIMEOUT_SECONDS = 90

try:
    from ..config import config
except ImportError:
    from config import Config
    config = Config()

logger = logging.getLogger(__name__)


class BlockchainClient:
    """Asynchronous web3 client with retries on transient RPC failures."""
    
    def __init__(self, blockchain_config = None):
        self.config = blockchain_config or config
        self.w3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.contracts: Dict[str, Any] = {}
        self._initialized = False
    
    async def __aenter__(self):
        await self._initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
    
    async def _initialize(self):
        """Connect to the RPC endpoint and set up the account, if a key is given."""
        if self._initialized:
            return
        
        try:
            self.w3 = AsyncWeb3(
                AsyncWeb3.AsyncHTTPProvider(
                    self.config.rpc_url,
                    request_kwargs={"timeout": ClientTimeout(total=HTTP_REQUEST_TIMEOUT_SECONDS)},
                )
            )
                       
            if self.config.private_key:
                self.account = Account.from_key(self.config.private_key)
                logger.info(f"Initialized account: {self.account.address}")
            
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
        try:
            if not self.w3:
                return False
            await self.w3.eth.block_number
            return True
        except Exception:
            return False
    
    async def get_balance(self, address: Optional[str] = None) -> int:
        """Native-token balance of an address, in wei."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        addr = address or (self.account.address if self.account else None)
        if not addr:
            raise ValueError("No address provided and no account configured")
        
        balance = await self.w3.eth.get_balance(to_checksum_address(addr))
        return balance
    
    async def get_gas_price(self) -> int:
        """The configured gas price, or the node's."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        gas_price = getattr(self.config, 'gas_price', None)
        if gas_price:
            return gas_price
        
        return await self.w3.eth.gas_price
    
    async def load_contract(self, name: str, address: str, abi: List[Dict]) -> Any:
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
        """Call a read-only contract method, retrying transient failures.

        A ``.call()`` has no side effects, so retrying is safe.
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
        """Build, sign and send a transaction to a contract method, retrying nonce races."""
        if not self.account:
            raise RuntimeError("No account configured for sending transactions")
        
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")
        
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        contract = self.contracts[contract_name]
        method = getattr(contract.functions, method_name)

        # Nonce races are common when one key is used by several processes.
        last_error = None
        for attempt in range(3):
            try:
                transaction = await method(*args, **kwargs).build_transaction({
                    'from': self.account.address,
                    'value': value,
                    'gas': gas_limit or self.config.gas_limit,
                    'gasPrice': gas_price or await self.get_gas_price(),
                    'nonce': await self.w3.eth.get_transaction_count(self.account.address, 'pending'),
                    'chainId': self.config.chain_id,
                })

                signed_txn = self.account.sign_transaction(transaction)

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
        """Wait for a transaction's receipt, retrying transient RPC errors.

        The transaction is already broadcast, so retrying the wait (not the send) is safe.
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
        """Contract events in a block range, via eth_getLogs."""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")
        
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        contract = self.contracts[contract_name]
        event = getattr(contract.events, event_name)
        
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
        """Estimate a transaction's gas, retrying transient failures (a read-only simulation)."""
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
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        block = await self.w3.eth.get_block(block_identifier)
        return dict(block)
    
    async def get_block_number(self) -> int:
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        return await self.w3.eth.block_number
    
    async def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        tx = await self.w3.eth.get_transaction(tx_hash)
        return dict(tx)
    
    async def close(self):
        if self.w3 and self.w3.provider:
            try:
                await self.w3.provider.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting provider: {e}")
        self._initialized = False
        logger.info("BlockchainClient closed")