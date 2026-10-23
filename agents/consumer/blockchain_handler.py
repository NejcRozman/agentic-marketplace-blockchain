"""The consumer's chain operations: creating and ending auctions, reading their state, rating providers."""

import logging
from typing import Any, Dict, List, Optional

try:
    from ..config import config
    from ..common.blockchain_client import BlockchainClient
except ImportError:
    from config import Config
    config = Config()
    from common.blockchain_client import BlockchainClient

logger = logging.getLogger(__name__)


class ConsumerBlockchainHandler:
    """The consumer's chain operations, over the ReverseAuction contract and the ERC-8004 registries."""

    def __init__(self, consumer_config=None):
        self.config = consumer_config or config
        self.client = BlockchainClient(self.config)
        self.consumer_agent_id = self.config.consumer_agent_id
        self._initialized = False

        logger.info(f"ConsumerBlockchainHandler created for consumer agent {self.consumer_agent_id}")

    async def initialize(self) -> bool:
        """Connect to the chain and load the contracts; False on failure."""
        if self._initialized:
            return True

        try:
            await self.client._initialize()

            reverse_auction_abi = self._load_reverse_auction_abi()
            await self.client.load_contract(
                "ReverseAuction",
                self.config.reverse_auction_address,
                reverse_auction_abi
            )

            identity_registry_abi = self._load_identity_registry_abi()
            await self.client.load_contract(
                "IdentityRegistry",
                self.config.identity_registry_address,
                identity_registry_abi
            )

            reputation_registry_abi = self._load_reputation_registry_abi()
            await self.client.load_contract(
                "ReputationRegistry",
                self.config.reputation_registry_address,
                reputation_registry_abi
            )

            self._initialized = True
            logger.info("✅ Consumer blockchain handler initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize consumer blockchain handler: {e}")
            return False

    async def approve_tokens(self, amount: int) -> bool:
        """Allow the ReverseAuction contract to take ``amount`` USDC (micro-USDC) into escrow."""
        try:
            if not self._initialized:
                await self.initialize()

            logger.info(f"Approving {amount} USDC for ReverseAuction contract...")

            erc20_abi = [
                {
                    "inputs": [
                        {"name": "spender", "type": "address"},
                        {"name": "amount", "type": "uint256"}
                    ],
                    "name": "approve",
                    "outputs": [{"name": "", "type": "bool"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]

            await self.client.load_contract(
                "USDC",
                self.config.payment_token_address,
                erc20_abi
            )

            tx_hash = await self.client.send_transaction(
                "USDC",
                "approve",
                self.config.reverse_auction_address,
                amount
            )

            receipt = await self.client.wait_for_transaction(tx_hash)

            if receipt['status'] != 1:
                raise Exception("Approval transaction failed")

            logger.info(f"✅ Approved {amount} USDC, tx={tx_hash}")
            return True

        except Exception as e:
            logger.error(f"Failed to approve tokens: {e}")
            return False

    async def create_auction(
        self,
        service_cid: str,
        max_price: int,
        commit_duration: int,
        reveal_duration: int,
        eligible_agent_ids: Optional[List[int]] = None,
        reputation_weight: int = 30
    ) -> Dict[str, Any]:
        """Create an auction and escrow its maximum price.

        ``eligible_agent_ids`` defaults to the configured providers; ``reputation_weight``
        is 0-100. Returns the auction id and transaction hash, or an error.
        """
        try:
            if not self._initialized:
                await self.initialize()

            if eligible_agent_ids is None:
                eligible_agent_ids = self.config.eligible_providers

            logger.info(f"Creating auction: service={service_cid}, max_price={max_price}, "
                       f"commit_duration={commit_duration}s, reveal_duration={reveal_duration}s, "
                       f"eligible_agents={eligible_agent_ids}")

            # The maximum price is escrowed when the auction is created.
            approval_success = await self.approve_tokens(max_price)
            if not approval_success:
                raise Exception("Failed to approve USDC tokens for auction")

            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction",
                "createAuction",
                service_cid,
                max_price,
                commit_duration,
                reveal_duration,
                eligible_agent_ids,
                reputation_weight
            )

            tx_hash = await self.client.send_transaction(
                "ReverseAuction",
                "createAuction",
                service_cid,
                max_price,
                commit_duration,
                reveal_duration,
                eligible_agent_ids,
                reputation_weight,
                gas_limit=estimated_gas + 50000
            )

            receipt = await self.client.wait_for_transaction(tx_hash)

            if receipt['status'] != 1:
                raise Exception("Transaction failed")

            auction_id = await self._extract_auction_id_from_receipt(receipt)

            logger.info(f"✅ Auction created: ID={auction_id}, tx={tx_hash}")

            return {
                "auction_id": auction_id,
                "tx_hash": tx_hash,
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to create auction: {e}")
            return {
                "auction_id": None,
                "tx_hash": None,
                "error": str(e)
            }

    async def end_auction(self, auction_id: int) -> Dict[str, Any]:
        """End an auction; returns the transaction hash, the winner and the winning bid."""
        try:
            if not self._initialized:
                await self.initialize()

            logger.info(f"Ending auction {auction_id}...")

            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction",
                "endAuction",
                auction_id
            )

            tx_hash = await self.client.send_transaction(
                "ReverseAuction",
                "endAuction",
                auction_id,
                gas_limit=estimated_gas + 50000
            )

            receipt = await self.client.wait_for_transaction(tx_hash)

            if receipt['status'] != 1:
                raise Exception("Transaction failed")

            event_data = await self._extract_auction_ended_event(receipt, auction_id)

            logger.info(f"✅ Auction ended: ID={auction_id}, winner={event_data['winning_agent_id']}, "
                       f"bid={event_data['winning_bid']}")

            return {
                "tx_hash": tx_hash,
                "winning_agent_id": event_data['winning_agent_id'],
                "winning_bid": event_data['winning_bid'],
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to end auction {auction_id}: {e}")
            return {
                "tx_hash": None,
                "winning_agent_id": None,
                "winning_bid": None,
                "error": str(e)
            }

    async def get_auction_status(self, auction_id: int) -> Dict[str, Any]:
        """The fields of an auction's on-chain struct, as a dict."""
        try:
            if not self._initialized:
                await self.initialize()

            auction_data = await self.client.call_contract_method(
                "ReverseAuction",
                "getAuctionDetails",
                auction_id
            )

            # Auction struct: id, buyer, serviceDescriptionCid, maxPrice, commitDuration, startTime,
            # eligibleAgentIds, winningAgentId, winningBid, isActive, isCompleted, escrowAmount,
            # reputationWeight, revealDuration.
            start_time = auction_data[5]
            commit_duration = auction_data[4]
            reveal_duration = auction_data[13]
            commit_deadline = start_time + commit_duration
            reveal_deadline = commit_deadline + reveal_duration
            return {
                "id": auction_data[0],
                "buyer": auction_data[1],
                "service_cid": auction_data[2],
                "max_price": auction_data[3],
                "commit_duration": commit_duration,
                "reveal_duration": reveal_duration,
                "start_time": start_time,
                "commit_deadline": commit_deadline,
                "reveal_deadline": reveal_deadline,
                "eligible_agent_ids": auction_data[6],
                "winning_agent_id": auction_data[7],
                "winning_bid": auction_data[8],
                "active": auction_data[9],
                "completed": auction_data[10],
                "escrow_amount": auction_data[11],
                "reputation_weight": auction_data[12],
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to get auction status for {auction_id}: {e}")
            return {
                "error": str(e)
            }

    async def get_auction_phase(self, auction_id: int) -> Optional[int]:
        """An auction's phase: 0 commit, 1 reveal, 2 ended; None on error."""
        try:
            if not self._initialized:
                await self.initialize()

            phase = await self.client.call_contract_method(
                "ReverseAuction",
                "getAuctionPhase",
                auction_id
            )
            return int(phase)

        except Exception as e:
            logger.error(f"Failed to get auction phase for {auction_id}: {e}")
            return None

    async def get_feedback_auth(
        self,
        auction_id: int,
        from_block: Optional[int] = None,
        lookback_blocks: int = 1000
    ) -> Optional[bytes]:
        """The feedbackAuth the winner provided for an auction, from past events; None if absent.

        Searches from ``from_block``, or the last ``lookback_blocks`` blocks.
        """
        try:
            if not self._initialized:
                await self.initialize()

            if from_block is None:
                current_block = await self.client.get_block_number()
                from_block = max(0, current_block - lookback_blocks)

            logger.info(f"Querying FeedbackAuthProvided event for auction {auction_id} from block {from_block}...")

            events = await self.client.get_contract_events(
                "ReverseAuction",
                "FeedbackAuthProvided",
                from_block=from_block,
                to_block="latest",
                filters={"auctionId": auction_id}
            )

            if events:
                feedback_auth = events[0]['args']['feedbackAuth']
                logger.info(f"✅ Found feedbackAuth for auction {auction_id}")
                return feedback_auth
            else:
                logger.warning(f"No FeedbackAuthProvided event found for auction {auction_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to get feedbackAuth for auction {auction_id}: {e}")
            return None

    async def submit_feedback(
        self,
        auction_id: int,
        agent_id: int,
        rating: int,
        feedback_text: str,
        tag1: str = "",
        tag2: str = "",
        endpoint: str = "",
        feedback_uri: str = ""
    ) -> Dict[str, Any]:
        """Rate a provider in the Reputation Registry (ERC-8004 giveFeedback).

        ``rating`` is 0-100 and is stored as value = rating × 100 with two decimals.
        """
        try:
            if not self._initialized:
                await self.initialize()

            if not (0 <= rating <= 100):
                raise ValueError(f"Rating must be 0-100, got {rating}")

            logger.info(f"Submitting feedback for auction {auction_id}, agent {agent_id}, rating {rating}")

            # Two decimals, so the registry's averaging keeps fractions: 85 and 84 average to
            # 84.50 rather than 84.
            value = int(rating * 100)
            value_decimals = 2

            # The feedback is plain text, so there is no file hash.
            feedback_hash = b'\x00' * 32

            estimated_gas = await self.client.estimate_gas(
                "ReputationRegistry",
                "giveFeedback",
                agent_id,
                value,
                value_decimals,
                tag1,
                tag2,
                endpoint,
                feedback_uri,
                feedback_hash
            )

            tx_hash = await self.client.send_transaction(
                "ReputationRegistry",
                "giveFeedback",
                agent_id,
                value,
                value_decimals,
                tag1,
                tag2,
                endpoint,
                feedback_uri,
                feedback_hash,
                gas_limit=estimated_gas + 50000
            )

            receipt = await self.client.wait_for_transaction(tx_hash)

            if receipt['status'] != 1:
                raise Exception("Transaction failed")

            logger.info(f"✅ Feedback submitted: tx={tx_hash}")

            return {
                "tx_hash": tx_hash,
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return {
                "tx_hash": None,
                "error": str(e)
            }

    async def _extract_auction_id_from_receipt(self, receipt: Dict[str, Any]) -> int:
        events = await self.client.get_contract_events(
            "ReverseAuction",
            "AuctionCreated",
            from_block=receipt['blockNumber'],
            to_block=receipt['blockNumber']
        )

        # Both transaction hashes are hex strings.
        receipt_tx_hash = receipt['transactionHash'] if isinstance(receipt['transactionHash'], str) else receipt['transactionHash'].hex()
        for event in events:
            event_tx_hash = event['transactionHash'] if isinstance(event['transactionHash'], str) else event['transactionHash'].hex()
            if event_tx_hash == receipt_tx_hash:
                return event['args']['auctionId']

        raise Exception("AuctionCreated event not found in receipt")

    async def _extract_auction_ended_event(self, receipt: Dict[str, Any], auction_id: int) -> Dict[str, Any]:
        events = await self.client.get_contract_events(
            "ReverseAuction",
            "AuctionEnded",
            from_block=receipt['blockNumber'],
            to_block=receipt['blockNumber'],
            filters={"auctionId": auction_id}
        )

        if not events:
            raise Exception("AuctionEnded event not found in receipt")

        event = events[0]
        return {
            "winning_agent_id": event['args']['winningAgentId'],
            "winning_bid": event['args']['winningBid']
        }

    def _load_reverse_auction_abi(self) -> List[Dict]:
        import json
        from pathlib import Path

        abi_path = Path(__file__).parent.parent.parent / "contracts" / "out" / "ReverseAuction.sol" / "ReverseAuction.json"

        with open(abi_path, 'r') as f:
            contract_json = json.load(f)
            return contract_json['abi']

    def _load_identity_registry_abi(self) -> List[Dict]:
        import json
        from pathlib import Path

        abi_path = Path(__file__).parent.parent.parent / "contracts" / "out" / "IIdentityRegistry.sol" / "IIdentityRegistry.json"

        with open(abi_path, 'r') as f:
            contract_json = json.load(f)
            return contract_json['abi']

    def _load_reputation_registry_abi(self) -> List[Dict]:
        import json
        from pathlib import Path

        abi_path = Path(__file__).parent.parent.parent / "contracts" / "out" / "IReputationRegistry.sol" / "IReputationRegistry.json"

        with open(abi_path, 'r') as f:
            contract_json = json.load(f)
            return contract_json['abi']
