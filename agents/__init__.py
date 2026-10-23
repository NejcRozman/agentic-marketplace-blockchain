"""The marketplace agents, built with LangGraph.

``common/``    blockchain client, contract ABIs, IPFS, feedback authorisation, dataset replay
``consumer/``  the consumer agent: publishes jobs, runs their auctions, rates the results
``provider/``  the provider agent: bidding strategies, cost estimators, service execution
``config.py``  configuration shared by all agents
"""

__version__ = "0.1.0"

from .config import config
from .common import (
    BlockchainClient,
    IPFSClient,
    IPFSUploadResult,
    ServiceDescription,
    create_ipfs_client,
    contract_abis,
    get_reverse_auction_abi,
    get_identity_registry_abi,
    get_reputation_registry_abi,
    generate_feedback_auth,
    parse_feedback_auth,
    verify_feedback_auth_format
)
from .provider import (
    ServiceExecutor,
    BlockchainHandler,
    AgentState
)

__all__ = [
    "config",
    "BlockchainClient",
    "IPFSClient",
    "IPFSUploadResult",
    "ServiceDescription",
    "create_ipfs_client",
    "contract_abis",
    "get_reverse_auction_abi",
    "get_identity_registry_abi",
    "get_reputation_registry_abi",
    "generate_feedback_auth",
    "parse_feedback_auth",
    "verify_feedback_auth_format",
    "ServiceExecutor",
    "BlockchainHandler",
    "AgentState",
    "__version__"
]
