"""The ``feedbackAuth`` a provider signs to let the consumer rate it (ERC-8004).

Its format is the one ReverseAuction.sol expects: 224 bytes of
``abi.encode(agentId, clientAddress, indexLimit, expiry, chainId,
identityRegistry, signerAddress)`` followed by a 65-byte EIP-191 signature.
"""

import logging
from typing import Tuple
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
from eth_abi import encode

logger = logging.getLogger(__name__)


def generate_feedback_auth(
    agent_id: int,
    client_address: str,
    index_limit: int,
    expiry: int,
    chain_id: int,
    identity_registry_address: str,
    signer_address: str,
    private_key: str
) -> bytes:
    """Encode the seven parameters, sign their keccak256 hash with the EIP-191 prefix,
    and return parameters plus signature (289 bytes).

    ``private_key`` must belong to ``signer_address``.
    """
    try:
        client_address = Web3.to_checksum_address(client_address)
        identity_registry_address = Web3.to_checksum_address(identity_registry_address)
        signer_address = Web3.to_checksum_address(signer_address)

        logger.debug(f"Generating feedbackAuth for agent {agent_id}")
        logger.debug(f"  Client: {client_address}")
        logger.debug(f"  Expiry: {expiry}")
        logger.debug(f"  Chain ID: {chain_id}")

        # Types: (uint256, address, uint64, uint256, uint256, address, address).
        encoded_params = encode(
            ['uint256', 'address', 'uint64', 'uint256', 'uint256', 'address', 'address'],
            [agent_id, client_address, index_limit, expiry, chain_id, identity_registry_address, signer_address]
        )

        logger.debug(f"  Encoded params length: {len(encoded_params)} bytes")

        message_hash = Web3.keccak(encoded_params)
        logger.debug(f"  Message hash: {message_hash.hex()}")

        # EIP-191 prefix, as MessageHashUtils.toEthSignedMessageHash(keccak256(...)).
        signable_message = encode_defunct(primitive=message_hash)

        account = Account.from_key(private_key)

        if account.address != signer_address:
            raise ValueError(
                f"Private key does not match signer address. "
                f"Expected: {signer_address}, Got: {account.address}"
            )

        signed_message = account.sign_message(signable_message)

        # Signature layout: r (32 bytes), s (32 bytes), v (1 byte).
        signature = signed_message.signature  # 65 bytes

        logger.debug(f"  Signature length: {len(signature)} bytes")
        logger.debug(f"  Signature: {signature.hex()}")

        feedback_auth = encoded_params + signature

        logger.info(f"✅ Generated feedbackAuth: {len(feedback_auth)} bytes")

        if len(feedback_auth) != 289:
            raise ValueError(
                f"Invalid feedbackAuth length: {len(feedback_auth)} bytes (expected 289)"
            )

        return feedback_auth

    except Exception as e:
        logger.error(f"Error generating feedbackAuth: {e}", exc_info=True)
        raise


def parse_feedback_auth(feedback_auth: bytes) -> Tuple[dict, bytes]:
    """Split a feedbackAuth into its decoded parameters and its signature."""
    if len(feedback_auth) < 289:
        raise ValueError(f"Invalid feedbackAuth length: {len(feedback_auth)} (expected >= 289)")

    encoded_params = feedback_auth[:224]
    signature = feedback_auth[224:289]

    from eth_abi import decode
    decoded = decode(
        ['uint256', 'address', 'uint64', 'uint256', 'uint256', 'address', 'address'],
        encoded_params
    )

    params = {
        'agent_id': decoded[0],
        'client_address': decoded[1],
        'index_limit': decoded[2],
        'expiry': decoded[3],
        'chain_id': decoded[4],
        'identity_registry_address': decoded[5],
        'signer_address': decoded[6]
    }

    return params, signature


def verify_feedback_auth_format(feedback_auth: bytes) -> bool:
    """Whether a feedbackAuth has the expected length and decodes."""
    try:
        if len(feedback_auth) < 289:
            logger.error(f"Invalid length: {len(feedback_auth)} bytes (expected >= 289)")
            return False

        params, signature = parse_feedback_auth(feedback_auth)

        if len(signature) != 65:
            logger.error(f"Invalid signature length: {len(signature)} bytes (expected 65)")
            return False

        logger.info("✅ FeedbackAuth format is valid")
        logger.debug(f"  Agent ID: {params['agent_id']}")
        logger.debug(f"  Client: {params['client_address']}")
        logger.debug(f"  Expiry: {params['expiry']}")
        logger.debug(f"  Chain ID: {params['chain_id']}")

        return True

    except Exception as e:
        logger.error(f"Format verification failed: {e}")
        return False
