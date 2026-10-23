"""
FeedbackAuth Generator for ERC-8004 Compliance

Generates feedbackAuth in the exact format expected by ReverseAuction.sol:
- First 224 bytes: abi.encode(agentId, clientAddress, indexLimit, expiry, chainId, identityRegistry, signerAddress)
- Last 65 bytes: signature

Matches the official ERC-8004 implementation with EIP-191 prefix.
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
    """
    Generate ERC-8004 compliant feedbackAuth.
    
    This function creates the exact format expected by ReverseAuction.sol:
    1. Encode 7 parameters (224 bytes)
    2. Create message hash with keccak256
    3. Apply EIP-191 prefix
    4. Sign with private key
    5. Concatenate encoded params + signature (224 + 65 = 289 bytes)
    
    Args:
        agent_id: Agent ID from IdentityRegistry
        client_address: Address of the client/buyer
        index_limit: Index limit (uint64)
        expiry: Unix timestamp when auth expires
        chain_id: Blockchain chain ID
        identity_registry_address: Address of IdentityRegistry contract
        signer_address: Address that will sign the message
        private_key: Private key for signing (must match signer_address)
        
    Returns:
        feedbackAuth bytes (289 bytes total)
    """
    try:
        # Normalize addresses to checksum format
        client_address = Web3.to_checksum_address(client_address)
        identity_registry_address = Web3.to_checksum_address(identity_registry_address)
        signer_address = Web3.to_checksum_address(signer_address)
        
        logger.debug(f"Generating feedbackAuth for agent {agent_id}")
        logger.debug(f"  Client: {client_address}")
        logger.debug(f"  Expiry: {expiry}")
        logger.debug(f"  Chain ID: {chain_id}")
        
        # Step 1: Encode the 7 parameters (results in 224 bytes)
        # Types: (uint256, address, uint64, uint256, uint256, address, address)
        encoded_params = encode(
            ['uint256', 'address', 'uint64', 'uint256', 'uint256', 'address', 'address'],
            [agent_id, client_address, index_limit, expiry, chain_id, identity_registry_address, signer_address]
        )
        
        logger.debug(f"  Encoded params length: {len(encoded_params)} bytes")
        
        # Step 2: Hash the encoded parameters
        message_hash = Web3.keccak(encoded_params)
        logger.debug(f"  Message hash: {message_hash.hex()}")
        
        # Step 3: Apply EIP-191 prefix (Ethereum Signed Message)
        # This matches: MessageHashUtils.toEthSignedMessageHash(keccak256(...))
        signable_message = encode_defunct(primitive=message_hash)
        
        # Step 4: Sign the message
        account = Account.from_key(private_key)
        
        # Verify the signing account matches expected signer
        if account.address != signer_address:
            raise ValueError(
                f"Private key does not match signer address. "
                f"Expected: {signer_address}, Got: {account.address}"
            )
        
        signed_message = account.sign_message(signable_message)
        
        # Extract signature components (v, r, s) and combine into 65 bytes
        # Format: [r (32 bytes)][s (32 bytes)][v (1 byte)]
        signature = signed_message.signature  # Already 65 bytes
        
        logger.debug(f"  Signature length: {len(signature)} bytes")
        logger.debug(f"  Signature: {signature.hex()}")
        
        # Step 5: Concatenate encoded params (224 bytes) + signature (65 bytes)
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
    """
    Parse feedbackAuth into its components.
    
    Useful for debugging and verification.
    
    Args:
        feedback_auth: The complete feedbackAuth bytes
        
    Returns:
        Tuple of (parameters dict, signature bytes)
    """
    if len(feedback_auth) < 289:
        raise ValueError(f"Invalid feedbackAuth length: {len(feedback_auth)} (expected >= 289)")
    
    # Split into params and signature
    encoded_params = feedback_auth[:224]
    signature = feedback_auth[224:289]
    
    # Decode parameters
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
    """
    Verify that feedbackAuth has the correct format.
    
    Args:
        feedback_auth: The feedbackAuth bytes to verify
        
    Returns:
        True if format is valid, False otherwise
    """
    try:
        if len(feedback_auth) < 289:
            logger.error(f"Invalid length: {len(feedback_auth)} bytes (expected >= 289)")
            return False
        
        # Try to parse
        params, signature = parse_feedback_auth(feedback_auth)
        
        # Verify signature length
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
