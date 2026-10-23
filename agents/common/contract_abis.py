"""
Contract ABI loader for the agentic marketplace.

This module loads compiled contract ABIs from the Foundry output directory
and provides easy access to them for blockchain interactions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Path to the contracts directory
CONTRACTS_DIR = Path(__file__).parent.parent.parent / "contracts"
CONTRACTS_OUT_DIR = CONTRACTS_DIR / "out"


class ContractABIs:
    """
    Utility class to load and manage contract ABIs.
    
    This class loads ABIs from Foundry's compilation output and provides
    methods to access them for different contracts.
    """
    
    def __init__(self):
        """Initialize the contract ABI loader."""
        self._abis: Dict[str, List[Dict[str, Any]]] = {}
        self._bytecodes: Dict[str, str] = {}
        self._loaded = False
    
    def _load_contract_artifact(self, contract_name: str, file_name: Optional[str] = None) -> bool:
        """
        Load a contract artifact from the Foundry output directory.
        
        Args:
            contract_name: Name of the contract (e.g., "ReverseAuction")
            file_name: Optional specific file name if different from contract_name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_name is None:
                file_name = contract_name
            
            # Construct path: out/<FileName>.sol/<ContractName>.json
            artifact_path = CONTRACTS_OUT_DIR / f"{file_name}.sol" / f"{contract_name}.json"
            
            if not artifact_path.exists():
                logger.warning(f"Contract artifact not found: {artifact_path}")
                return False
            
            with open(artifact_path, 'r') as f:
                artifact = json.load(f)
            
            # Extract ABI and bytecode
            self._abis[contract_name] = artifact.get('abi', [])
            
            # Store bytecode for deployment if available
            bytecode = artifact.get('bytecode', {})
            if isinstance(bytecode, dict):
                self._bytecodes[contract_name] = bytecode.get('object', '')
            else:
                self._bytecodes[contract_name] = bytecode
            
            logger.info(f"Loaded ABI for {contract_name} ({len(self._abis[contract_name])} entries)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading contract artifact for {contract_name}: {e}")
            return False
    
    def load_all(self) -> bool:
        """
        Load all required contract ABIs.
        
        Returns:
            True if all contracts loaded successfully, False otherwise
        """
        if self._loaded:
            logger.debug("ABIs already loaded")
            return True
        
        contracts_to_load = [
            ("ReverseAuction", "ReverseAuction"),
            ("IIdentityRegistry", "IIdentityRegistry"),
            ("IReputationRegistry", "IReputationRegistry"),
        ]
        
        success = True
        for contract_name, file_name in contracts_to_load:
            if not self._load_contract_artifact(contract_name, file_name):
                logger.error(f"Failed to load {contract_name}")
                success = False
        
        if success:
            self._loaded = True
            logger.info("All contract ABIs loaded successfully")
        
        return success
    
    def get_abi(self, contract_name: str) -> List[Dict[str, Any]]:
        """
        Get the ABI for a specific contract.
        
        Args:
            contract_name: Name of the contract
            
        Returns:
            List of ABI entries
            
        Raises:
            ValueError: If contract ABI not loaded
        """
        if not self._loaded:
            self.load_all()
        
        if contract_name not in self._abis:
            raise ValueError(f"ABI for {contract_name} not found. Available: {list(self._abis.keys())}")
        
        return self._abis[contract_name]
    
# Global instance for easy access
contract_abis = ContractABIs()


# Convenience functions
def get_reverse_auction_abi() -> List[Dict[str, Any]]:
    """Get the ReverseAuction contract ABI."""
    return contract_abis.get_abi("ReverseAuction")


def get_identity_registry_abi() -> List[Dict[str, Any]]:
    """Get the IIdentityRegistry contract ABI."""
    return contract_abis.get_abi("IIdentityRegistry")


def get_reputation_registry_abi() -> List[Dict[str, Any]]:
    """Get the IReputationRegistry contract ABI."""
    return contract_abis.get_abi("IReputationRegistry")


