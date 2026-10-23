"""Durable local storage for sealed-bid commit-reveal nonces.

The ReverseAuction contract's commit phase only stores a hash on-chain; the
bid amount and nonce needed to reveal it later exist nowhere on-chain. If a
provider process loses that nonce between commit and reveal (minutes to
hours later), the commitment becomes permanently unrevealable. This module
persists it to disk so a process restart doesn't forfeit a pending bid.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CommitmentStore:
    """Tracks pending sealed-bid commitments for one provider agent, keyed by auction ID."""

    def __init__(self, agent_id: int, storage_dir: Optional[Path] = None):
        self.agent_id = agent_id
        self.storage_dir = storage_dir or (Path(__file__).parent.parent / "data" / "commitments")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / f"agent_{agent_id}.json"
        self._data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.file_path.exists():
            return {}
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning(f"Commitment store {self.file_path} did not contain a JSON object; ignoring")
                return {}
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load commitment store {self.file_path}: {e}")
            return {}

    def _save(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(self.storage_dir), prefix=".tmp_commitments_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp_path, self.file_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save_commitment(
        self,
        auction_id: int,
        *,
        bid_amount: int,
        nonce_hex: str,
        sender: str,
        effort_tier: Optional[str] = None,
        committed_at: Optional[str] = None,
    ) -> None:
        """Persist a pending commitment. Must be called BEFORE the commitBid tx is sent,
        so a crash right after broadcast still leaves the nonce recoverable on disk."""
        from datetime import datetime

        self._data[str(int(auction_id))] = {
            "auction_id": int(auction_id),
            "bid_amount": int(bid_amount),
            "nonce_hex": nonce_hex,
            "sender": sender,
            "effort_tier": effort_tier,
            "committed_at": committed_at or datetime.utcnow().isoformat() + "Z",
            "revealed": False,
        }
        self._save()

    def mark_revealed(self, auction_id: int) -> None:
        entry = self._data.get(str(int(auction_id)))
        if entry is not None:
            entry["revealed"] = True
            self._save()

    def get_commitment(self, auction_id: int) -> Optional[Dict[str, Any]]:
        return self._data.get(str(int(auction_id)))

    def pending_auction_ids(self) -> List[int]:
        """Auction IDs with a locally stored commitment that hasn't been marked revealed yet."""
        return [
            int(key)
            for key, entry in self._data.items()
            if not entry.get("revealed", False)
        ]

    def remove_commitment(self, auction_id: int) -> None:
        if self._data.pop(str(int(auction_id)), None) is not None:
            self._save()
