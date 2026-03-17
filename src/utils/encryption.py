"""
Encryption utilities for the knowledge library.
Uses Fernet (AES-128-CBC with HMAC) for symmetric encryption.
"""

import base64
import os
from typing import Optional

from cryptography.fernet import Fernet
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("encryption")


class TextEncryptor:
    """Handles encryption/decryption of text chunks."""
    
    def __init__(self, key: Optional[str] = None):
        """
        Initialize with a key.
        
        Args:
            key: Base64-encoded Fernet key. If None, looks for ENCRYPTION_KEY env var.
        """
        if key:
            self.fernet = Fernet(key.encode() if isinstance(key, str) else key)
        else:
            from src.config.settings import APIConfig
            config = APIConfig()
            if not config.encryption_key:
                logger.warning("No encryption key configured - encryption disabled")
                self.fernet = None
            else:
                self.fernet = Fernet(config.encryption_key.encode())
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext text."""
        if not self.fernet:
            return plaintext
        return self.fernet.encrypt(plaintext.encode()).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext text."""
        if not self.fernet:
            return ciphertext
        return self.fernet.decrypt(ciphertext.encode()).decode()
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()


def get_encryptor() -> TextEncryptor:
    """Get the global encryptor instance."""
    return TextEncryptor()
