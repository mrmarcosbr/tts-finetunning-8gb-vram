"""Generate an RSA key pair and write public/private PEM files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an RSA public/private key pair (PEM).")
    parser.add_argument(
        "--key-size",
        type=int,
        default=2048,
        choices=(2048, 3072, 4096),
        help="RSA modulus size in bits (default: 2048).",
    )
    parser.add_argument(
        "--public-key",
        type=Path,
        default=Path("public_key.pem"),
        help="Output path for the public key (default: public_key.pem).",
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        default=Path("private_key.pem"),
        help="Output path for the private key (default: private_key.pem).",
    )
    parser.add_argument(
        "--passphrase",
        help="Optional passphrase to encrypt the private key PEM.",
    )
    args = parser.parse_args(argv)

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=args.key_size,
    )
    public_key = private_key.public_key()

    if args.passphrase:
        encryption = serialization.BestAvailableEncryption(args.passphrase.encode("utf-8"))
    else:
        encryption = serialization.NoEncryption()

    pem_private = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )
    pem_public = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    args.public_key.write_bytes(pem_public)
    args.private_key.write_bytes(pem_private)

    print(f"Wrote public key:  {args.public_key.resolve()}")
    print(f"Wrote private key: {args.private_key.resolve()}")
    if not args.passphrase:
        print("Warning: private key is unencrypted. Use --passphrase for production.")
    print()
    print(pem_public.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
