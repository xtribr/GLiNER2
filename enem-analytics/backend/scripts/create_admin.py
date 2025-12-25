#!/usr/bin/env python3
"""Script to create the initial admin user"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.config import SessionLocal, engine
from database.models import Base, User
from api.auth.service import hash_password


def create_admin(email: str, password: str, nome: str = "Administrador X-TRI"):
    """Create an admin user"""
    # Create tables
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Check if admin exists
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"User with email {email} already exists")
            return

        # Create admin user
        admin = User(
            codigo_inep="00000000",  # Admin doesn't need real INEP
            nome_escola=nome,
            email=email,
            password_hash=hash_password(password),
            is_admin=True,
            is_active=True
        )
        db.add(admin)
        db.commit()
        print(f"Admin user created: {email}")

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_admin.py <email> <password> [nome]")
        print("Example: python create_admin.py admin@xtri.online senha123 'Admin X-TRI'")
        sys.exit(1)

    email = sys.argv[1]
    password = sys.argv[2]
    nome = sys.argv[3] if len(sys.argv) > 3 else "Administrador X-TRI"

    create_admin(email, password, nome)
