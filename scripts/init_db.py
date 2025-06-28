import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base

DATABASE_PATH = './tmp/avt_demo.db'
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Import your models and BaseMd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from task import BaseMd, TaskMd, AdsbMd  # Make sure this import path is correct



# Example model
# from sqlalchemy import Column, Integer, String
# class User(Base):
#     __tablename__ = 'users'
#     id = Column(Integer, primary_key=True)
#     name = Column(String)

def create_database():
    """Create SQLite3 database file and initialize schema using SQLAlchemy."""
    try:
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        engine = create_engine(DATABASE_URL)
        BaseMd.metadata.create_all(engine)
        print(f"Created or opened SQLite3 database at '{DATABASE_PATH}' and initialized schema.")
    except Exception as e:
        print(f"Error creating SQLite3 database: {e}")
        sys.exit(1)

def main():
    print("Setting up SQLite3 database with SQLAlchemy...")
    create_database()
    print("Database initialized successfully!")

if __name__ == "__main__":
    main()