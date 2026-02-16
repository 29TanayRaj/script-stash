from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from sqlalchemy.pool import StaticPool

# SQLAlchemy Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Model
class BookDB(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    author = Column(String)
    is_borrowed = Column(Boolean, default=False)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Model for API
class Book(BaseModel):
    id: int
    title: str
    author: str
    is_borrowed: bool = False

    class Config:
        from_attributes = True
        orm_mode = True # For backwards compatibility with v1

app = FastAPI(title="Library Management API")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initial Data (Seeding)
db = SessionLocal()
if not db.query(BookDB).first():
    db.add(BookDB(id=1, title="The Great Gatsby", author="F. Scott Fitzgerald"))
    db.add(BookDB(id=2, title="1984", author="George Orwell"))
    db.commit()
db.close()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Library Management System"}

@app.get("/books", response_model=List[Book])
def get_books(db: Session = Depends(get_db)):
    return db.query(BookDB).all()

@app.get("/books/{book_id}", response_model=Book)
def get_book(book_id: int, db: Session = Depends(get_db)):
    book = db.query(BookDB).filter(BookDB.id == book_id).first()
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.post("/books", response_model=Book, status_code=201)
def create_book(book: Book, db: Session = Depends(get_db)):
    # Check if book with same ID already exists
    db_book = db.query(BookDB).filter(BookDB.id == book.id).first()
    if db_book:
        raise HTTPException(status_code=400, detail="Book with this ID already exists")
    new_book = BookDB(**book.dict())
    db.add(new_book)
    db.commit()
    db.refresh(new_book)
    return new_book

@app.post("/books/{book_id}/borrow", response_model=Book)
def borrow_book(book_id: int, db: Session = Depends(get_db)):
    # Check if book exists
    book = db.query(BookDB).filter(BookDB.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    # Check if book is already borrowed
    if book.is_borrowed:
        raise HTTPException(status_code=400, detail="Book is already borrowed")
    book.is_borrowed = True
    db.commit()
    db.refresh(book)
    return book

@app.post("/books/{book_id}/return", response_model=Book)
def return_book(book_id: int, db: Session = Depends(get_db)):
    # Check if book exists
    book = db.query(BookDB).filter(BookDB.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    # Return the book
    book.is_borrowed = False
    db.commit()
    db.refresh(book)
    return book

@app.delete("/books/{book_id}")
def delete_book(book_id: int, db: Session = Depends(get_db)):
    # Check if book exists
    book = db.query(BookDB).filter(BookDB.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    # Delete the book
    db.delete(book)
    db.commit()
    return {"message": "Book deleted successfully"}

