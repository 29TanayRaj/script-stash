import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test root endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Library Management System"}

# Test get all books
def test_get_books():
    response = client.get("/books")
    assert response.status_code == 200
    assert len(response.json()) >= 2

# Test get single book
def test_get_book():
    response = client.get("/books/1")
    assert response.status_code == 200
    assert response.json()["title"] == "The Great Gatsby"

# Test get non-existent book
def test_get_book_not_found():
    response = client.get("/books/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Book not found"

# Test create book
def test_create_book():
    new_book = {
        "id": 3,
        "title": "The Hobbit",
        "author": "J.R.R. Tolkien",
        "is_borrowed": False
    }
    response = client.post("/books", json=new_book)
    assert response.status_code == 201
    assert response.json()["title"] == "The Hobbit"

    # Verify it was added
    response = client.get("/books/3")
    assert response.status_code == 200

# Test duplicate ID
def test_create_duplicate_book():
    existing_book = {
        "id": 1,
        "title": "Duplicate",
        "author": "Author",
        "is_borrowed": False
    }
    response = client.post("/books", json=existing_book)
    assert response.status_code == 400
    assert response.json()["detail"] == "Book with this ID already exists"

# Test borrow book
def test_borrow_book():
    response = client.post("/books/1/borrow")
    assert response.status_code == 200
    assert response.json()["is_borrowed"] is True

# Test double borrowing
def test_borrow_book_already_borrowed():
    # Attempt to borrow the same book again
    response = client.post("/books/1/borrow")
    assert response.status_code == 400
    assert response.json()["detail"] == "Book is already borrowed"

# Test return book
def test_return_book():
    # First borrow it
    client.post("/books/2/borrow")
    # Then return it
    response = client.post("/books/2/return")
    assert response.status_code == 200
    assert response.json()["is_borrowed"] is False
